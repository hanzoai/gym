//! Where finished rows go: appended to a `.jsonl` file as they arrive, buffered
//! into row groups of a `.parquet` file, or printed to stdout for a preview.

use anyhow::{bail, Context, Result};
use parquet::basic::{LogicalType, Repetition, Type as Physical};
use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use serde_json::{Map, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

pub type Row = Map<String, Value>;

const GROUP: usize = 512;

pub enum Sink {
    Stdout,
    Jsonl(BufWriter<File>),
    Parquet(Box<Parquet>),
}

impl Sink {
    pub fn open(path: &str) -> Result<Sink> {
        let p = Path::new(path);
        if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        Ok(match p.extension().and_then(|e| e.to_str()) {
            Some("jsonl") => Sink::Jsonl(BufWriter::new(File::create(p)?)),
            Some("parquet") => Sink::Parquet(Box::new(Parquet {
                file: None,
                path: path.to_string(),
                rows: Vec::new(),
            })),
            _ => bail!("output `{path}`: use a .jsonl or .parquet path"),
        })
    }

    pub fn write(&mut self, row: Row) -> Result<()> {
        match self {
            Sink::Stdout => println!("{}", serde_json::to_string_pretty(&row)?),
            Sink::Jsonl(w) => {
                serde_json::to_writer(&mut *w, &row)?;
                w.write_all(b"\n")?;
                w.flush()?;
            }
            Sink::Parquet(p) => {
                p.rows.push(row);
                if p.rows.len() >= GROUP {
                    p.flush()?;
                }
            }
        }
        Ok(())
    }

    pub fn close(self) -> Result<()> {
        match self {
            Sink::Stdout => Ok(()),
            Sink::Jsonl(mut w) => Ok(w.flush()?),
            Sink::Parquet(mut p) => {
                if !p.rows.is_empty() || p.file.is_none() {
                    p.flush()?;
                }
                p.file.take().context("parquet writer")?.close()?;
                Ok(())
            }
        }
    }
}

/// A parquet file being written one row group at a time; the schema is fixed
/// by the first group.
pub struct Parquet {
    file: Option<SerializedFileWriter<File>>,
    path: String,
    rows: Vec<Row>,
}

/// Column types a parquet file can hold; JSON objects and arrays go in as text.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Bool,
    Int,
    Float,
    Text,
}

type Columns = Vec<(String, Kind)>;

fn kind(v: &Value) -> Option<Kind> {
    Some(match v {
        Value::Null => return None,
        Value::Bool(_) => Kind::Bool,
        Value::Number(n) if n.is_i64() => Kind::Int,
        Value::Number(_) => Kind::Float,
        _ => Kind::Text,
    })
}

fn text(v: &Value) -> ByteArray {
    match v {
        Value::String(s) => s.as_bytes().into(),
        v => v.to_string().into_bytes().into(),
    }
}

/// One optional column per key, typed by the values seen: ints widen to double
/// when floats share the column, anything else mixed becomes text.
fn schema(rows: &[Row]) -> Result<(Arc<Type>, Columns)> {
    let mut cols: Columns = Vec::new();
    for row in rows {
        for (k, v) in row {
            let Some(kv) = kind(v) else { continue };
            match cols.iter_mut().find(|(n, _)| n == k) {
                None => cols.push((k.clone(), kv)),
                Some((_, kc)) => match (*kc, kv) {
                    (a, b) if a == b => {}
                    (Kind::Int, Kind::Float) => *kc = Kind::Float,
                    (Kind::Float, Kind::Int) => {}
                    _ => *kc = Kind::Text,
                },
            }
        }
    }
    let fields = cols
        .iter()
        .map(|(name, k)| {
            let (phys, logical) = match k {
                Kind::Bool => (Physical::BOOLEAN, None),
                Kind::Int => (Physical::INT64, None),
                Kind::Float => (Physical::DOUBLE, None),
                Kind::Text => (Physical::BYTE_ARRAY, Some(LogicalType::String)),
            };
            Ok(Arc::new(
                Type::primitive_type_builder(name, phys)
                    .with_repetition(Repetition::OPTIONAL)
                    .with_logical_type(logical)
                    .build()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let root = Type::group_type_builder("row")
        .with_fields(fields)
        .build()?;
    Ok((Arc::new(root), cols))
}

impl Parquet {
    fn flush(&mut self) -> Result<()> {
        let cols: Columns = match &self.file {
            Some(w) => w
                .schema_descr()
                .columns()
                .iter()
                .map(|c| {
                    let k = match c.physical_type() {
                        Physical::BOOLEAN => Kind::Bool,
                        Physical::INT64 => Kind::Int,
                        Physical::DOUBLE => Kind::Float,
                        _ => Kind::Text,
                    };
                    (c.name().to_string(), k)
                })
                .collect(),
            None => {
                let (root, cols) = schema(&self.rows)?;
                let props = Arc::new(WriterProperties::builder().build());
                self.file = Some(SerializedFileWriter::new(
                    File::create(&self.path)?,
                    root,
                    props,
                )?);
                cols
            }
        };
        let w = self.file.as_mut().context("parquet writer")?;
        let mut group = w.next_row_group()?;
        for (name, k) in &cols {
            let cells: Vec<Option<&Value>> = self
                .rows
                .iter()
                .map(|r| r.get(name).filter(|v| !v.is_null()))
                .collect();
            let def: Vec<i16> = cells.iter().map(|c| c.is_some() as i16).collect();
            let present = cells.iter().flatten().copied();
            let mut col = group.next_column()?.context("column count")?;
            let bad = || {
                anyhow::anyhow!(
                    "parquet column `{name}`: value does not fit the type of the first rows"
                )
            };
            match k {
                Kind::Bool => {
                    let v = present
                        .map(|v| v.as_bool().ok_or_else(bad))
                        .collect::<Result<Vec<_>>>()?;
                    col.typed::<BoolType>().write_batch(&v, Some(&def), None)?;
                }
                Kind::Int => {
                    let v = present
                        .map(|v| v.as_i64().ok_or_else(bad))
                        .collect::<Result<Vec<_>>>()?;
                    col.typed::<Int64Type>().write_batch(&v, Some(&def), None)?;
                }
                Kind::Float => {
                    let v = present
                        .map(|v| v.as_f64().ok_or_else(bad))
                        .collect::<Result<Vec<_>>>()?;
                    col.typed::<DoubleType>()
                        .write_batch(&v, Some(&def), None)?;
                }
                Kind::Text => {
                    let v: Vec<ByteArray> = present.map(text).collect();
                    col.typed::<ByteArrayType>()
                        .write_batch(&v, Some(&def), None)?;
                }
            }
            col.close()?;
        }
        group.close()?;
        self.rows.clear();
        Ok(())
    }
}
