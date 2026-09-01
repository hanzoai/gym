//! Rows of a dataset. `path` is a local file, a local directory, or a Hugging
//! Face dataset id; files are `.jsonl`, `.json` (an array) or `.parquet`. Rows
//! are read in file order and cut to the requested split slice.

use crate::config::Dataset;
use anyhow::{bail, Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Range;
use std::path::{Path, PathBuf};

pub fn rows(ds: &Dataset) -> Result<Vec<Value>> {
    let split = Split::parse(ds.split.as_deref().unwrap_or("train"))?;
    let mut all = Vec::new();
    for f in files(ds, &split.name)? {
        all.extend(read(&f).with_context(|| f.display().to_string())?);
    }
    let r = split.range(all.len());
    all.truncate(r.end);
    all.drain(..r.start);
    Ok(all)
}

fn files(ds: &Dataset, split: &str) -> Result<Vec<PathBuf>> {
    let p = Path::new(&ds.path);
    if p.is_file() {
        return Ok(vec![p.to_path_buf()]);
    }
    if p.is_dir() {
        let names = match &ds.data_files {
            Some(d) => d.clone(),
            None => {
                let mut v: Vec<String> = std::fs::read_dir(p)?
                    .filter_map(|e| e.ok()?.file_name().into_string().ok())
                    .filter(|n| is_data(n))
                    .collect();
                v.sort();
                v
            }
        };
        return Ok(choose(names, split, ds.data_files.is_some())?
            .iter()
            .map(|n| p.join(n))
            .collect());
    }
    let repo = Api::new()?.repo(Repo::new(ds.path.clone(), RepoType::Dataset));
    let names = match &ds.data_files {
        Some(d) => d.clone(),
        None => repo
            .info()?
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|n| is_data(n))
            .collect(),
    };
    choose(names, split, ds.data_files.is_some())?
        .iter()
        .map(|n| Ok(repo.get(n)?))
        .collect()
}

fn is_data(name: &str) -> bool {
    ["jsonl", "json", "parquet"].contains(
        &Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or(""),
    )
}

/// Files for `split`: those in a directory named after it or whose name
/// mentions it. Files that name no split at all belong to every split.
fn choose(names: Vec<String>, split: &str, explicit: bool) -> Result<Vec<String>> {
    if explicit {
        return Ok(names);
    }
    let mentions = |n: &str, s: &str| {
        Path::new(n)
            .iter()
            .any(|c| c.to_str().is_some_and(|c| c.contains(s)))
    };
    let mut v: Vec<String> = names
        .iter()
        .filter(|n| mentions(n, split))
        .cloned()
        .collect();
    if v.is_empty() {
        v = names
            .into_iter()
            .filter(|n| {
                !["train", "test", "validation", "valid", "dev"]
                    .iter()
                    .any(|s| mentions(n, s))
            })
            .collect();
    }
    if v.is_empty() {
        bail!("no data files for split `{split}`");
    }
    Ok(v)
}

fn read(p: &Path) -> Result<Vec<Value>> {
    match p.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => BufReader::new(File::open(p)?)
            .lines()
            .filter(|l| l.as_ref().is_ok_and(|l| !l.trim().is_empty()))
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect(),
        Some("json") => match serde_json::from_reader(BufReader::new(File::open(p)?))? {
            Value::Array(v) => Ok(v),
            _ => bail!("expected a JSON array"),
        },
        Some("parquet") => SerializedFileReader::new(File::open(p)?)?
            .get_row_iter(None)?
            .map(|r| Ok(r?.to_json_value()))
            .collect(),
        _ => bail!("unsupported file type"),
    }
}

/// A Hugging Face split spec: `name`, `name[a:b]`, with bounds absolute
/// (`1000`) or percent (`20%`, floored) and either side open.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub name: String,
    lo: Bound,
    hi: Bound,
}

#[derive(Debug, Clone, PartialEq)]
enum Bound {
    Open,
    Abs(usize),
    Pct(f64),
}

impl Split {
    pub fn parse(s: &str) -> Result<Split> {
        let (name, slice) = match s.split_once('[') {
            Some((n, rest)) => (
                n,
                Some(
                    rest.strip_suffix(']')
                        .with_context(|| format!("bad split `{s}`"))?,
                ),
            ),
            None => (s, None),
        };
        let (lo, hi) = match slice {
            Some(sl) => {
                let (a, b) = sl
                    .split_once(':')
                    .with_context(|| format!("bad split `{s}`"))?;
                (Bound::parse(a)?, Bound::parse(b)?)
            }
            None => (Bound::Open, Bound::Open),
        };
        Ok(Split {
            name: name.to_string(),
            lo,
            hi,
        })
    }

    pub fn range(&self, n: usize) -> Range<usize> {
        let lo = self.lo.at(n, 0);
        lo..self.hi.at(n, n).max(lo)
    }
}

impl Bound {
    fn parse(s: &str) -> Result<Bound> {
        let s = s.trim();
        Ok(match s.strip_suffix('%') {
            _ if s.is_empty() => Bound::Open,
            Some(p) => Bound::Pct(
                p.parse::<f64>()
                    .with_context(|| format!("bad percent `{s}`"))?,
            ),
            None => Bound::Abs(s.parse().with_context(|| format!("bad index `{s}`"))?),
        })
    }

    fn at(&self, n: usize, open: usize) -> usize {
        match *self {
            Bound::Open => open,
            Bound::Abs(i) => i.min(n),
            Bound::Pct(p) => ((p * n as f64 / 100.0).floor() as usize).min(n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fixture;
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::schema::parser::parse_message_type;
    use std::sync::Arc;

    fn r(spec: &str, n: usize) -> Range<usize> {
        Split::parse(spec).unwrap().range(n)
    }

    #[test]
    fn split_grammar() {
        assert_eq!(Split::parse("train").unwrap().name, "train");
        assert_eq!(r("train", 10), 0..10);
        assert_eq!(r("test", 10), 0..10);
        assert_eq!(r("validation[:3]", 10), 0..3);
        assert_eq!(r("train[:20%]", 10), 0..2);
        assert_eq!(r("train[:1000]", 10), 0..10);
        assert_eq!(r("train[10%:]", 10), 1..10);
        assert_eq!(r("train[100:200]", 150), 100..150);
        assert_eq!(r("train[25%:75%]", 7), 1..5);
        assert_eq!(r("train[33%:]", 100), 33..100);
        assert!(Split::parse("train[1]").is_err());
        assert!(Split::parse("train[a:b]").is_err());
    }

    #[test]
    fn jsonl_and_json_from_directory() {
        let dir = fixture::dir("source");
        std::fs::write(
            dir.join("train.jsonl"),
            "{\"text\":\"a\"}\n\n{\"text\":\"b\"}\n",
        )
        .unwrap();
        std::fs::write(dir.join("test.json"), "[{\"text\":\"t\"}]").unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
        let mut ds = fixture::dataset(
            crate::config::DatasetKind::Completion,
            dir.to_str().unwrap(),
        );
        let texts = |ds: &Dataset| {
            rows(ds)
                .unwrap()
                .iter()
                .map(|v| v["text"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(texts(&ds), ["a", "b"]);
        ds.split = Some("test".into());
        assert_eq!(texts(&ds), ["t"]);
        ds.split = Some("validation".into());
        assert!(rows(&ds).is_err());
        ds.split = Some("train[1:]".into());
        ds.data_files = Some(vec!["train.jsonl".into(), "test.json".into()]);
        assert_eq!(texts(&ds), ["b", "t"]);
        ds.split = None;
        ds.data_files = None;
        ds.path = dir.join("test.json").to_str().unwrap().into();
        assert_eq!(texts(&ds), ["t"]);
    }

    #[test]
    fn parquet_rows_with_nested_messages() {
        let dir = fixture::dir("parquet");
        let path = dir.join("train.parquet");
        let schema = Arc::new(
            parse_message_type(
                "message row {
                    required binary text (UTF8);
                    optional group messages (LIST) {
                        repeated group list {
                            required group element {
                                required binary role (UTF8);
                                required binary content (UTF8);
                            }
                        }
                    }
                }",
            )
            .unwrap(),
        );
        let mut w = SerializedFileWriter::new(
            File::create(&path).unwrap(),
            schema,
            Arc::new(WriterProperties::builder().build()),
        )
        .unwrap();
        let mut g = w.next_row_group().unwrap();
        let col = |g: &mut SerializedRowGroupWriter<'_, File>,
                   vals: &[&str],
                   levels: Option<(&[i16], &[i16])>| {
            let vals: Vec<ByteArray> = vals
                .iter()
                .map(|s| ByteArray::from(s.as_bytes().to_vec()))
                .collect();
            let mut c = g.next_column().unwrap().unwrap();
            c.typed::<ByteArrayType>()
                .write_batch(&vals, levels.map(|l| l.0), levels.map(|l| l.1))
                .unwrap();
            c.close().unwrap();
        };
        let nested = Some((&[2i16, 2, 2, 2][..], &[0i16, 1, 0, 1][..]));
        col(&mut g, &["hi there", "fine thanks"], None);
        col(&mut g, &["user", "assistant", "user", "assistant"], nested);
        col(
            &mut g,
            &["hi there", "fine thanks", "how are you", "you too"],
            nested,
        );
        g.close().unwrap();
        w.close().unwrap();

        let v = read(&path).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["text"], "hi there");
        assert_eq!(v[1]["messages"][1]["role"], "assistant");
        assert_eq!(v[1]["messages"][1]["content"], "you too");
        assert_eq!(v[0]["messages"].as_array().unwrap().len(), 2);
    }
}
