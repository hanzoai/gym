//! Statistical columns. Each sampler draws one value for a row from its
//! parameters and, for `subcategory`, from the parent's value in that row.

use anyhow::{bail, Context, Result};
use chrono::format::{Item, StrftimeItems};
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand_distr::Normal;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "sampler", rename_all = "snake_case", deny_unknown_fields)]
pub enum Sampler {
    /// One of `values`; `weights` are relative and need not sum to one.
    Category {
        values: Vec<Value>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    /// One of `values[parent value]`, uniformly.
    Subcategory {
        parent: String,
        values: BTreeMap<String, Vec<Value>>,
    },
    /// Uniform over `[low, high)`; `integer` makes it `low..=high` in whole numbers.
    Uniform {
        low: f64,
        high: f64,
        #[serde(default)]
        integer: bool,
    },
    /// Normal with `mean` and `std`; draws outside `min..=max` are rejected and redrawn.
    Gaussian {
        mean: f64,
        std: f64,
        #[serde(default)]
        min: Option<f64>,
        #[serde(default)]
        max: Option<f64>,
    },
    /// 32 hex characters of a random UUID.
    Uuid {},
    /// A moment uniform over the seconds from `start` (inclusive) to `end` (exclusive), as `format`.
    Datetime {
        start: String,
        end: String,
        #[serde(default = "d_format")]
        format: String,
    },
    /// 1 with probability `p`, else 0.
    Bernoulli { p: f64 },
}

fn d_format() -> String {
    "%Y-%m-%dT%H:%M:%S".into()
}

/// A value as it is matched against subcategory keys: strings bare, the rest as JSON.
pub fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// `YYYY-MM-DD`, optionally followed by `THH:MM:SS` or ` HH:MM:SS`.
fn stamp(s: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d").map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        })
        .with_context(|| format!("`{s}`: not a date or datetime"))
}

impl Sampler {
    /// Parameter checks that do not need a row; `parent` is the parent column's sampler when there is one.
    pub fn check(&self, parent: Option<&Sampler>) -> Result<()> {
        match self {
            Sampler::Category { values, weights } => {
                if values.is_empty() {
                    bail!("category needs at least one value");
                }
                if let Some(w) = weights {
                    if w.len() != values.len() {
                        bail!("{} weights for {} values", w.len(), values.len());
                    }
                    if w.iter().any(|x| *x < 0.0 || !x.is_finite()) || w.iter().sum::<f64>() <= 0.0
                    {
                        bail!("weights must be non-negative and sum to more than zero");
                    }
                }
            }
            Sampler::Subcategory {
                parent: name,
                values,
            } => {
                let Some(Sampler::Category { values: pv, .. }) = parent else {
                    bail!("parent `{name}` must be a category sampler column");
                };
                let want: Vec<String> = pv.iter().map(text).collect();
                let missing: Vec<&String> =
                    want.iter().filter(|k| !values.contains_key(*k)).collect();
                let extra: Vec<&String> = values.keys().filter(|k| !want.contains(k)).collect();
                if !missing.is_empty() || !extra.is_empty() {
                    bail!("values must cover parent `{name}` exactly: missing {missing:?}, unknown {extra:?}");
                }
                if let Some((k, _)) = values.iter().find(|(_, v)| v.is_empty()) {
                    bail!("no values for parent value `{k}`");
                }
            }
            Sampler::Uniform { low, high, .. } => {
                if high.partial_cmp(low) != Some(Ordering::Greater) {
                    bail!("uniform needs high > low");
                }
            }
            Sampler::Gaussian { std, min, max, .. } => {
                if std.partial_cmp(&0.0) != Some(Ordering::Greater) {
                    bail!("gaussian needs std > 0");
                }
                if let (Some(a), Some(b)) = (min, max) {
                    if a > b {
                        bail!("gaussian needs min <= max");
                    }
                }
            }
            Sampler::Uuid {} => {}
            Sampler::Datetime { start, end, format } => {
                if stamp(start)? >= stamp(end)? {
                    bail!("datetime needs start < end");
                }
                if StrftimeItems::new(format).any(|i| i == Item::Error) {
                    bail!("bad datetime format `{format}`");
                }
            }
            Sampler::Bernoulli { p } => {
                if !(0.0..=1.0).contains(p) {
                    bail!("bernoulli needs p in 0..=1");
                }
            }
        }
        Ok(())
    }

    pub fn draw(&self, row: &Map<String, Value>, rng: &mut impl Rng) -> Result<Value> {
        Ok(match self {
            Sampler::Category { values, weights } => match weights {
                Some(w) => values[WeightedIndex::new(w)?.sample(rng)].clone(),
                None => values.choose(rng).unwrap().clone(),
            },
            Sampler::Subcategory { parent, values } => {
                let key = text(
                    row.get(parent)
                        .with_context(|| format!("no `{parent}` in row"))?,
                );
                let v = values
                    .get(&key)
                    .with_context(|| format!("no values for `{parent}` = `{key}`"))?;
                v.choose(rng).unwrap().clone()
            }
            Sampler::Uniform {
                low,
                high,
                integer: true,
            } => Value::from(rng.random_range(low.ceil() as i64..=high.floor() as i64)),
            Sampler::Uniform { low, high, .. } => Value::from(rng.random_range(*low..*high)),
            Sampler::Gaussian {
                mean,
                std,
                min,
                max,
            } => {
                let normal = Normal::new(*mean, *std)?;
                let inside = |x: f64| min.is_none_or(|m| x >= m) && max.is_none_or(|m| x <= m);
                let x = (0..1000)
                    .map(|_| normal.sample(rng))
                    .find(|x| inside(*x))
                    .context("gaussian: 1000 draws fell outside min..=max")?;
                Value::from(x)
            }
            Sampler::Uuid {} => {
                let mut b = [0u8; 16];
                rng.fill_bytes(&mut b);
                b[6] = (b[6] & 0x0f) | 0x40;
                b[8] = (b[8] & 0x3f) | 0x80;
                Value::from(b.iter().map(|x| format!("{x:02x}")).collect::<String>())
            }
            Sampler::Datetime { start, end, format } => {
                let (a, b) = (
                    stamp(start)?.and_utc().timestamp(),
                    stamp(end)?.and_utc().timestamp(),
                );
                let t = DateTime::from_timestamp(rng.random_range(a..b), 0).unwrap();
                Value::from(t.format(format).to_string())
            }
            Sampler::Bernoulli { p } => Value::from(rng.random_bool(*p) as i64),
        })
    }
}
