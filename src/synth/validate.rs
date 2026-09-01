//! Checks run on a finished row. A failing check drops the row or regenerates
//! the checked column (and what depends on it) up to a retry budget.

use anyhow::{bail, Context, Result};
use minijinja::Environment;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Validator {
    pub column: String,
    pub kind: Kind,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub json_schema: Option<Value>,
    #[serde(default)]
    pub min: Option<usize>,
    #[serde(default)]
    pub max: Option<usize>,
    #[serde(default)]
    pub expression: Option<String>,
    #[serde(default)]
    pub on_fail: OnFail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Regex,
    JsonSchema,
    Length,
    Expression,
}

/// `drop` or `retry:<n>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnFail {
    #[default]
    Drop,
    Retry(usize),
}

impl<'de> Deserialize<'de> for OnFail {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match s.strip_prefix("retry:") {
            _ if s == "drop" => Ok(OnFail::Drop),
            Some(n) => n
                .parse()
                .map(OnFail::Retry)
                .map_err(serde::de::Error::custom),
            None => Err(serde::de::Error::custom(format!(
                "on_fail `{s}`: use drop or retry:<n>"
            ))),
        }
    }
}

/// A validator with its pattern, schema or expression compiled.
pub struct Check {
    pub column: String,
    pub on_fail: OnFail,
    rule: Rule,
}

enum Rule {
    Regex(regex::Regex),
    Schema(jsonschema::Validator),
    Length(Option<usize>, Option<usize>),
    Expression(String),
}

impl Check {
    pub fn compile(v: &Validator) -> Result<Check> {
        let at = |s: &str| format!("validator on `{}`: {s}", v.column);
        let rule = match v.kind {
            Kind::Regex => Rule::Regex(
                regex::Regex::new(
                    v.pattern
                        .as_deref()
                        .with_context(|| at("regex needs `pattern`"))?,
                )
                .with_context(|| at("bad pattern"))?,
            ),
            Kind::JsonSchema => Rule::Schema(
                jsonschema::validator_for(
                    v.json_schema
                        .as_ref()
                        .with_context(|| at("needs `json_schema`"))?,
                )
                .map_err(|e| anyhow::anyhow!(at(&e.to_string())))?,
            ),
            Kind::Length => {
                if v.min.is_none() && v.max.is_none() {
                    bail!(at("length needs `min` or `max`"));
                }
                Rule::Length(v.min, v.max)
            }
            Kind::Expression => Rule::Expression(
                v.expression
                    .clone()
                    .with_context(|| at("needs `expression`"))?,
            ),
        };
        Ok(Check {
            column: v.column.clone(),
            on_fail: v.on_fail,
            rule,
        })
    }

    /// The variables an expression rule reads, for the load-time reference check.
    pub fn references(&self, env: &Environment<'_>) -> Result<Vec<String>> {
        Ok(match &self.rule {
            Rule::Expression(e) => env
                .compile_expression(e)?
                .undeclared_variables(false)
                .into_iter()
                .collect(),
            _ => vec![self.column.clone()],
        })
    }

    /// Whether `row` passes; an error means the check itself could not run.
    pub fn passes(&self, row: &Map<String, Value>, env: &Environment<'_>) -> Result<bool> {
        let cell = row.get(&self.column).unwrap_or(&Value::Null);
        let text = || match cell {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        Ok(match &self.rule {
            Rule::Regex(re) => re.is_match(&text()),
            Rule::Schema(schema) => {
                let parsed = match cell {
                    Value::String(s) => serde_json::from_str(s).ok(),
                    v => Some(v.clone()),
                };
                parsed.is_some_and(|v| schema.is_valid(&v))
            }
            Rule::Length(min, max) => {
                let n = text().chars().count();
                min.is_none_or(|m| n >= m) && max.is_none_or(|m| n <= m)
            }
            Rule::Expression(e) => env
                .compile_expression(e)?
                .eval(row)
                .with_context(|| format!("validator expression `{e}`"))?
                .is_true(),
        })
    }
}
