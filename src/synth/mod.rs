//! `gym synth design.yml`: a synthetic dataset from a column design, after
//! NVIDIA NeMo Data Designer. A row is built column by column in dependency
//! order — statistical samplers, Jinja expressions and LLM calls whose prompts
//! are templates over the row so far — then checked by validators that drop it
//! or regenerate the checked column. Rows are built concurrently and written as
//! they finish.

pub mod llm;
pub mod sample;
#[cfg(test)]
mod tests;
pub mod validate;
pub mod write;

use anyhow::{bail, Context, Result};
use llm::Client;
use minijinja::{Environment, UndefinedBehavior};
use rand::seq::SliceRandom;
use rand::Rng;
use sample::Sampler;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc;
use validate::{Check, OnFail, Validator};
use write::{Row, Sink};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Design {
    #[serde(default)]
    pub model: Option<Model>,
    #[serde(default)]
    pub seed: Option<Seed>,
    #[serde(default = "d_rows")]
    pub rows: usize,
    #[serde(default)]
    pub output: Option<String>,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub validators: Vec<Validator>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(default = "d_endpoint")]
    pub endpoint: String,
    #[serde(default = "d_key")]
    pub api_key_env: String,
    pub name: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default = "d_concurrency")]
    pub concurrency: usize,
}

/// Rows of an existing dataset (any path `datasets:` accepts) whose columns
/// templates can read; `ordered` cycles through it, `shuffle` reshuffles each pass.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seed {
    pub path: String,
    #[serde(default)]
    pub sampling: Sampling,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    #[default]
    Ordered,
    Shuffle,
}

fn d_rows() -> usize {
    10
}
fn d_endpoint() -> String {
    "https://api.hanzo.ai/v1".into()
}
fn d_key() -> String {
    "HANZO_API_KEY".into()
}
fn d_concurrency() -> usize {
    4
}

#[derive(Debug)]
pub struct Column {
    pub name: String,
    pub kind: Kind,
}

#[derive(Debug)]
pub enum Kind {
    Sample(Sampler),
    Llm(Llm),
    /// A Jinja template over the row, stored as its rendered text.
    Expression(String),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "llm", rename_all = "snake_case", deny_unknown_fields)]
pub enum Llm {
    Text {
        prompt: String,
        #[serde(default)]
        system: Option<String>,
    },
    /// A JSON object matching `json_schema`, stored nested under the column name.
    Structured {
        prompt: String,
        #[serde(default)]
        system: Option<String>,
        json_schema: Value,
    },
    /// An integer score between the lowest and highest rubric key, stored under
    /// the column name, with the model's reasoning under `<name>_reasoning`.
    Judge {
        prompt: String,
        #[serde(default)]
        system: Option<String>,
        rubric: BTreeMap<i64, String>,
        columns: Vec<String>,
    },
}

impl Llm {
    fn prompt(&self) -> &str {
        match self {
            Llm::Text { prompt, .. }
            | Llm::Structured { prompt, .. }
            | Llm::Judge { prompt, .. } => prompt,
        }
    }
    fn system(&self) -> Option<&str> {
        match self {
            Llm::Text { system, .. }
            | Llm::Structured { system, .. }
            | Llm::Judge { system, .. } => system.as_deref(),
        }
    }
}

/// A column is `name` plus exactly one of `sampler`, `llm` or `expression`.
impl<'de> Deserialize<'de> for Column {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Expr {
            expression: String,
        }
        let mut m = serde_yaml::Mapping::deserialize(d)?;
        let Some(serde_yaml::Value::String(name)) = m.remove("name") else {
            return Err(D::Error::missing_field("name"));
        };
        let has = |k: &str| m.contains_key(k);
        let (sampler, llm, expr) = (has("sampler"), has("llm"), has("expression"));
        let v = serde_yaml::Value::Mapping(m);
        let kind = match (sampler, llm, expr) {
            (true, false, false) => Sampler::deserialize(v).map(Kind::Sample),
            (false, true, false) => Llm::deserialize(v).map(Kind::Llm),
            (false, false, true) => Expr::deserialize(v).map(|e| Kind::Expression(e.expression)),
            _ => {
                return Err(D::Error::custom(format!(
                    "column `{name}`: give one of sampler, llm, expression"
                )))
            }
        }
        .map_err(|e| D::Error::custom(format!("column `{name}`: {e}")))?;
        Ok(Column { name, kind })
    }
}

/// A design checked and compiled: templates, generation order, the reverse
/// dependency graph for retries, validators, seed rows and the model client.
pub struct Plan {
    design: Design,
    env: Environment<'static>,
    order: Vec<usize>,
    users: Vec<Vec<usize>>,
    checks: Vec<Check>,
    seed: Vec<Row>,
    client: Option<Client>,
}

enum Outcome {
    Row(Row),
    Dropped,
    Failed,
}

#[derive(Default)]
struct Stats {
    rows: usize,
    dropped: usize,
    failed: usize,
}

const ATTEMPTS: usize = 3;
const JUDGE: &str = "You are a strict, impartial judge. Score only against the rubric.";

pub fn run(design: &str, preview: Option<usize>) -> Result<()> {
    let plan = Plan::load(design)?;
    let (n, sink) = match preview {
        Some(n) => (n, Sink::Stdout),
        None => {
            let out = plan
                .design
                .output
                .as_deref()
                .context("`output` is required unless --preview is given")?;
            (plan.design.rows, Sink::open(out)?)
        }
    };
    plan.generate(n, sink)
}

impl Plan {
    pub fn load(path: &str) -> Result<Plan> {
        let text = std::fs::read_to_string(path).with_context(|| path.to_string())?;
        let design: Design = serde_yaml::from_str(&text).with_context(|| path.to_string())?;
        Plan::new(design)
    }

    pub fn new(design: Design) -> Result<Plan> {
        let seed = match &design.seed {
            Some(s) => {
                let ds = serde_yaml::from_value(serde_yaml::to_value(
                    json!({"path": s.path, "type": "completion"}),
                )?)?;
                crate::data::source::rows(&ds)
                    .with_context(|| format!("seed `{}`", s.path))?
                    .into_iter()
                    .map(|v| match v {
                        Value::Object(m) => Ok(m),
                        v => bail!("seed `{}`: row is not an object: {v}", s.path),
                    })
                    .collect::<Result<Vec<Row>>>()?
            }
            None => Vec::new(),
        };
        if design.seed.is_some() && seed.is_empty() {
            bail!("seed dataset is empty");
        }
        let given: BTreeSet<&String> = seed.iter().flat_map(|r| r.keys()).collect();
        let cols = &design.columns;
        let index = |name: &str| cols.iter().position(|c| c.name == name);
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); cols.len()];
        for (i, c) in cols.iter().enumerate() {
            if cols[..i].iter().any(|o| o.name == c.name) || given.contains(&c.name) {
                bail!("column `{}` is defined twice", c.name);
            }
            let at = |s: String| format!("column `{}`: {s}", c.name);
            let mut refs = BTreeSet::new();
            let mut template = |key: String, src: &str| -> Result<()> {
                env.add_template_owned(key.clone(), src.to_string())
                    .map_err(|e| anyhow::anyhow!(at(e.to_string())))?;
                refs.extend(env.get_template(&key)?.undeclared_variables(false));
                Ok(())
            };
            match &c.kind {
                Kind::Sample(s) => {
                    let parent = match s {
                        Sampler::Subcategory { parent, .. } => Some(parent.clone()),
                        _ => None,
                    };
                    let ps = parent.as_deref().and_then(index).map(|p| &cols[p].kind);
                    s.check(match ps {
                        Some(Kind::Sample(p)) => Some(p),
                        _ => None,
                    })
                    .map_err(|e| anyhow::anyhow!(at(e.to_string())))?;
                    refs.extend(parent);
                }
                Kind::Expression(src) => template(c.name.clone(), src)?,
                Kind::Llm(l) => {
                    template(format!("{}.prompt", c.name), l.prompt())?;
                    if let Some(s) = l.system() {
                        template(format!("{}.system", c.name), s)?;
                    }
                    match l {
                        Llm::Structured { json_schema, .. } => {
                            jsonschema::validator_for(json_schema)
                                .map_err(|e| anyhow::anyhow!(at(e.to_string())))?;
                        }
                        Llm::Judge {
                            rubric, columns, ..
                        } => {
                            if rubric.is_empty() || columns.is_empty() {
                                bail!(at("judge needs a rubric and columns".into()));
                            }
                            refs.extend(columns.iter().cloned());
                        }
                        Llm::Text { .. } => {}
                    }
                    if refs.is_empty() {
                        tracing::warn!(
                            column = c.name,
                            "prompt references no column: every row gets the same prompt"
                        );
                    }
                }
            }
            for r in refs {
                match index(&r) {
                    Some(j) if j == i => bail!(at("references itself".into())),
                    Some(j) => deps[i].push(j),
                    None if given.contains(&r) => {}
                    None => bail!(at(format!("references `{r}`, which is not a column"))),
                }
            }
        }
        let order = order(&deps).map_err(|cycle| {
            let names: Vec<&str> = cycle.iter().map(|&i| cols[i].name.as_str()).collect();
            anyhow::anyhow!(
                "columns depend on each other in a cycle: {}",
                names.join(", ")
            )
        })?;
        let mut users = vec![Vec::new(); cols.len()];
        for (i, d) in deps.iter().enumerate() {
            for &j in d {
                users[j].push(i);
            }
        }
        let mut checks = Vec::new();
        for v in &design.validators {
            let check = Check::compile(v)?;
            if index(&check.column).is_none() {
                bail!("validator on `{}`: not a column", check.column);
            }
            for r in check.references(&env)? {
                if index(&r).is_none() && !given.contains(&r) {
                    bail!(
                        "validator on `{}`: references `{r}`, which is not a column",
                        check.column
                    );
                }
            }
            checks.push(check);
        }
        let client = match (
            &design.model,
            cols.iter().any(|c| matches!(c.kind, Kind::Llm(_))),
        ) {
            (_, false) => None,
            (None, true) => bail!("llm columns need `model`"),
            (Some(m), true) => {
                let key = std::env::var(&m.api_key_env).with_context(|| {
                    format!("model: `{}` is not set (api_key_env)", m.api_key_env)
                })?;
                Some(Client::new(&m.endpoint, key, m.name.clone(), m.temperature))
            }
        };
        Ok(Plan {
            design,
            env,
            order,
            users,
            checks,
            seed,
            client,
        })
    }

    /// Column names in generation order.
    pub fn order(&self) -> Vec<&str> {
        self.order
            .iter()
            .map(|&i| self.design.columns[i].name.as_str())
            .collect()
    }

    /// Build `n` rows with the model's concurrency and write them to `sink`.
    pub fn generate(&self, n: usize, mut sink: Sink) -> Result<()> {
        let seeds = self.seeds(n);
        let workers = self
            .design
            .model
            .as_ref()
            .map_or(1, |m| m.concurrency)
            .clamp(1, n.max(1));
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let (tx, rx) = mpsc::channel();
        let (next, stop, seeds) = (&next, &stop, &seeds);
        std::thread::scope(|s| {
            for _ in 0..workers {
                let tx = tx.clone();
                s.spawn(move || {
                    let mut rng = rand::rng();
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= n || stop.load(Relaxed) {
                            break;
                        }
                        let out = self.row(seeds.get(i).map(|&k| &self.seed[k]), &mut rng);
                        if tx.send(out).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut stats = Stats::default();
            let result = (1..=n).try_for_each(|done| -> Result<()> {
                match rx.recv().context("every worker stopped")?? {
                    Outcome::Row(r) => {
                        sink.write(r)?;
                        stats.rows += 1;
                    }
                    Outcome::Dropped => stats.dropped += 1,
                    Outcome::Failed => stats.failed += 1,
                }
                if done % 50 == 0 || done == n {
                    self.log(&stats);
                }
                Ok(())
            });
            stop.store(true, Relaxed);
            drop(rx);
            result
        })?;
        sink.close()
    }

    fn log(&self, s: &Stats) {
        let (calls, tokens) = self
            .client
            .as_ref()
            .map_or((0, 0), |c| (c.calls.load(Relaxed), c.tokens.load(Relaxed)));
        tracing::info!(
            rows = s.rows,
            dropped = s.dropped,
            failed = s.failed,
            calls,
            tokens,
            "synth"
        );
    }

    /// Which seed row each of the `n` rows starts from.
    fn seeds(&self, n: usize) -> Vec<usize> {
        let len = self.seed.len();
        if len == 0 {
            return Vec::new();
        }
        match self.design.seed.as_ref().map(|s| s.sampling) {
            Some(Sampling::Shuffle) => {
                let mut rng = rand::rng();
                let mut all = Vec::with_capacity(n + len);
                while all.len() < n {
                    let mut pass: Vec<usize> = (0..len).collect();
                    pass.shuffle(&mut rng);
                    all.extend(pass);
                }
                all.truncate(n);
                all
            }
            _ => (0..n).map(|i| i % len).collect(),
        }
    }

    fn row(&self, seed: Option<&Row>, rng: &mut impl Rng) -> Result<Outcome> {
        let mut row = seed.cloned().unwrap_or_default();
        if !self.fill(&mut row, &self.order, rng)? {
            return Ok(Outcome::Failed);
        }
        let mut left: Vec<usize> = self
            .checks
            .iter()
            .map(|c| match c.on_fail {
                OnFail::Retry(n) => n,
                OnFail::Drop => 0,
            })
            .collect();
        'again: loop {
            for (k, check) in self.checks.iter().enumerate() {
                if check.passes(&row, &self.env)? {
                    continue;
                }
                if left[k] == 0 {
                    return Ok(Outcome::Dropped);
                }
                left[k] -= 1;
                let again = self.downstream(&check.column);
                if !self.fill(&mut row, &again, rng)? {
                    return Ok(Outcome::Failed);
                }
                continue 'again;
            }
            return Ok(Outcome::Row(row));
        }
    }

    /// `name` and every column that depends on it, in generation order.
    fn downstream(&self, name: &str) -> Vec<usize> {
        let start = self
            .design
            .columns
            .iter()
            .position(|c| c.name == name)
            .expect("checked at load");
        let mut set = BTreeSet::from([start]);
        let mut todo = vec![start];
        while let Some(i) = todo.pop() {
            for &u in &self.users[i] {
                if set.insert(u) {
                    todo.push(u);
                }
            }
        }
        self.order
            .iter()
            .copied()
            .filter(|i| set.contains(i))
            .collect()
    }

    /// Generate `cols` into `row`; false when a model never produced usable output.
    fn fill(&self, row: &mut Row, cols: &[usize], rng: &mut impl Rng) -> Result<bool> {
        for &i in cols {
            let c = &self.design.columns[i];
            let v = match &c.kind {
                Kind::Sample(s) => s
                    .draw(row, rng)
                    .with_context(|| format!("column `{}`", c.name))?,
                Kind::Expression(_) => Value::from(self.render(&c.name, row)?),
                Kind::Llm(l) => {
                    if !self.cell(&c.name, l, row)? {
                        return Ok(false);
                    }
                    continue;
                }
            };
            row.insert(c.name.clone(), v);
        }
        Ok(true)
    }

    fn render(&self, key: &str, row: &Row) -> Result<String> {
        self.env
            .get_template(key)?
            .render(row)
            .with_context(|| format!("template `{key}`"))
    }

    fn cell(&self, name: &str, llm: &Llm, row: &mut Row) -> Result<bool> {
        let client = self.client.as_ref().expect("llm columns compile a client");
        let system = llm
            .system()
            .map(|_| self.render(&format!("{name}.system"), row))
            .transpose()?;
        let user = self.render(&format!("{name}.prompt"), row)?;
        match llm {
            Llm::Text { .. } => {
                let text = client.chat(system.as_deref(), &user, None)?;
                row.insert(name.to_string(), Value::from(text));
                Ok(true)
            }
            Llm::Structured { json_schema, .. } => {
                let schema = jsonschema::validator_for(json_schema).expect("checked at load");
                let user = format!(
                    "{user}\n\nRespond with one JSON object matching this JSON Schema, inside a ```json fence:\n{json_schema}"
                );
                let format = llm::format(name, json_schema);
                for _ in 0..ATTEMPTS {
                    let text = client.chat(system.as_deref(), &user, Some(format.clone()))?;
                    match llm::object(&text) {
                        Ok(v) if schema.is_valid(&v) => {
                            row.insert(name.to_string(), v);
                            return Ok(true);
                        }
                        Ok(_) => tracing::warn!(column = name, "output does not match json_schema"),
                        Err(e) => tracing::warn!(column = name, %e, "unusable output"),
                    }
                }
                Ok(false)
            }
            Llm::Judge {
                rubric, columns, ..
            } => {
                let (lo, hi) = (
                    *rubric.keys().next().unwrap(),
                    *rubric.keys().last().unwrap(),
                );
                let mut user = user;
                for c in columns {
                    let v = row
                        .get(c)
                        .with_context(|| format!("judge `{name}`: no `{c}` in row"))?;
                    write!(user, "\n\n{c}:\n{}", sample::text(v)).unwrap();
                }
                user.push_str("\n\nRubric:\n");
                for (k, d) in rubric {
                    writeln!(user, "{k}: {d}").unwrap();
                }
                write!(
                    user,
                    "\nRespond with one JSON object {{\"score\": <integer {lo}-{hi}>, \"reasoning\": \"<why>\"}} inside a ```json fence."
                )
                .unwrap();
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": lo, "maximum": hi},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"]
                });
                let format = llm::format(name, &schema);
                for _ in 0..ATTEMPTS {
                    let text = client.chat(
                        system.as_deref().or(Some(JUDGE)),
                        &user,
                        Some(format.clone()),
                    )?;
                    match llm::verdict(&text, lo, hi) {
                        Ok((score, reasoning)) => {
                            row.insert(name.to_string(), Value::from(score));
                            row.insert(format!("{name}_reasoning"), Value::from(reasoning));
                            return Ok(true);
                        }
                        Err(e) => tracing::warn!(column = name, %e, "unusable verdict"),
                    }
                }
                Ok(false)
            }
        }
    }
}

/// Kahn's algorithm, taking the earliest declared ready column first; a cycle
/// comes back as the columns left in it.
fn order(deps: &[Vec<usize>]) -> Result<Vec<usize>, Vec<usize>> {
    let mut left: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut ready: BTreeSet<usize> = (0..deps.len()).filter(|&i| left[i] == 0).collect();
    let mut out = Vec::with_capacity(deps.len());
    while let Some(i) = ready.pop_first() {
        out.push(i);
        for (j, d) in deps.iter().enumerate() {
            if d.contains(&i) {
                left[j] -= 1;
                if left[j] == 0 {
                    ready.insert(j);
                }
            }
        }
    }
    if out.len() == deps.len() {
        Ok(out)
    } else {
        Err((0..deps.len()).filter(|&i| left[i] > 0).collect())
    }
}
