//! Chat completions over any OpenAI-compatible endpoint, with retry on 429
//! and 5xx, call and token counters, and the JSON extraction that structured
//! and judge columns rely on when a server ignores `response_format`.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

pub struct Client {
    agent: ureq::Agent,
    url: String,
    key: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub calls: AtomicU64,
    pub tokens: AtomicU64,
}

#[derive(Deserialize)]
struct Completion {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    total_tokens: u64,
}

const ATTEMPTS: u32 = 3;

impl Client {
    pub fn new(endpoint: &str, key: String, model: String, temperature: Option<f64>) -> Client {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(300)))
            .build()
            .new_agent();
        Client {
            agent,
            url: format!("{}/chat/completions", endpoint.trim_end_matches('/')),
            key,
            model,
            temperature,
            calls: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
        }
    }

    /// One completion; `format` is an optional `response_format` object.
    pub fn chat(&self, system: Option<&str>, user: &str, format: Option<Value>) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(s) = system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.push(json!({"role": "user", "content": user}));
        let mut body = json!({ "model": self.model, "messages": messages });
        if let Some(t) = self.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(f) = format {
            body["response_format"] = f;
        }
        let mut last = None;
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(500 << attempt));
            }
            self.calls.fetch_add(1, Relaxed);
            let sent = self
                .agent
                .post(&self.url)
                .header("Authorization", &format!("Bearer {}", self.key))
                .send_json(&body);
            let mut resp = match sent {
                Ok(r) => r,
                Err(e) => {
                    last = Some(anyhow::Error::new(e).context(self.url.clone()));
                    continue;
                }
            };
            let status = resp.status().as_u16();
            let text = resp
                .body_mut()
                .read_to_string()
                .with_context(|| self.url.clone())?;
            if status == 429 || status >= 500 {
                last = Some(anyhow::anyhow!(
                    "{}: http {status}: {}",
                    self.url,
                    text.trim()
                ));
                continue;
            }
            if status >= 400 {
                bail!("{}: http {status}: {}", self.url, text.trim());
            }
            let c: Completion = serde_json::from_str(&text)
                .with_context(|| format!("{}: not a chat completion: {}", self.url, text.trim()))?;
            if let Some(u) = c.usage {
                self.tokens.fetch_add(u.total_tokens, Relaxed);
            }
            return c
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .with_context(|| format!("{}: completion has no content", self.url));
        }
        Err(last
            .expect("attempts > 0")
            .context(format!("gave up after {ATTEMPTS} attempts")))
    }
}

/// `response_format` asking for a JSON object matching `schema`.
pub fn format(name: &str, schema: &Value) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": { "name": name, "schema": schema, "strict": true }
    })
}

/// The JSON object in `text`: the whole text when it is one, else the last
/// ```json fence, else the first balanced `{…}` outside a string.
pub fn object(text: &str) -> Result<Value> {
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(text.trim()) {
        return Ok(v);
    }
    if let Some((_, fence)) = text.rsplit_once("```json") {
        if let Ok(v) = object(fence.split("```").next().unwrap_or("")) {
            return Ok(v);
        }
    }
    let bytes = text.as_bytes();
    for start in text.match_indices('{').map(|(i, _)| i) {
        let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
        for (i, &b) in bytes.iter().enumerate().skip(start) {
            match b {
                _ if escaped => escaped = false,
                b'\\' if quoted => escaped = true,
                b'"' => quoted = !quoted,
                b'{' if !quoted => depth += 1,
                b'}' if !quoted => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v @ Value::Object(_)) = serde_json::from_str(&text[start..=i]) {
                            return Ok(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    bail!("no JSON object in model output: {}", text.trim())
}

/// A judge's `{"score", "reasoning"}`: the score may arrive as an integer, an
/// integral float, or a numeric string, and must lie in `lo..=hi`.
pub fn verdict(text: &str, lo: i64, hi: i64) -> Result<(i64, String)> {
    let v = object(text)?;
    let score = match &v["score"] {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|f| f.fract() == 0.0)
            .map(|f| f as i64),
        _ => None,
    }
    .with_context(|| format!("no integer score in {v}"))?;
    if !(lo..=hi).contains(&score) {
        bail!("score {score} is outside {lo}..={hi}");
    }
    let reasoning = match &v["reasoning"] {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    Ok((score, reasoning))
}
