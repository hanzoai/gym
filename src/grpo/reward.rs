//! Verifiers named in `trl.reward_funcs`, weighted by `trl.reward_weights`.
//!
//! - `exact_match`: completion equals `Prompt.answer` after whitespace and case normalisation.
//! - `numeric_match`: the last number in the completion (inside `\boxed{…}` when
//!   present) equals the answer's number within 1e-6.
//! - `format:<regex>`: 1.0 when the regex matches the completion.
//! - `length:<max_tokens>`: 0 within the budget, -1 over it.
//! - `http:<url>`: POST `{"prompt","completion","answer"}`, read `{"reward": number}`.

use crate::{Config, Prompt};
use anyhow::Context;
use regex::Regex;
use std::sync::LazyLock;

static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-?\d+(?:,\d{3})*(?:\.\d+)?").unwrap());
static BOXED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\boxed\{([^{}]*)\}").unwrap());

pub struct Reward {
    pub name: String,
    pub weight: f32,
    kind: Kind,
}

enum Kind {
    Exact,
    Numeric,
    Format(Regex),
    Length(usize),
    Http(String),
    /// Occurrences of a token id; a test fixture for policies without a tokenizer.
    #[cfg(test)]
    Token(u32),
}

#[derive(serde::Deserialize)]
struct Score {
    reward: f64,
}

impl Reward {
    pub fn parse(spec: &str, weight: f32) -> anyhow::Result<Self> {
        let kind = match spec {
            "exact_match" => Kind::Exact,
            "numeric_match" => Kind::Numeric,
            s if s.starts_with("http://") || s.starts_with("https://") => Kind::Http(s.to_string()),
            s => match s.split_once(':') {
                Some(("format", re)) => {
                    Kind::Format(Regex::new(re).with_context(|| format!("reward {spec}"))?)
                }
                Some(("length", n)) => {
                    Kind::Length(n.parse().with_context(|| format!("reward {spec}"))?)
                }
                Some(("http", url)) => Kind::Http(url.to_string()),
                _ => anyhow::bail!("unknown reward function {spec:?}"),
            },
        };
        Ok(Self {
            name: spec.to_string(),
            weight,
            kind,
        })
    }

    #[cfg(test)]
    pub fn token(id: u32) -> Self {
        Self {
            name: format!("token:{id}"),
            weight: 1.0,
            kind: Kind::Token(id),
        }
    }

    /// Unweighted score of one completion.
    pub fn score(&self, prompt: &Prompt, tokens: &[u32], text: &str) -> anyhow::Result<f32> {
        let answer = prompt.answer.as_deref();
        Ok(match &self.kind {
            Kind::Exact => answer.is_some_and(|a| normal(a) == normal(text)) as u8 as f32,
            Kind::Numeric => match (answer.and_then(number), number(text)) {
                (Some(a), Some(c)) => ((a - c).abs() < 1e-6) as u8 as f32,
                _ => 0.0,
            },
            Kind::Format(re) => re.is_match(text) as u8 as f32,
            Kind::Length(max) => {
                if tokens.len() <= *max {
                    0.0
                } else {
                    -1.0
                }
            }
            Kind::Http(url) => {
                let body = serde_json::json!({ "prompt": prompt.text, "completion": text, "answer": answer });
                let score: Score = ureq::post(url)
                    .send_json(body)
                    .with_context(|| format!("reward {url}"))?
                    .body_mut()
                    .read_json()
                    .with_context(|| format!("reward {url}: expected {{\"reward\": number}}"))?;
                score.reward as f32
            }
            #[cfg(test)]
            Kind::Token(id) => tokens.iter().filter(|&t| t == id).count() as f32,
        })
    }
}

fn normal(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The last number in `s`, preferring the contents of the last `\boxed{…}`.
fn number(s: &str) -> Option<f64> {
    let field = BOXED
        .captures_iter(s)
        .last()
        .map(|c| c.get(1).unwrap().as_str())
        .unwrap_or(s);
    NUMBER
        .find_iter(field)
        .last()?
        .as_str()
        .replace(',', "")
        .parse()
        .ok()
}

pub fn parse_rewards(cfg: &Config) -> anyhow::Result<Vec<Reward>> {
    let t = &cfg.trl;
    t.reward_funcs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            Reward::parse(spec, t.reward_weights.get(i).copied().unwrap_or(1.0) as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn prompt(answer: Option<&str>) -> Prompt {
        Prompt {
            input_ids: vec![1],
            text: "q?".into(),
            answer: answer.map(String::from),
        }
    }

    fn score(spec: &str, answer: Option<&str>, text: &str) -> f32 {
        Reward::parse(spec, 1.0)
            .unwrap()
            .score(&prompt(answer), &[0; 3], text)
            .unwrap()
    }

    #[test]
    fn exact_normalises_whitespace_and_case() {
        assert_eq!(
            score("exact_match", Some("Paris  France"), " paris\nfrance "),
            1.0
        );
        assert_eq!(score("exact_match", Some("Paris"), "London"), 0.0);
        assert_eq!(score("exact_match", None, "Paris"), 0.0);
    }

    #[test]
    fn numeric_takes_last_number_and_boxed() {
        assert_eq!(score("numeric_match", Some("42"), "First 7, then 42."), 1.0);
        assert_eq!(score("numeric_match", Some("42"), "First 42, then 7."), 0.0);
        assert_eq!(
            score("numeric_match", Some("42"), r"\boxed{42} and then 7"),
            1.0
        );
        assert_eq!(
            score("numeric_match", Some("#### 1,234"), r"so \boxed{1234.0}"),
            1.0
        );
        assert_eq!(score("numeric_match", Some("-3.5"), "answer is -3.5"), 1.0);
        assert_eq!(score("numeric_match", Some("3"), "no digits"), 0.0);
        assert_eq!(score("numeric_match", None, "3"), 0.0);
    }

    #[test]
    fn format_and_length() {
        assert_eq!(
            score(r"format:^<think>.*</think>", None, "<think>x</think> y"),
            1.0
        );
        assert_eq!(score(r"format:^<think>.*</think>", None, "y"), 0.0);
        assert_eq!(score("length:3", None, ""), 0.0);
        assert_eq!(score("length:2", None, ""), -1.0);
    }

    #[test]
    fn unknown_and_malformed_are_errors() {
        assert!(Reward::parse("bleu", 1.0).is_err());
        assert!(Reward::parse("length:many", 1.0).is_err());
        assert!(Reward::parse("format:(", 1.0).is_err());
    }

    #[test]
    fn weights_default_to_one() {
        let y = "base_model: x\noutput_dir: o\ndatasets: [{path: p, type: completion}]\nrl: grpo\n\
                 trl: {reward_funcs: [exact_match, 'length:8'], reward_weights: [2.0]}\n";
        let cfg: Config = serde_yaml::from_str(y).unwrap();
        let r = parse_rewards(&cfg).unwrap();
        assert_eq!((r[0].weight, r[1].weight), (2.0, 1.0));
        assert_eq!(r[1].name, "length:8");
    }

    #[test]
    fn http_posts_json_and_reads_reward() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/score", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 4096];
            let mut n = 0;
            while !String::from_utf8_lossy(&buf[..n]).contains("\r\n\r\n") {
                n += s.read(&mut buf[n..]).unwrap();
            }
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let len: usize = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap())
                })
                .unwrap();
            let start = head.find("\r\n\r\n").unwrap() + 4;
            while n < start + len {
                n += s.read(&mut buf[n..]).unwrap();
            }
            let body: serde_json::Value = serde_json::from_slice(&buf[start..start + len]).unwrap();
            let reply = r#"{"reward":0.75}"#;
            write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).unwrap();
            body
        });
        let r = Reward::parse(&format!("http:{url}"), 1.0).unwrap();
        assert_eq!(r.score(&prompt(Some("4")), &[], "2+2=4").unwrap(), 0.75);
        let body = server.join().unwrap();
        assert_eq!(
            body,
            serde_json::json!({"prompt": "q?", "completion": "2+2=4", "answer": "4"})
        );
    }
}
