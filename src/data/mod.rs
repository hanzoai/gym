//! Datasets to tokens. Rows come from [`source`], each dataset `type` turns a
//! row into an [`Example`] or a [`Prompt`], and this module truncates, drops
//! rows with nothing to learn from, shuffles, splits off validation, and pads
//! batches.

pub mod alpaca;
pub mod chat_template;
pub mod completion;
pub mod source;
pub mod text;

pub use text::{render, Message, Text};

use crate::config::{Dataset, DatasetKind};
use crate::{Batch, Config, Example, Prompt, IGNORE_INDEX};
use anyhow::Result;
use hanzo_ml::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::Value;

pub struct Tokenized {
    pub train: Vec<Example>,
    pub val: Vec<Example>,
}

/// One dataset tokenized, with what was cut.
struct Set {
    examples: Vec<Example>,
    tokens: usize,
    trained: usize,
    truncated: usize,
    dropped: usize,
}

fn set(cfg: &Config, ds: &Dataset, text: &Text) -> Result<Set> {
    let mut s = Set {
        examples: Vec::new(),
        tokens: 0,
        trained: 0,
        truncated: 0,
        dropped: 0,
    };
    for row in source::rows(ds)? {
        let mut ex = match ds.kind {
            DatasetKind::ChatTemplate => {
                chat_template::example(ds, text, cfg.train_on_inputs, &row)?
            }
            DatasetKind::Alpaca => alpaca::example(ds, text, cfg.train_on_inputs, &row)?,
            DatasetKind::Completion => completion::example(ds, text, &row)?,
        };
        if ex.input_ids.len() > cfg.sequence_len {
            ex.input_ids.truncate(cfg.sequence_len);
            ex.labels.truncate(cfg.sequence_len);
            s.truncated += 1;
        }
        let trained = ex.labels.iter().filter(|&&l| l != IGNORE_INDEX).count();
        if ex.input_ids.len() < 2 || trained == 0 {
            s.dropped += 1;
            continue;
        }
        s.tokens += ex.input_ids.len();
        s.trained += trained;
        s.examples.push(ex);
    }
    Ok(s)
}

/// Every dataset tokenized and concatenated, shuffled with `cfg.seed`, with
/// `val_set_size` (a fraction when <= 1, a count when > 1) taken from the end.
pub fn load(cfg: &Config, text: &Text) -> Result<Tokenized> {
    let mut all = Vec::new();
    for ds in &cfg.datasets {
        all.extend(set(cfg, ds, text)?.examples);
    }
    all.shuffle(&mut rand::rngs::StdRng::seed_from_u64(cfg.seed));
    let n = all.len();
    let val = if cfg.val_set_size > 1.0 {
        cfg.val_set_size as usize
    } else {
        (cfg.val_set_size * n as f64).ceil() as usize
    }
    .min(n);
    let val = all.split_off(n - val);
    Ok(Tokenized { train: all, val })
}

/// GRPO prompts from every dataset; rows longer than `sequence_len` are dropped.
pub fn load_prompts(cfg: &Config, text: &Text) -> Result<Vec<Prompt>> {
    let mut out = Vec::new();
    let mut long = 0;
    for ds in &cfg.datasets {
        for row in source::rows(ds)? {
            let p = match ds.kind {
                DatasetKind::ChatTemplate => chat_template::prompt(ds, text, &row)?,
                DatasetKind::Alpaca => alpaca::prompt(ds, text, &row)?,
                DatasetKind::Completion => completion::prompt(ds, text, &row)?,
            };
            if p.input_ids.len() > cfg.sequence_len {
                long += 1;
                continue;
            }
            out.push(p);
        }
    }
    if long > 0 {
        tracing::info!(dropped = long, "prompts longer than sequence_len");
    }
    Ok(out)
}

/// A row's reference answer for a verifier: `answer`, else `solution`.
fn answer(row: &Value) -> Option<String> {
    row["answer"]
        .as_str()
        .or_else(|| row["solution"].as_str())
        .map(String::from)
}

/// Right-pad to the longest row. Labels are not shifted; padding is
/// [`IGNORE_INDEX`]. The mask is `None` when no row needed padding.
pub fn collate(exs: &[Example], pad_id: u32, device: &Device) -> hanzo_ml::Result<Batch> {
    let b = exs.len();
    let t = exs.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
    let (mut ids, mut labels, mut mask) = (
        Vec::with_capacity(b * t),
        Vec::with_capacity(b * t),
        Vec::with_capacity(b * t),
    );
    let mut padded = false;
    for e in exs {
        let n = e.input_ids.len();
        ids.extend(&e.input_ids);
        ids.resize(ids.len() + t - n, pad_id);
        labels.extend(&e.labels);
        labels.resize(labels.len() + t - n, IGNORE_INDEX);
        mask.resize(mask.len() + n, 1u8);
        mask.resize(mask.len() + t - n, 0);
        padded |= n < t;
    }
    Ok(Batch {
        input_ids: Tensor::from_vec(ids, (b, t), device)?,
        labels: Tensor::from_vec(labels, (b, t), device)?,
        attention_mask: padded
            .then(|| Tensor::from_vec(mask, (b, t), device))
            .transpose()?,
    })
}

/// Tokenize every dataset and report what training would see.
pub fn preprocess(cfg: &Config) -> Result<()> {
    let text = Text::load(cfg)?;
    let mut total = Set {
        examples: Vec::new(),
        tokens: 0,
        trained: 0,
        truncated: 0,
        dropped: 0,
    };
    let line = |name: &str, s: &Set| {
        println!(
            "{name}: {} examples, {} tokens, {} trained, {} truncated, {} dropped",
            s.examples.len(),
            s.tokens,
            s.trained,
            s.truncated,
            s.dropped
        )
    };
    for ds in &cfg.datasets {
        let s = set(cfg, ds, &text)?;
        line(&ds.path, &s);
        total.tokens += s.tokens;
        total.trained += s.trained;
        total.truncated += s.truncated;
        total.dropped += s.dropped;
        total.examples.extend(s.examples);
    }
    line("total", &total);
    Ok(())
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::Text;
    use crate::config::{Config, Dataset, DatasetKind};
    use std::path::PathBuf;

    const WORDS: [&str; 24] = [
        "<unk>",
        "system",
        "user",
        "assistant",
        "be",
        "good",
        "hi",
        "there",
        "fine",
        "thanks",
        "how",
        "are",
        "you",
        "too",
        "one",
        "two",
        "three",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    ];

    /// WordLevel over [`WORDS`] with chatml special tokens; eos is `<|im_end|>`.
    pub fn text() -> Text {
        let special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"];
        let vocab: serde_json::Map<String, serde_json::Value> = special
            .iter()
            .chain(WORDS.iter())
            .enumerate()
            .map(|(i, w)| (w.to_string(), i.into()))
            .collect();
        let added: Vec<serde_json::Value> = special
            .iter()
            .enumerate()
            .map(|(i, t)| {
                serde_json::json!({ "id": i, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true })
            })
            .collect();
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": added,
            "pre_tokenizer": { "type": "Whitespace" },
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
        });
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap();
        Text {
            template: Some(super::text::CHATML.into()),
            eos_id: 1,
            pad_id: 2,
            bos_id: None,
            tokenizer,
        }
    }

    pub fn dataset(kind: DatasetKind, path: &str) -> Dataset {
        serde_yaml::from_str(&format!(
            "{{path: '{path}', type: {}}}",
            serde_yaml::to_string(&kind).unwrap().trim()
        ))
        .unwrap()
    }

    pub fn config(datasets: Vec<Dataset>) -> Config {
        let mut cfg: Config = serde_yaml::from_str(
            "base_model: x\noutput_dir: o\ndatasets: [{path: p, type: completion}]",
        )
        .unwrap();
        cfg.datasets = datasets;
        cfg
    }

    /// A fresh empty directory under the system temp dir.
    pub fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gym-data-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetKind;

    fn ex(ids: &[u32]) -> Example {
        Example {
            input_ids: ids.to_vec(),
            labels: ids.iter().map(|&i| i as i64).collect(),
        }
    }

    #[test]
    fn collate_pads_right() {
        let dev = Device::Cpu;
        let b = collate(&[ex(&[5, 6, 7]), ex(&[8])], 2, &dev).unwrap();
        assert_eq!(
            b.input_ids.to_vec2::<u32>().unwrap(),
            [[5, 6, 7], [8, 2, 2]]
        );
        assert_eq!(
            b.labels.to_vec2::<i64>().unwrap(),
            [[5, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
        );
        assert_eq!(
            b.attention_mask.unwrap().to_vec2::<u8>().unwrap(),
            [[1, 1, 1], [1, 0, 0]]
        );
        let b = collate(&[ex(&[5, 6]), ex(&[7, 8])], 2, &dev).unwrap();
        assert!(b.attention_mask.is_none());
        assert_eq!(b.input_ids.dims(), [2, 2]);
    }

    #[test]
    fn load_truncates_drops_shuffles_and_splits() {
        let dir = fixture::dir("load");
        let mut rows = String::new();
        for w in WORDS10 {
            rows += &format!("{{\"text\": \"{w}\"}}\n");
        }
        rows += "{\"text\": \"\"}\n";
        std::fs::write(dir.join("train.jsonl"), rows).unwrap();
        let text = fixture::text();
        let mut cfg = fixture::config(vec![fixture::dataset(
            DatasetKind::Completion,
            dir.join("train.jsonl").to_str().unwrap(),
        )]);
        cfg.sequence_len = 3;
        cfg.val_set_size = 0.2;

        let s = set(&cfg, &cfg.datasets[0], &text).unwrap();
        assert_eq!((s.examples.len(), s.truncated, s.dropped), (10, 2, 1));
        assert!(s.examples.iter().all(|e| e.input_ids.len() <= 3));
        assert_eq!(s.tokens, 7 * 2 + 3 * 3);
        assert_eq!(s.trained, s.tokens);

        let t = load(&cfg, &text).unwrap();
        assert_eq!((t.train.len(), t.val.len()), (8, 2));
        let again = load(&cfg, &text).unwrap();
        assert_eq!(t.train, again.train);
        let order: Vec<u32> = t
            .train
            .iter()
            .chain(&t.val)
            .map(|e| e.input_ids[0])
            .collect();
        let mut sorted = order.clone();
        sorted.sort();
        assert_ne!(order, sorted);

        cfg.val_set_size = 3.0;
        assert_eq!(load(&cfg, &text).unwrap().val.len(), 3);
        cfg.val_set_size = 0.0;
        assert_eq!(load(&cfg, &text).unwrap().val.len(), 0);

        cfg.datasets.push(cfg.datasets[0].clone());
        assert_eq!(load(&cfg, &text).unwrap().train.len(), 20);
    }

    const WORDS10: [&str; 10] = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "hi there",
        "fine thanks you",
        "how are you too",
    ];

    #[test]
    fn load_prompts_drops_long_rows() {
        let dir = fixture::dir("prompts");
        std::fs::write(
            dir.join("train.jsonl"),
            "{\"text\": \"hi\", \"answer\": \"a\"}\n{\"text\": \"hi there fine thanks\"}\n",
        )
        .unwrap();
        let text = fixture::text();
        let mut cfg = fixture::config(vec![fixture::dataset(
            DatasetKind::Completion,
            dir.to_str().unwrap(),
        )]);
        cfg.sequence_len = 2;
        let p = load_prompts(&cfg, &text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(
            (p[0].text.as_str(), p[0].answer.as_deref()),
            ("hi", Some("a"))
        );
    }
}
