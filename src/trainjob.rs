//! `gym trainjob`: the training container behind Hanzo's hosted fine-tuning
//! API. The broker (hanzoai/ai `cluster/finetune.go`, `trainerEnv`) submits a
//! Kubeflow TrainJob whose initializers place the base model in `MODEL_DIR`
//! and the dataset files in `DATASET_DIR`, and describes the run as
//! environment variables. This module maps that environment onto a
//! [`Config`] and trains. Defaults follow `object.RecommendHyperparams` in the
//! same repo, so a variable the broker omits resolves to what it would have
//! sent. `HF_TOKEN` is read by hf-hub itself and never logged.

use crate::config::{Config, DatasetKind};
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};

pub fn run() -> Result<()> {
    let cfg = config(|k| std::env::var(k).ok())?;
    tracing::info!("trainjob config:\n{}", serde_yaml::to_string(&cfg)?);
    crate::train::run(&cfg)
}

/// The [`Config`] the TrainJob environment describes, read through `env`.
pub fn config(env: impl Fn(&str) -> Option<String>) -> Result<Config> {
    let get = |k: &str| {
        env(k)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let model_dir = get("MODEL_DIR").unwrap_or_else(|| "/workspace/model".into());
    let dataset_dir = get("DATASET_DIR").unwrap_or_else(|| "/workspace/dataset".into());

    // The broker's normalizeMethod: anything but lora|full is qlora.
    let method = match get("METHOD").as_deref() {
        Some("lora") => "lora",
        Some("full") => "full",
        _ => "qlora",
    };
    if method != "lora" {
        bail!(
            "METHOD={method}: not implemented in hanzo-gym 0.1; use the hanzo-ft-{method} runtime"
        );
    }
    if parse::<bool>(&get, "QUANT_4BIT", "false")? {
        bail!("QUANT_4BIT=true: 4-bit quantization is not implemented in hanzo-gym 0.1; use the hanzo-ft-qlora runtime");
    }
    if parse::<bool>(&get, "GRADIENT_CHECKPOINTING", "false")? {
        bail!("GRADIENT_CHECKPOINTING=true: not implemented in hanzo-gym 0.1 — hanzo-ml's autograd cannot recompute a layer in the backward pass; send false");
    }
    let kind = match get("TASK").as_deref() {
        Some("chat") => DatasetKind::ChatTemplate,
        Some("instruct") => DatasetKind::Alpaca,
        Some("completion") => DatasetKind::Completion,
        other => bail!(
            "TASK={}: must be chat, instruct or completion",
            other.unwrap_or("")
        ),
    };

    let base_model = if Path::new(&model_dir).is_dir() {
        model_dir
    } else {
        get("BASE_MODEL").with_context(|| {
            format!("MODEL_DIR={model_dir} is not a directory and BASE_MODEL is unset")
        })?
    };
    let files =
        files(Path::new(&dataset_dir)).with_context(|| format!("DATASET_DIR={dataset_dir}"))?;
    if files.is_empty() {
        bail!("DATASET_DIR={dataset_dir}: no .jsonl, .json or .parquet files");
    }
    let datasets: Vec<serde_json::Value> = files
        .iter()
        .map(|p| json!({ "path": p, "type": kind, "field_messages": "messages" }))
        .collect();

    let epochs: f64 = parse(&get, "EPOCHS", "3")?;
    if epochs < 1.0 || epochs.fract() != 0.0 {
        bail!("EPOCHS={epochs}: hanzo-gym trains a whole number of epochs");
    }

    let cfg: Config = serde_json::from_value(json!({
        "base_model": base_model,
        "output_dir": get("OUTPUT_DIR").unwrap_or_else(|| "/workspace/output".into()),
        "datasets": datasets,
        "sequence_len": parse::<usize>(&get, "MAX_SEQ_LEN", "2048")?,
        "adapter": "lora",
        "lora_r": parse::<usize>(&get, "LORA_RANK", "16")?,
        "lora_alpha": parse::<f64>(&get, "LORA_ALPHA", "32")?,
        "lora_dropout": parse::<f64>(&get, "LORA_DROPOUT", "0.05")?,
        "lora_target_linear": true,
        "micro_batch_size": parse::<usize>(&get, "BATCH_SIZE", "2")?,
        "gradient_accumulation_steps": parse::<usize>(&get, "GRAD_ACCUM", "8")?,
        "num_epochs": epochs as usize,
        "learning_rate": parse::<f64>(&get, "LEARNING_RATE", "2e-4")?,
        "warmup_ratio": parse::<f64>(&get, "WARMUP_RATIO", "0.03")?,
        "weight_decay": parse::<f64>(&get, "WEIGHT_DECAY", "0.01")?,
        "bf16": "auto",
    }))?;
    cfg.validate()?;
    Ok(cfg)
}

fn parse<T: std::str::FromStr>(
    get: &impl Fn(&str) -> Option<String>,
    key: &str,
    default: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = get(key).unwrap_or_else(|| default.into());
    raw.parse().map_err(|e| anyhow::anyhow!("{key}={raw}: {e}"))
}

/// Every `.jsonl`, `.json` or `.parquet` file under `dir`, sorted; hidden
/// entries (`.cache`, `.gitattributes`) are skipped.
fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'))
        {
            continue;
        }
        if p.is_dir() {
            out.extend(files(&p)?);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| ["jsonl", "json", "parquet"].contains(&x))
        {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Adapter;
    use crate::data::fixture;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    /// A dataset dir with `data/train.jsonl` and a hidden `.cache` file.
    fn dataset(name: &str) -> String {
        let d = fixture::dir(&format!("trainjob-{name}"));
        std::fs::create_dir_all(d.join("data")).unwrap();
        std::fs::write(d.join("data/train.jsonl"), "{}\n").unwrap();
        std::fs::write(d.join(".cache"), "").unwrap();
        d.to_string_lossy().into_owned()
    }

    #[test]
    fn chat_is_chat_template_over_every_file() {
        let d = dataset("chat");
        let cfg = config(env(&[
            ("BASE_MODEL", "Qwen/Qwen3-0.6B"),
            ("METHOD", "lora"),
            ("TASK", "chat"),
            ("MODEL_DIR", "/nonexistent/model"),
            ("DATASET_DIR", &d),
            ("OUTPUT_DIR", "/out"),
            ("EPOCHS", "2"),
            ("LEARNING_RATE", "1e-4"),
            ("BATCH_SIZE", "4"),
            ("GRAD_ACCUM", "2"),
            ("MAX_SEQ_LEN", "512"),
            ("LORA_RANK", "8"),
            ("LORA_ALPHA", "16"),
            ("LORA_DROPOUT", "0.1"),
            ("QUANT_4BIT", "false"),
            ("GRADIENT_CHECKPOINTING", "false"),
            ("WARMUP_RATIO", "0.1"),
            ("WEIGHT_DECAY", "0.0"),
        ]))
        .unwrap();
        assert_eq!(cfg.base_model, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.output_dir, "/out");
        assert_eq!(cfg.datasets.len(), 1);
        assert_eq!(cfg.datasets[0].kind, DatasetKind::ChatTemplate);
        assert_eq!(cfg.datasets[0].field_messages, "messages");
        assert!(cfg.datasets[0].path.ends_with("data/train.jsonl"));
        assert_eq!(cfg.adapter, Adapter::Lora);
        assert!(cfg.lora_target_linear);
        assert_eq!(cfg.num_epochs, 2);
        assert_eq!(cfg.learning_rate, 1e-4);
        assert_eq!(cfg.micro_batch_size, 4);
        assert_eq!(cfg.gradient_accumulation_steps, 2);
        assert_eq!(cfg.sequence_len, 512);
        assert_eq!(cfg.lora_r, 8);
        assert_eq!(cfg.lora_alpha, 16.0);
        assert_eq!(cfg.lora_dropout, 0.1);
        assert_eq!(cfg.warmup_ratio, Some(0.1));
        assert_eq!(cfg.weight_decay, 0.0);
        assert!(!cfg.gradient_checkpointing);
    }

    #[test]
    fn instruct_is_alpaca_and_completion_is_completion() {
        let d = dataset("kinds");
        for (task, kind) in [
            ("instruct", DatasetKind::Alpaca),
            ("completion", DatasetKind::Completion),
        ] {
            let cfg = config(env(&[
                ("BASE_MODEL", "m"),
                ("METHOD", "lora"),
                ("TASK", task),
                ("DATASET_DIR", &d),
            ]))
            .unwrap();
            assert_eq!(cfg.datasets[0].kind, kind);
        }
        let err = config(env(&[
            ("BASE_MODEL", "m"),
            ("METHOD", "lora"),
            ("DATASET_DIR", &d),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("TASK="), "{err}");
    }

    #[test]
    fn model_dir_wins_when_it_exists() {
        let d = dataset("modeldir");
        let cfg = config(env(&[
            ("BASE_MODEL", "Qwen/Qwen3-0.6B"),
            ("METHOD", "lora"),
            ("TASK", "chat"),
            ("MODEL_DIR", &d),
            ("DATASET_DIR", &d),
        ]))
        .unwrap();
        assert_eq!(cfg.base_model, d);
    }

    #[test]
    fn defaults_match_the_broker() {
        let d = dataset("defaults");
        let cfg = config(env(&[
            ("BASE_MODEL", "m"),
            ("METHOD", "lora"),
            ("TASK", "chat"),
            ("DATASET_DIR", &d),
        ]))
        .unwrap();
        assert_eq!(cfg.output_dir, "/workspace/output");
        assert_eq!(cfg.num_epochs, 3);
        assert_eq!(cfg.learning_rate, 2e-4);
        assert_eq!(cfg.micro_batch_size, 2);
        assert_eq!(cfg.gradient_accumulation_steps, 8);
        assert_eq!(cfg.sequence_len, 2048);
        assert_eq!(cfg.lora_r, 16);
        assert_eq!(cfg.lora_alpha, 32.0);
        assert_eq!(cfg.lora_dropout, 0.05);
        assert_eq!(cfg.warmup_ratio, Some(0.03));
        assert_eq!(cfg.weight_decay, 0.01);
        assert_eq!(cfg.bf16, crate::config::Precision::Auto);
        assert!(!cfg.gradient_checkpointing);
        assert_eq!(cfg.lora_targets().len(), 7);
    }

    #[test]
    fn qlora_full_4bit_and_checkpointing_are_rejected() {
        let d = dataset("reject");
        let base = [("BASE_MODEL", "m"), ("TASK", "chat"), ("DATASET_DIR", &d)];
        let msg = |extra: &[(&str, &str)]| {
            config(env(&[&base[..], extra].concat()))
                .unwrap_err()
                .to_string()
        };
        assert!(msg(&[("METHOD", "qlora")]).contains("use the hanzo-ft-qlora runtime"));
        assert!(msg(&[("METHOD", "full")]).contains("use the hanzo-ft-full runtime"));
        assert!(
            msg(&[]).contains("METHOD=qlora"),
            "unset METHOD is qlora, as in the broker"
        );
        assert!(msg(&[("METHOD", "lora"), ("QUANT_4BIT", "true")]).contains("QUANT_4BIT=true"));
        assert!(
            msg(&[("METHOD", "lora"), ("GRADIENT_CHECKPOINTING", "true")])
                .contains("backward pass")
        );
        assert!(msg(&[("METHOD", "lora"), ("EPOCHS", "1.5")]).contains("whole number"));
        assert!(msg(&[("METHOD", "lora"), ("LORA_RANK", "x")]).starts_with("LORA_RANK=x"));
    }

    #[test]
    fn empty_or_missing_dataset_dir_is_rejected() {
        let empty = fixture::dir("trainjob-empty")
            .to_string_lossy()
            .into_owned();
        let base = [("BASE_MODEL", "m"), ("METHOD", "lora"), ("TASK", "chat")];
        let err = config(env(&[&base[..], &[("DATASET_DIR", &empty)]].concat())).unwrap_err();
        assert!(
            err.to_string().contains("no .jsonl, .json or .parquet"),
            "{err}"
        );
        let err = config(env(
            &[&base[..], &[("DATASET_DIR", "/nonexistent/ds")]].concat()
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("DATASET_DIR=/nonexistent/ds"),
            "{err}"
        );
    }
}
