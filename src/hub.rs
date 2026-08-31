//! Where model files come from: a local directory is used as is; anything else
//! is a Hugging Face repo id, fetched into the hf-hub cache.

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};

/// The files a decoder-only checkpoint consists of.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub weights: Vec<PathBuf>,
}

pub fn snapshot(id_or_path: &str) -> Result<Snapshot> {
    let p = Path::new(id_or_path);
    if p.is_dir() {
        return local(p);
    }
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(id_or_path.to_string(), RepoType::Model, "main".into()));
    let config = repo.get("config.json").context("config.json")?;
    let tokenizer = repo.get("tokenizer.json").context("tokenizer.json")?;
    let tokenizer_config = repo.get("tokenizer_config.json").ok();
    let weights = match repo.get("model.safetensors.index.json") {
        Ok(index) => shards(&repo, &index)?,
        Err(_) => vec![repo.get("model.safetensors").context("model.safetensors")?],
    };
    Ok(Snapshot { config, tokenizer, tokenizer_config, weights })
}

fn shards(repo: &ApiRepo, index: &Path) -> Result<Vec<PathBuf>> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
    let mut names: Vec<String> = v["weight_map"]
        .as_object()
        .context("weight_map")?
        .values()
        .filter_map(|f| f.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    names.iter().map(|n| Ok(repo.get(n)?)).collect()
}

fn local(dir: &Path) -> Result<Snapshot> {
    let f = |n: &str| dir.join(n);
    let index = f("model.safetensors.index.json");
    let weights = if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let mut names: Vec<String> = v["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect();
        names.sort();
        names.dedup();
        names.iter().map(|n| dir.join(n)).collect()
    } else {
        vec![f("model.safetensors")]
    };
    Ok(Snapshot {
        config: f("config.json"),
        tokenizer: f("tokenizer.json"),
        tokenizer_config: Some(f("tokenizer_config.json")).filter(|p| p.exists()),
        weights,
    })
}
