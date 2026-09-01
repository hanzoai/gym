//! Trainable models. `load` builds a [`Transformer`] with LoRA from a Hugging
//! Face checkpoint; adapters are written and read in PEFT layout; `merge`
//! folds an adapter into the base weights.

pub mod llama;
pub mod lora;
pub mod qwen3;
#[cfg(test)]
mod tests;
pub mod transformer;

use crate::config::{Adapter, Config};
use crate::{hub, Cache, CausalLm};
use anyhow::Context;
use hanzo_ml::{DType, Device, Result, Tensor, Var};
use hanzo_nn::VarBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
pub use transformer::{Arch, Lora, Transformer};

/// The transformer described by `cfg.base_model`, with LoRA on `cfg.lora_targets()`.
pub fn build(cfg: &Config, device: &Device, dtype: DType) -> anyhow::Result<Transformer> {
    anyhow::ensure!(
        cfg.adapter == Adapter::Lora,
        "full fine-tuning is not implemented; set adapter: lora"
    );
    anyhow::ensure!(
        !cfg.gradient_checkpointing,
        "gradient_checkpointing is not implemented: hanzo-ml's autograd cannot recompute a layer in the backward pass; set gradient_checkpointing: false"
    );
    let snap = hub::snapshot(&cfg.base_model)?;
    let (arch, json) = arch(&snap.config)?;
    let lora = Lora {
        r: cfg.lora_r,
        alpha: cfg.lora_alpha,
        dropout: cfg.lora_dropout,
        targets: cfg.lora_targets(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&snap.weights, dtype, device)? };
    let mut model = Transformer::new(arch, lora, &vb)?;
    let eos = &json["eos_token_id"];
    model.eos = eos
        .as_u64()
        .or_else(|| eos.get(0)?.as_u64())
        .map(|e| e as u32);
    model.base = cfg.base_model.clone();
    Ok(model)
}

/// `config.json` parsed by its `model_type`.
fn arch(config: &Path) -> anyhow::Result<(Arch, serde_json::Value)> {
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(config)?)?;
    let arch = match json["model_type"].as_str() {
        Some("qwen3") => qwen3::arch(&json)?,
        Some("llama") => llama::arch(&json)?,
        other => anyhow::bail!("model_type {other:?} is not supported (qwen3, llama)"),
    };
    Ok((arch, json))
}

pub fn load(cfg: &Config, device: &Device, dtype: DType) -> anyhow::Result<Box<dyn CausalLm>> {
    Ok(Box::new(build(cfg, device, dtype)?))
}

/// Reads `adapter_model.safetensors` from `dir` into the model's LoRA vars.
pub fn load_adapter(model: &Transformer, dir: &Path) -> anyhow::Result<()> {
    let file = dir.join("adapter_model.safetensors");
    let tensors = hanzo_ml::safetensors::load(&file, model.device())?;
    for (name, l) in model.adapters() {
        for (suffix, var) in [("lora_A", &l.a), ("lora_B", &l.b)] {
            let key = format!("{name}.{suffix}.weight");
            let t = tensors
                .get(&key)
                .with_context(|| format!("{}: missing {key}", file.display()))?;
            var.set(&t.to_dtype(model.dtype())?)?;
        }
    }
    Ok(())
}

/// Writes the base weights with the adapter from `cfg.output_dir` folded in,
/// plus config and tokenizer files, to `out` or `<output_dir>/merged`.
pub fn merge(cfg: &Config, out: Option<&str>) -> anyhow::Result<()> {
    let snap = hub::snapshot(&cfg.base_model)?;
    let device = Device::Cpu;
    let mut base = HashMap::new();
    for w in &snap.weights {
        base.extend(hanzo_ml::safetensors::load(w, &device)?);
    }
    let dtype = base
        .get("model.embed_tokens.weight")
        .context("model.embed_tokens.weight")?
        .dtype();
    let (arch, _) = arch(&snap.config)?;
    let lora = Lora {
        r: cfg.lora_r,
        alpha: cfg.lora_alpha,
        dropout: 0.0,
        targets: cfg.lora_targets(),
    };
    let model = Transformer::new(
        arch,
        lora,
        &VarBuilder::from_tensors(base.clone(), dtype, &device),
    )?;
    load_adapter(&model, Path::new(&cfg.output_dir))?;
    for (name, l) in model.adapters() {
        let key = format!("{}.weight", name.trim_start_matches("base_model.model."));
        base.insert(key, l.merged_weight()?);
    }
    let out = out
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&cfg.output_dir).join("merged"));
    std::fs::create_dir_all(&out)?;
    hanzo_ml::safetensors::save(&base, out.join("model.safetensors"))?;
    for f in [
        Some(&snap.config),
        Some(&snap.tokenizer),
        snap.tokenizer_config.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        std::fs::copy(f, out.join(f.file_name().context("file name")?))?;
    }
    Ok(())
}

impl CausalLm for Transformer {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        Transformer::forward(self, input_ids, attention_mask)
    }

    fn forward_cached(&self, input_ids: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        Transformer::forward_cached(self, input_ids, cache)
    }

    fn trainable_vars(&self) -> Vec<Var> {
        Transformer::trainable_vars(self)
    }

    fn device(&self) -> &Device {
        Transformer::device(self)
    }

    fn dtype(&self) -> DType {
        Transformer::dtype(self)
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos
    }

    fn vocab_size(&self) -> usize {
        self.arch.vocab
    }

    fn save_adapter(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut tensors = HashMap::new();
        for (name, l) in self.adapters() {
            tensors.insert(format!("{name}.lora_A.weight"), l.a.as_tensor().clone());
            tensors.insert(format!("{name}.lora_B.weight"), l.b.as_tensor().clone());
        }
        hanzo_ml::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
        let config = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.lora.r,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "target_modules": self.lora.targets,
            "base_model_name_or_path": self.base,
            "bias": "none",
        });
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&config)?,
        )?;
        Ok(())
    }
}
