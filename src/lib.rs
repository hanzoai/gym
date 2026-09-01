//! Hanzo Gym: LoRA fine-tuning and GRPO for language models on hanzo-ml.
//!
//! The crate is four modules around two shared types. [`config::Config`] is the
//! Axolotl-shaped YAML a run is described by, so a `zooai/gym` config ports with
//! its keys intact. [`CausalLm`] is what a trainable model provides; `train` and
//! `grpo` are written against it and nothing else, so a new architecture is one
//! `impl`.

pub mod config;
pub mod data;
pub mod grpo;
pub mod hub;
pub mod model;
pub mod synth;
pub mod train;
pub mod trainjob;

pub use config::Config;

use hanzo_ml::{DType, Device, Result, Tensor, Var};

/// Label value that excludes a position from the loss (Hugging Face convention).
pub const IGNORE_INDEX: i64 = -100;

/// One tokenized training example. `labels[i]` is the target for position `i`
/// (already shifted by the collator, not here) or [`IGNORE_INDEX`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i64>,
}

/// A padded batch on device. `input_ids` is `[b, t]` u32, `labels` is `[b, t]`
/// i64 with [`IGNORE_INDEX`] at padding and non-trained positions, and
/// `attention_mask` is `[b, t]` u8 (1 = real token) or `None` when nothing is padded.
#[derive(Debug, Clone)]
pub struct Batch {
    pub input_ids: Tensor,
    pub labels: Tensor,
    pub attention_mask: Option<Tensor>,
}

/// Per-layer keys and values from the positions already decoded. `len` is how
/// many positions they cover; the next `forward_cached` call continues at it.
#[derive(Debug, Default)]
pub struct Cache {
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    pub len: usize,
}

/// A decoder-only language model that can be trained.
///
/// `forward` runs the full sequence with a causal mask and returns logits
/// `[b, t, vocab]` in the model dtype; the graph must reach every tensor in
/// `trainable_vars`, and only those. Frozen base weights are plain tensors.
/// `forward_cached` runs only the new positions `[b, t_new]` against `cache`,
/// appends to it, and returns their logits — the decode path for generation;
/// no graph is needed there.
pub trait CausalLm {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;
    fn forward_cached(&self, input_ids: &Tensor, cache: &mut Cache) -> Result<Tensor>;
    fn trainable_vars(&self) -> Vec<Var>;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn eos_token_id(&self) -> Option<u32>;
    fn vocab_size(&self) -> usize;
    /// Write the trainable weights in PEFT layout (`adapter_config.json` +
    /// `adapter_model.safetensors`) so Engine, vLLM and `peft` load them.
    fn save_adapter(&self, dir: &std::path::Path) -> anyhow::Result<()>;
}

/// One GRPO prompt: the tokenized chat prefix the policy completes, and the
/// reference answer a verifier scores against, when the dataset has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    pub input_ids: Vec<u32>,
    pub text: String,
    pub answer: Option<String>,
}
