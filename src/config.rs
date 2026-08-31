//! The run description. Keys follow Axolotl so an existing config ports
//! unchanged; anything Axolotl-specific that this crate does not implement is
//! rejected at load, not silently ignored.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub base_model: String,
    #[serde(default)]
    pub tokenizer_config: Option<String>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub datasets: Vec<Dataset>,
    #[serde(default)]
    pub val_set_size: f64,
    pub output_dir: String,
    #[serde(default = "d_seq_len")]
    pub sequence_len: usize,
    #[serde(default)]
    pub train_on_inputs: bool,

    #[serde(default)]
    pub adapter: Adapter,
    #[serde(default = "d_lora_r")]
    pub lora_r: usize,
    #[serde(default = "d_lora_alpha")]
    pub lora_alpha: f64,
    #[serde(default)]
    pub lora_dropout: f64,
    #[serde(default)]
    pub lora_target_modules: Vec<String>,
    #[serde(default)]
    pub lora_target_linear: bool,

    #[serde(default = "d_one")]
    pub micro_batch_size: usize,
    #[serde(default = "d_one")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "d_one")]
    pub num_epochs: usize,
    #[serde(default)]
    pub max_steps: Option<usize>,
    #[serde(default = "d_lr")]
    pub learning_rate: f64,
    #[serde(default)]
    pub lr_scheduler: Scheduler,
    #[serde(default)]
    pub warmup_steps: Option<usize>,
    #[serde(default)]
    pub warmup_ratio: Option<f64>,
    #[serde(default)]
    pub weight_decay: f64,
    #[serde(default = "d_grad_norm")]
    pub max_grad_norm: f64,
    #[serde(default = "d_ten")]
    pub logging_steps: usize,
    #[serde(default)]
    pub save_steps: Option<usize>,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default)]
    pub bf16: Precision,
    #[serde(default)]
    pub gradient_checkpointing: bool,

    #[serde(default)]
    pub rl: Option<Rl>,
    #[serde(default)]
    pub trl: Trl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    /// Local file, directory, or Hugging Face dataset id.
    pub path: String,
    #[serde(rename = "type")]
    pub kind: DatasetKind,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub data_files: Option<Vec<String>>,
    #[serde(default = "d_messages")]
    pub field_messages: String,
    #[serde(default)]
    pub message_property_mappings: Option<MessageMap>,
    #[serde(default = "d_roles")]
    pub roles_to_train: Vec<String>,
    #[serde(default)]
    pub train_on_eos: TrainOnEos,
    #[serde(default = "d_instruction")]
    pub field_instruction: String,
    #[serde(default = "d_input")]
    pub field_input: String,
    #[serde(default = "d_output")]
    pub field_output: String,
    #[serde(default = "d_text")]
    pub field: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    ChatTemplate,
    Alpaca,
    Completion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMap {
    #[serde(default = "d_role")]
    pub role: String,
    #[serde(default = "d_content")]
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainOnEos {
    #[default]
    Turn,
    Last,
    All,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adapter {
    #[default]
    None,
    Lora,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scheduler {
    #[default]
    Cosine,
    Linear,
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// bf16 on GPU, f32 on CPU.
    #[default]
    Auto,
    #[serde(rename = "true")]
    On,
    #[serde(rename = "false")]
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rl {
    Grpo,
}

/// GRPO settings, under Axolotl's `trl:` key with its names.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trl {
    #[serde(default = "d_gens")]
    pub num_generations: usize,
    #[serde(default = "d_completion")]
    pub max_completion_length: usize,
    #[serde(default)]
    pub beta: f64,
    #[serde(default = "d_one_f")]
    pub temperature: f64,
    #[serde(default = "d_true")]
    pub scale_rewards: bool,
    #[serde(default)]
    pub reward_funcs: Vec<String>,
    #[serde(default)]
    pub reward_weights: Vec<f64>,
}

impl Default for Trl {
    fn default() -> Self {
        serde_yaml::from_str("{}").expect("all Trl fields default")
    }
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let cfg: Config = serde_yaml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.datasets.is_empty(), "at least one dataset is required");
        anyhow::ensure!(self.sequence_len > 1, "sequence_len must be > 1");
        anyhow::ensure!(self.micro_batch_size >= 1, "micro_batch_size must be >= 1");
        if self.adapter == Adapter::Lora {
            anyhow::ensure!(self.lora_r > 0, "lora_r must be > 0");
            anyhow::ensure!(
                self.lora_target_linear || !self.lora_target_modules.is_empty(),
                "lora needs lora_target_modules or lora_target_linear: true"
            );
        }
        if self.rl == Some(Rl::Grpo) {
            anyhow::ensure!(self.trl.num_generations >= 2, "trl.num_generations must be >= 2");
            anyhow::ensure!(!self.trl.reward_funcs.is_empty(), "grpo needs trl.reward_funcs");
            if !self.trl.reward_weights.is_empty() {
                anyhow::ensure!(
                    self.trl.reward_weights.len() == self.trl.reward_funcs.len(),
                    "trl.reward_weights must match trl.reward_funcs"
                );
            }
        }
        Ok(())
    }

    /// LoRA scaling `alpha / r`.
    pub fn lora_scale(&self) -> f64 {
        self.lora_alpha / self.lora_r as f64
    }

    /// Modules LoRA is applied to. `lora_target_linear: true` means every
    /// projection in attention and MLP.
    pub fn lora_targets(&self) -> Vec<String> {
        if self.lora_target_linear {
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            self.lora_target_modules.clone()
        }
    }

    /// Warmup steps for a run of `total` optimizer steps.
    pub fn warmup(&self, total: usize) -> usize {
        match (self.warmup_steps, self.warmup_ratio) {
            (Some(s), _) => s,
            (None, Some(r)) => (total as f64 * r).round() as usize,
            (None, None) => 0,
        }
    }
}

fn d_seq_len() -> usize { 2048 }
fn d_lora_r() -> usize { 8 }
fn d_lora_alpha() -> f64 { 16.0 }
fn d_one() -> usize { 1 }
fn d_one_f() -> f64 { 1.0 }
fn d_ten() -> usize { 10 }
fn d_lr() -> f64 { 2e-4 }
fn d_grad_norm() -> f64 { 1.0 }
fn d_seed() -> u64 { 42 }
fn d_true() -> bool { true }
fn d_gens() -> usize { 4 }
fn d_completion() -> usize { 256 }
fn d_messages() -> String { "messages".into() }
fn d_roles() -> Vec<String> { vec!["assistant".into()] }
fn d_instruction() -> String { "instruction".into() }
fn d_input() -> String { "input".into() }
fn d_output() -> String { "output".into() }
fn d_text() -> String { "text".into() }
fn d_role() -> String { "role".into() }
fn d_content() -> String { "content".into() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axolotl_qwen3_lora_config_parses() {
        let y = r#"
base_model: Qwen/Qwen3-0.6B
chat_template: qwen3
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    split: train[:20%]
    field_messages: conversations
    message_property_mappings:
      role: from
      content: value
output_dir: ./outputs/out
sequence_len: 2048
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, down_proj, up_proj]
gradient_accumulation_steps: 2
micro_batch_size: 1
num_epochs: 1
lr_scheduler: cosine
learning_rate: 0.0002
bf16: auto
warmup_ratio: 0.03
"#;
        let c: Config = serde_yaml::from_str(y).unwrap();
        c.validate().unwrap();
        assert_eq!(c.lora_scale(), 2.0);
        assert_eq!(c.lora_targets().len(), 6);
        assert_eq!(c.warmup(100), 3);
        assert_eq!(c.datasets[0].message_property_mappings.as_ref().unwrap().role, "from");
    }

    #[test]
    fn unknown_axolotl_keys_are_rejected() {
        let y = "base_model: x\noutput_dir: o\ndatasets: [{path: p, type: completion}]\nsample_packing: true\n";
        assert!(serde_yaml::from_str::<Config>(y).is_err());
    }

    #[test]
    fn grpo_requires_rewards() {
        let y = "base_model: x\noutput_dir: o\ndatasets: [{path: p, type: completion}]\nrl: grpo\n";
        let c: Config = serde_yaml::from_str(y).unwrap();
        assert!(c.validate().is_err());
    }
}
