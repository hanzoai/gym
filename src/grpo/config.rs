//! GRPO settings, derived from `cfg.trl` and the shared optimizer keys.

use crate::Config;

/// How the next token is chosen during rollout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sampling {
    /// Arg-max. Every completion in a group is identical, so advantages are zero;
    /// useful only for debugging.
    Greedy,
    /// Sample from the softmax of `logits / temperature`.
    #[default]
    Temperature,
}

#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Completions sampled per prompt (`trl.num_generations`). Must be >= 2.
    pub group_size: usize,
    /// Maximum tokens generated per completion.
    pub max_completion_len: usize,
    pub sampling: Sampling,
    /// Only used when `sampling == Temperature`.
    pub temperature: f64,
    /// Divide the group-centred reward by `std + advantage_eps`.
    pub scale_rewards: bool,
    pub advantage_eps: f64,
    /// KL coefficient (`trl.beta`). The reference log-probabilities are the
    /// policy's own, computed at the start of the step and detached; the term
    /// is a penalty against the pre-update policy, not KL to the frozen base
    /// model, because a [`crate::CausalLm`] cannot run without its adapter.
    /// `None` disables the term.
    pub kl_beta: Option<f64>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    /// Global gradient-norm clip; `<= 0` disables clipping.
    pub max_grad_norm: f64,
    pub seed: u64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            max_completion_len: 64,
            sampling: Sampling::Temperature,
            temperature: 1.0,
            scale_rewards: true,
            advantage_eps: 1e-4,
            kl_beta: None,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            seed: 0,
        }
    }
}

impl From<&Config> for GrpoConfig {
    fn from(cfg: &Config) -> Self {
        let t = &cfg.trl;
        Self {
            group_size: t.num_generations,
            max_completion_len: t.max_completion_length,
            sampling: Sampling::Temperature,
            temperature: t.temperature,
            scale_rewards: t.scale_rewards,
            advantage_eps: 1e-4,
            kl_beta: (t.beta > 0.0).then_some(t.beta),
            learning_rate: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
            max_grad_norm: cfg.max_grad_norm,
            seed: cfg.seed,
        }
    }
}

impl GrpoConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.group_size >= 2,
            "group_size must be >= 2 (got {}); group-relative advantages need two completions",
            self.group_size
        );
        anyhow::ensure!(
            self.max_completion_len > 0,
            "max_completion_len must be >= 1"
        );
        if self.sampling == Sampling::Temperature {
            anyhow::ensure!(
                self.temperature > 0.0,
                "temperature must be > 0 (got {})",
                self.temperature
            );
        }
        if let Some(beta) = self.kl_beta {
            anyhow::ensure!(beta >= 0.0, "kl_beta must be >= 0 (got {beta})");
        }
        Ok(())
    }

    pub fn kl_active(&self) -> bool {
        matches!(self.kl_beta, Some(b) if b > 0.0)
    }
}
