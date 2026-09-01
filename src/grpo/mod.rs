//! GRPO (group relative policy optimization) over a [`CausalLm`].
//!
//! One step: for every prompt, sample a group of `num_generations` completions
//! from the current policy, score each with the verifiers in `trl.reward_funcs`,
//! centre the rewards within the group (`math::group_advantages`), form
//! `-mean(A_i * logprob_i / len_i)` (`math::policy_gradient_loss`), and take a
//! norm-clipped AdamW step on `model.trainable_vars()`.
//!
//! When `trl.beta > 0` a KL term is added. A [`CausalLm`] cannot run without its
//! adapter, so there is no frozen reference model: the reference
//! log-probabilities are the policy's own, computed at the start of the step and
//! detached. The term therefore penalises drift from the pre-update policy, not
//! from the base model. With a single optimizer update per step the term and its
//! gradient are exactly zero at the point they are evaluated.

pub mod config;
pub mod math;
pub mod metrics;
pub mod policy;
pub mod reward;
pub mod sampler;
pub mod trainer;

pub use config::{GrpoConfig, Sampling};
pub use metrics::Metrics;
pub use policy::{LmPolicy, Policy};
pub use reward::{parse_rewards, Reward};
pub use sampler::Sampler;
pub use trainer::Trainer;

use crate::{CausalLm, Config, Prompt};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

/// Train `model` on `prompts` as described by `cfg`, writing the adapter to
/// `cfg.output_dir` (and `checkpoint-{step}` under it every `save_steps`).
pub fn run(
    cfg: &Config,
    model: &dyn CausalLm,
    text: &crate::data::Text,
    mut prompts: Vec<Prompt>,
) -> anyhow::Result<()> {
    anyhow::ensure!(!prompts.is_empty(), "grpo: no prompts");
    let policy = LmPolicy {
        model,
        tokenizer: &text.tokenizer,
        eos: Some(text.eos_id),
    };
    let mut trainer = Trainer::new(GrpoConfig::from(cfg), policy, parse_rewards(cfg)?)?;
    let out = Path::new(&cfg.output_dir);
    let max_steps = cfg.max_steps.unwrap_or(usize::MAX);
    let log_every = cfg.logging_steps.max(1);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut step = 0;
    'epochs: for _ in 0..cfg.num_epochs {
        prompts.shuffle(&mut rng);
        for batch in prompts.chunks(cfg.micro_batch_size) {
            if step >= max_steps {
                break 'epochs;
            }
            let m = trainer.step(batch)?;
            step += 1;
            if step % log_every == 0 {
                tracing::info!(
                    step,
                    reward = m.mean_reward,
                    reward_std = m.reward_std,
                    loss = m.loss,
                    kl = m.kl_loss,
                    completion_len = m.mean_completion_len,
                    grad_norm = m.grad_norm,
                );
            }
            if cfg.save_steps.is_some_and(|s| step % s == 0) {
                model.save_adapter(&out.join(format!("checkpoint-{step}")))?;
            }
        }
    }
    model.save_adapter(out)
}
