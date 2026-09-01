//! One GRPO step: rollout, reward, group-relative advantage, loss, clipped AdamW update.

use crate::grpo::config::GrpoConfig;
use crate::grpo::math;
use crate::grpo::metrics::Metrics;
use crate::grpo::policy::Policy;
use crate::grpo::reward::Reward;
use crate::grpo::sampler::Sampler;
use crate::Prompt;
use hanzo_ml::backprop::GradStore;
use hanzo_ml::{DType, Var};
use hanzo_nn::{AdamW, Optimizer, ParamsAdamW};

pub struct Trainer<P: Policy> {
    cfg: GrpoConfig,
    policy: P,
    rewards: Vec<Reward>,
    vars: Vec<Var>,
    optimizer: AdamW,
    steps: usize,
    /// Advances every step so rollouts do not replay.
    seed: u64,
}

impl<P: Policy> Trainer<P> {
    pub fn new(cfg: GrpoConfig, policy: P, rewards: Vec<Reward>) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            !rewards.is_empty(),
            "grpo needs at least one reward function"
        );
        let vars = policy.trainable_vars();
        anyhow::ensure!(
            !vars.is_empty(),
            "grpo: the policy has no trainable variables"
        );
        let params = ParamsAdamW {
            lr: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
            ..Default::default()
        };
        let optimizer = AdamW::new(vars.clone(), params)?;
        Ok(Self {
            seed: cfg.seed,
            cfg,
            policy,
            rewards,
            vars,
            optimizer,
            steps: 0,
        })
    }

    pub fn policy(&self) -> &P {
        &self.policy
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// `sum_f w_f * r_f(prompt, completion)`.
    fn score(&self, prompt: &Prompt, tokens: &[u32], text: &str) -> anyhow::Result<f32> {
        self.rewards
            .iter()
            .map(|r| Ok(r.weight * r.score(prompt, tokens, text)?))
            .sum()
    }

    /// One update over `prompts`; each prompt yields a group of `group_size` completions.
    pub fn step(&mut self, prompts: &[Prompt]) -> anyhow::Result<Metrics> {
        anyhow::ensure!(!prompts.is_empty(), "grpo: step with no prompts");
        let g = self.cfg.group_size;
        let n = prompts.len() * g;

        let mut rewards = Vec::with_capacity(n);
        let mut lengths = Vec::with_capacity(n);
        let mut logprobs = Vec::with_capacity(n);
        let mut group_mean = Vec::with_capacity(prompts.len());
        let mut group_std = Vec::with_capacity(prompts.len());
        for (pi, prompt) in prompts.iter().enumerate() {
            let seed = self
                .seed
                .wrapping_add((pi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut sampler = Sampler::new(&self.cfg, seed);
            let group = self.policy.sample_group(
                &prompt.input_ids,
                g,
                self.cfg.max_completion_len,
                &mut sampler,
            )?;
            anyhow::ensure!(
                group.len() == g,
                "grpo: policy returned {} completions, expected {g}",
                group.len()
            );

            let mut scores = Vec::with_capacity(g);
            for tokens in &group {
                let text = self.policy.decode(tokens)?;
                scores.push(self.score(prompt, tokens, &text)?);
                lengths.push(tokens.len());
            }
            let mean = scores.iter().sum::<f32>() / g as f32;
            let var = scores.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / (g as f32 - 1.0);
            group_mean.push(mean);
            group_std.push(var.sqrt());
            rewards.extend(scores);
            logprobs.extend(self.policy.group_logprob(&prompt.input_ids, &group)?);
        }

        let advantages =
            math::batch_advantages(&rewards, g, self.cfg.scale_rewards, self.cfg.advantage_eps);
        let pg_loss = math::policy_gradient_loss(&advantages, &logprobs, &lengths)?;
        let pg_val = pg_loss.to_scalar::<f32>()?;
        let (loss, kl_val) = match self.cfg.kl_beta.filter(|&b| b > 0.0) {
            Some(beta) => {
                let reference: Vec<f32> = logprobs
                    .iter()
                    .map(|lp| lp.to_scalar::<f32>())
                    .collect::<hanzo_ml::Result<_>>()?;
                let kl = math::kl_penalty(beta, &logprobs, &reference, &lengths)?;
                let kl_val = kl.to_scalar::<f32>()?;
                ((&pg_loss + &kl)?, kl_val)
            }
            None => (pg_loss, 0.0),
        };
        let loss_val = loss.to_scalar::<f32>()?;

        let mut grad_norm = 0.0;
        if loss_val.is_finite() {
            let mut grads = loss.backward()?;
            grad_norm = clip(&mut grads, &self.vars, self.cfg.max_grad_norm)? as f32;
            self.optimizer.step(&grads)?;
        }
        self.steps += 1;
        self.seed = self.seed.wrapping_add(prompts.len() as u64 + 1);

        let groups = prompts.len() as f32;
        Ok(Metrics {
            loss: loss_val,
            pg_loss: pg_val,
            kl_loss: kl_val,
            mean_reward: group_mean.iter().sum::<f32>() / groups,
            reward_std: group_std.iter().sum::<f32>() / groups,
            mean_completion_len: lengths.iter().sum::<usize>() as f32 / n as f32,
            grad_norm,
            num_groups: prompts.len(),
            num_completions: n,
        })
    }
}

/// Scale the gradients of `vars` so their global L2 norm is at most `max`
/// (`max <= 0` never scales). Returns the norm before scaling.
pub fn clip(grads: &mut GradStore, vars: &[Var], max: f64) -> hanzo_ml::Result<f64> {
    let mut sq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v) {
            sq += g
                .sqr()?
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()? as f64;
        }
    }
    let norm = sq.sqrt();
    if max > 0.0 && norm > max {
        let scale = max / (norm + 1e-6);
        for v in vars {
            if let Some(g) = grads.get(v) {
                let g = (g * scale)?;
                grads.insert(v, g);
            }
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    //! The toy policy is a contextual bandit: one learnable `[max_len, vocab]`
    //! logits table, one categorical per output position, prompt-independent.
    //! It exercises the whole step: rollout through [`Sampler`], a
    //! differentiable `log_softmax` + gather log-probability, the advantage and
    //! loss math, clipping and AdamW.
    use super::*;
    use crate::grpo::config::Sampling;
    use crate::grpo::policy::fake::{tokenizer, Bigram};
    use crate::grpo::policy::LmPolicy;
    use hanzo_ml::{Device, Tensor};
    use hanzo_nn::ops::log_softmax;

    struct Toy {
        logits: Var,
        max_len: usize,
    }

    impl Toy {
        fn new(vocab: usize, max_len: usize) -> Self {
            Self {
                logits: Var::zeros((max_len, vocab), DType::F32, &Device::Cpu).unwrap(),
                max_len,
            }
        }
    }

    impl Policy for Toy {
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.logits.clone()]
        }

        fn sample_group(
            &self,
            _: &[u32],
            group_size: usize,
            max_len: usize,
            sampler: &mut Sampler,
        ) -> anyhow::Result<Vec<Vec<u32>>> {
            let steps = max_len.min(self.max_len);
            (0..group_size)
                .map(|_| {
                    (0..steps)
                        .map(|pos| Ok(sampler.sample(&self.logits.get(pos)?)?))
                        .collect()
                })
                .collect()
        }

        fn sequence_logprob(&self, _: &[u32], completion: &[u32]) -> anyhow::Result<Tensor> {
            let logp = log_softmax(self.logits.as_tensor(), 1)?;
            let idx = Tensor::from_slice(completion, (completion.len(), 1), &Device::Cpu)?;
            Ok(logp
                .narrow(0, 0, completion.len())?
                .gather(&idx, 1)?
                .sum_all()?)
        }
    }

    fn prompt(ids: &[u32]) -> Prompt {
        Prompt {
            input_ids: ids.to_vec(),
            text: String::new(),
            answer: None,
        }
    }

    #[test]
    fn toy_reward_increases() -> anyhow::Result<()> {
        let (vocab, max_len, target) = (5, 4, 3);
        let cfg = GrpoConfig {
            group_size: 16,
            max_completion_len: max_len,
            sampling: Sampling::Temperature,
            temperature: 1.0,
            scale_rewards: true,
            advantage_eps: 1e-4,
            kl_beta: None,
            learning_rate: 0.5,
            weight_decay: 0.0,
            max_grad_norm: 0.0,
            seed: 42,
        };
        let mut trainer = Trainer::new(cfg, Toy::new(vocab, max_len), vec![Reward::token(target)])?;
        let prompts = [prompt(&[1, 2]), prompt(&[0]), prompt(&[4, 4, 4])];
        let first = trainer.step(&prompts)?;
        assert!(first.is_finite(), "{first:?}");
        assert_eq!((first.num_groups, first.num_completions), (3, 48));
        let mut last = first.clone();
        for _ in 0..40 {
            last = trainer.step(&prompts)?;
            assert!(last.is_finite(), "{last:?}");
        }
        println!(
            "[toy] reward {:.3} -> {:.3} of {max_len}",
            first.mean_reward, last.mean_reward
        );
        assert!(
            last.mean_reward > first.mean_reward + 0.5,
            "{} -> {}",
            first.mean_reward,
            last.mean_reward
        );
        assert!(
            last.mean_reward > max_len as f32 * 0.6,
            "{} vs {max_len}",
            last.mean_reward
        );
        Ok(())
    }

    #[test]
    fn toy_with_kl_stays_finite() -> anyhow::Result<()> {
        let cfg = GrpoConfig {
            group_size: 8,
            max_completion_len: 3,
            kl_beta: Some(0.05),
            learning_rate: 0.2,
            seed: 7,
            ..GrpoConfig::default()
        };
        let mut trainer = Trainer::new(cfg, Toy::new(5, 3), vec![Reward::token(2)])?;
        let prompts = [prompt(&[1]), prompt(&[2])];
        let mut last = trainer.step(&prompts)?;
        for _ in 0..10 {
            last = trainer.step(&prompts)?;
            assert!(last.is_finite(), "{last:?}");
        }
        assert!(last.kl_loss >= 0.0);
        assert_eq!(trainer.steps(), 11);
        Ok(())
    }

    #[test]
    fn clip_bounds_global_norm() -> hanzo_ml::Result<()> {
        let a = Var::from_slice(&[3.0f32, 0.0], 2, &Device::Cpu)?;
        let b = Var::from_slice(&[0.0f32, 4.0], 2, &Device::Cpu)?;
        let loss = (a.sqr()?.sum_all()? + b.sqr()?.sum_all()?)?.affine(0.5, 0.0)?;
        let mut grads = loss.backward()?;
        let vars = vec![a.clone(), b.clone()];
        assert!((clip(&mut grads, &vars, 0.0)? - 5.0).abs() < 1e-5);
        assert_eq!(grads.get(&a).unwrap().to_vec1::<f32>()?, vec![3.0, 0.0]);
        let before = clip(&mut grads, &vars, 1.0)?;
        assert!((before - 5.0).abs() < 1e-5);
        let after = clip(&mut grads, &vars, 1.0)?;
        assert!((after - 1.0).abs() < 1e-5, "{after}");
        Ok(())
    }

    #[test]
    fn lm_step_raises_rewarded_token() -> anyhow::Result<()> {
        // Vocab: unk q 3 7 <eos>. Prompt "q", answer "3", one-token completions.
        let model = Bigram::uniform(5, Some(4));
        let tk = tokenizer(&["<unk>", "q", "3", "7", "<eos>"]);
        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: Some(4),
        };
        let cfg = GrpoConfig {
            group_size: 8,
            max_completion_len: 1,
            learning_rate: 0.3,
            max_grad_norm: 1.0,
            seed: 1,
            ..GrpoConfig::default()
        };
        let mut trainer = Trainer::new(cfg, policy, vec![Reward::parse("exact_match", 1.0)?])?;
        let prompts = [Prompt {
            input_ids: vec![1],
            text: "q".into(),
            answer: Some("3".into()),
        }];
        let before = model.probs(1)[2];
        let mut m = trainer.step(&prompts)?;
        assert!(m.grad_norm > 0.0, "{m:?}");
        for _ in 0..30 {
            m = trainer.step(&prompts)?;
        }
        let after = model.probs(1)[2];
        println!(
            "[lm] P(3 | q) {before:.3} -> {after:.3}, reward {:.3}",
            m.mean_reward
        );
        assert!(after > 0.6 && after > before, "{before} -> {after}");
        assert!(
            model.probs(0).iter().all(|&p| (p - 0.2).abs() < 1e-6),
            "rows never seen stay uniform"
        );
        Ok(())
    }
}
