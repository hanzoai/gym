//! Group-relative advantages, the policy-gradient loss and the KL penalty.
//! Operates on reward scalars and log-probability tensors from any [`Policy`](super::Policy).
//!
//! Advantages follow TRL's GRPO trainer:
//!
//! ```text
//! advantages = rewards - mean_group(rewards)
//! if scale_rewards: advantages /= std_group(rewards) + 1e-4
//! ```
//!
//! and the loss is the single-iteration, unclipped form:
//!
//! ```text
//! loss = -mean_i(A_i * logprob_i / len_i)  (+ beta * KL)
//! ```

use hanzo_ml::{Result, Tensor};

/// Advantages for one group of `rewards` sharing a prompt: `r_i - mean(r)`,
/// divided by `std(r) + eps` when `scale`. The std is unbiased (`n - 1`),
/// matching `torch.std`.
pub fn group_advantages(rewards: &[f32], scale: bool, eps: f64) -> Vec<f32> {
    let n = rewards.len();
    debug_assert!(n >= 2, "group_advantages needs a group of >= 2");
    let mean = rewards.iter().sum::<f32>() / n as f32;
    if !scale {
        return rewards.iter().map(|&r| r - mean).collect();
    }
    let var = if n > 1 {
        rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / (n as f32 - 1.0)
    } else {
        0.0
    };
    let denom = var.sqrt() + eps as f32;
    rewards.iter().map(|&r| (r - mean) / denom).collect()
}

/// [`group_advantages`] over consecutive groups of `group_size` in `rewards`.
pub fn batch_advantages(rewards: &[f32], group_size: usize, scale: bool, eps: f64) -> Vec<f32> {
    debug_assert!(group_size >= 2);
    debug_assert_eq!(rewards.len() % group_size, 0);
    rewards
        .chunks(group_size)
        .flat_map(|g| group_advantages(g, scale, eps))
        .collect()
}

/// `-mean_i(A_i * logprob_i / len_i)`. `advantages` are detached constants;
/// `logprobs[i]` is the differentiable summed log-probability of completion `i`.
pub fn policy_gradient_loss(
    advantages: &[f32],
    logprobs: &[Tensor],
    lengths: &[usize],
) -> Result<Tensor> {
    assert_eq!(advantages.len(), logprobs.len());
    assert_eq!(advantages.len(), lengths.len());
    assert!(
        !advantages.is_empty(),
        "empty batch in policy_gradient_loss"
    );
    let mut total: Option<Tensor> = None;
    for ((&adv, lp), &len) in advantages.iter().zip(logprobs).zip(lengths) {
        let weighted = lp.affine(adv as f64 / len.max(1) as f64, 0.0)?;
        total = Some(match total {
            Some(acc) => (acc + weighted)?,
            None => weighted,
        });
    }
    total.unwrap().affine(-1.0 / advantages.len() as f64, 0.0)
}

/// `beta * mean_i k3_i` with `k3 = exp(d) - d - 1 >= 0` for
/// `d = ref_i/len_i - logprob_i/len_i`, the low-variance estimator of
/// `KL(policy || ref)` used by GRPO. `ref_logprobs` are detached constants.
pub fn kl_penalty(
    beta: f64,
    logprobs: &[Tensor],
    ref_logprobs: &[f32],
    lengths: &[usize],
) -> Result<Tensor> {
    assert_eq!(logprobs.len(), ref_logprobs.len());
    assert_eq!(logprobs.len(), lengths.len());
    assert!(!logprobs.is_empty());
    let mut total: Option<Tensor> = None;
    for ((lp, &ref_lp), &len) in logprobs.iter().zip(ref_logprobs).zip(lengths) {
        let len = len.max(1) as f64;
        let delta = lp.affine(-1.0 / len, ref_lp as f64 / len)?;
        let k3 = ((delta.exp()? - &delta)? - 1.0)?;
        total = Some(match total {
            Some(acc) => (acc + k3)?,
            None => k3,
        });
    }
    total.unwrap().affine(beta / logprobs.len() as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    #[test]
    fn advantages_are_zero_mean_within_group() {
        let a = group_advantages(&[1.0, 2.0, 3.0, 4.0], false, 1e-4);
        let sum: f32 = a.iter().sum();
        assert!(sum.abs() < 1e-5, "advantages should sum to ~0, got {sum}");
        assert!(a[0] < a[1] && a[1] < a[2] && a[2] < a[3]);
    }

    #[test]
    fn scaled_advantages_have_unit_scale() {
        let a = group_advantages(&[0.0, 10.0], true, 1e-4);
        assert!(
            (a[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2,
            "got {}",
            a[1]
        );
        assert!(
            (a[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2,
            "got {}",
            a[0]
        );
    }

    #[test]
    fn degenerate_group_gives_zero_advantage() {
        for v in group_advantages(&[5.0, 5.0, 5.0], true, 1e-4) {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn batch_splits_groups() {
        let a = batch_advantages(&[0.0, 2.0, 5.0, 5.0], 2, false, 1e-4);
        assert_eq!(a, vec![-1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn pg_loss_sign_and_finiteness() -> Result<()> {
        let dev = Device::Cpu;
        let lp0 = Tensor::new(-2.0f32, &dev)?;
        let lp1 = Tensor::new(-1.0f32, &dev)?;
        let v = policy_gradient_loss(&[-1.0, 1.0], &[lp0, lp1], &[1, 1])?.to_scalar::<f32>()?;
        assert!(v.is_finite());
        // -mean((-1)(-2) + (1)(-1)) = -1/2
        assert!((v + 0.5).abs() < 1e-5, "got {v}");
        Ok(())
    }

    #[test]
    fn kl_is_nonnegative_and_zero_at_match() -> Result<()> {
        let dev = Device::Cpu;
        let p = Tensor::new(-3.0f32, &dev)?;
        let v = kl_penalty(0.1, &[p], &[-3.0], &[1])?.to_scalar::<f32>()?;
        assert!(v.abs() < 1e-5, "KL at match should be 0, got {v}");
        let p2 = Tensor::new(-1.0f32, &dev)?;
        assert!(kl_penalty(0.1, &[p2], &[-3.0], &[1])?.to_scalar::<f32>()? > 0.0);
        Ok(())
    }
}
