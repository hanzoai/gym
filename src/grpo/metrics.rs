//! What one GRPO step reports.

#[derive(Debug, Clone)]
pub struct Metrics {
    /// Policy-gradient loss plus KL penalty; the value optimized.
    pub loss: f32,
    pub pg_loss: f32,
    /// 0.0 when the KL term is disabled.
    pub kl_loss: f32,
    /// Mean of the group reward means.
    pub mean_reward: f32,
    /// Mean of the within-group reward stds.
    pub reward_std: f32,
    pub mean_completion_len: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
    pub num_groups: usize,
    pub num_completions: usize,
}

impl Metrics {
    pub fn is_finite(&self) -> bool {
        self.loss.is_finite() && self.pg_loss.is_finite() && self.kl_loss.is_finite()
    }
}
