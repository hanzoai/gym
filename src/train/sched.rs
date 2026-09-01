//! Learning-rate schedule.

use crate::config::{Config, Scheduler};

/// Learning rate for optimizer step `step` of `total`: linear warmup over
/// `cfg.warmup(total)` steps to `learning_rate`, then decay per `lr_scheduler`.
pub fn lr_at(cfg: &Config, step: usize, total: usize) -> f64 {
    let peak = cfg.learning_rate;
    let warmup = cfg.warmup(total);
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let span = total.saturating_sub(warmup).max(1) as f64;
    let progress = ((step - warmup) as f64 / span).min(1.0);
    match cfg.lr_scheduler {
        Scheduler::Cosine => peak * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
        Scheduler::Linear => peak * (1.0 - progress),
        Scheduler::Constant => peak,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(scheduler: &str, warmup: usize) -> Config {
        serde_yaml::from_str(&format!(
            "base_model: x\noutput_dir: o\ndatasets: [{{path: p, type: completion}}]\n\
             learning_rate: 1.0\nlr_scheduler: {scheduler}\nwarmup_steps: {warmup}\n"
        ))
        .unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn warmup_then_cosine() {
        let c = cfg("cosine", 10);
        assert!(close(lr_at(&c, 0, 110), 0.1));
        assert!(close(lr_at(&c, 9, 110), 1.0));
        assert!(close(lr_at(&c, 10, 110), 1.0));
        assert!(close(lr_at(&c, 60, 110), 0.5));
        assert!(close(lr_at(&c, 110, 110), 0.0));
    }

    #[test]
    fn linear_and_constant() {
        let c = cfg("linear", 0);
        assert!(close(lr_at(&c, 0, 100), 1.0));
        assert!(close(lr_at(&c, 25, 100), 0.75));
        assert!(close(lr_at(&c, 100, 100), 0.0));
        let c = cfg("constant", 4);
        assert!(close(lr_at(&c, 1, 100), 0.5));
        assert!(close(lr_at(&c, 99, 100), 1.0));
    }
}
