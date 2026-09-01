//! Next-token sampling during rollout: arg-max or temperature sampling from a
//! seeded generator, so a step replays exactly for a given seed.

use crate::grpo::config::{GrpoConfig, Sampling};
use hanzo_ml::{DType, Result, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct Sampler {
    rng: StdRng,
    mode: Sampling,
    temperature: f64,
}

impl Sampler {
    pub fn new(cfg: &GrpoConfig, seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            mode: cfg.sampling,
            temperature: cfg.temperature,
        }
    }

    /// Next token id for `logits` of shape `[vocab]`.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if self.mode == Sampling::Greedy {
            return Ok(logits.iter().position(|&l| l == max).unwrap_or(0) as u32);
        }
        let t = self.temperature as f32;
        let weights: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
        let mut u = self.rng.random::<f32>() * weights.iter().sum::<f32>();
        for (i, w) in weights.iter().enumerate() {
            u -= w;
            if u < 0.0 {
                return Ok(i as u32);
            }
        }
        Ok((weights.len() - 1) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    fn cfg(mode: Sampling, temperature: f64) -> GrpoConfig {
        GrpoConfig {
            sampling: mode,
            temperature,
            ..GrpoConfig::default()
        }
    }

    #[test]
    fn greedy_picks_argmax() -> Result<()> {
        let logits = Tensor::new(&[0.1f32, 3.0, -1.0, 2.9], &Device::Cpu)?;
        let mut s = Sampler::new(&cfg(Sampling::Greedy, 1.0), 0);
        assert_eq!(s.sample(&logits)?, 1);
        Ok(())
    }

    #[test]
    fn temperature_matches_softmax_frequencies() -> Result<()> {
        // softmax([0, ln 3]) = [0.25, 0.75]
        let logits = Tensor::new(&[0.0f32, 3f32.ln()], &Device::Cpu)?;
        let mut s = Sampler::new(&cfg(Sampling::Temperature, 1.0), 1);
        let n = 20_000;
        let ones = (0..n)
            .map(|_| s.sample(&logits))
            .filter(|t| matches!(t, Ok(1)))
            .count();
        let p = ones as f64 / n as f64;
        assert!((p - 0.75).abs() < 0.02, "P(1) = {p}");
        Ok(())
    }

    #[test]
    fn same_seed_replays() -> Result<()> {
        let logits = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu)?;
        let draw = |seed| -> Result<Vec<u32>> {
            let mut s = Sampler::new(&cfg(Sampling::Temperature, 0.7), seed);
            (0..16).map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(draw(3)?, draw(3)?);
        assert_ne!(draw(3)?, draw(4)?);
        Ok(())
    }
}
