//! Gradient norm clipping and accumulation across micro-batches.

use hanzo_ml::backprop::GradStore;
use hanzo_ml::{DType, Result, Var};

/// Scale every gradient in `grads` so their global L2 norm is at most
/// `max_norm`. Returns the norm before clipping.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
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
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for v in vars {
            let scaled = match grads.get(v) {
                Some(g) => (g * scale)?,
                None => continue,
            };
            grads.insert(v, scaled);
        }
    }
    Ok(norm)
}

/// Add the gradients in `from` to those in `into`, per variable.
pub fn accumulate(into: &mut GradStore, from: GradStore, vars: &[Var]) -> Result<()> {
    for v in vars {
        let Some(g) = from.get(v) else { continue };
        let sum = match into.get(v) {
            Some(acc) => (acc + g)?,
            None => g.clone(),
        };
        into.insert(v, sum);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{Device, Tensor};

    fn grads_for(vars: &[Var], k: f64) -> GradStore {
        // d/dv sum(k * v^2 / 2) = k * v
        let loss = vars
            .iter()
            .map(|v| (v.sqr().unwrap().sum_all().unwrap() * (k / 2.0)).unwrap())
            .reduce(|a, b| (a + b).unwrap())
            .unwrap();
        loss.backward().unwrap()
    }

    fn vars() -> Vec<Var> {
        let dev = Device::Cpu;
        vec![
            Var::new(&[3f32, 0.0], &dev).unwrap(),
            Var::new(&[[0f32, 4.0]], &dev).unwrap(),
        ]
    }

    #[test]
    fn clips_to_max_norm_and_reports_pre_clip() {
        let vars = vars();
        let mut g = grads_for(&vars, 1.0);
        let before = clip_grad_norm(&mut g, &vars, 1.0).unwrap();
        assert!((before - 5.0).abs() < 1e-5);
        let after = clip_grad_norm(&mut g, &vars, 10.0).unwrap();
        assert!((after - 1.0).abs() < 1e-5, "{after}");
        let x: Vec<f32> = g.get(&vars[0]).unwrap().to_vec1().unwrap();
        assert!((x[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn under_max_norm_is_untouched() {
        let vars = vars();
        let mut g = grads_for(&vars, 1.0);
        assert!((clip_grad_norm(&mut g, &vars, 6.0).unwrap() - 5.0).abs() < 1e-5);
        let x: Vec<f32> = g.get(&vars[0]).unwrap().to_vec1().unwrap();
        assert_eq!(x, vec![3.0, 0.0]);
    }

    #[test]
    fn accumulate_sums_per_var() {
        let vars = vars();
        let mut acc = grads_for(&vars, 1.0);
        accumulate(&mut acc, grads_for(&vars, 2.0), &vars).unwrap();
        let x: Vec<f32> = acc.get(&vars[0]).unwrap().to_vec1().unwrap();
        assert_eq!(x, vec![9.0, 0.0]);
        let y: Vec<Vec<f32>> = acc.get(&vars[1]).unwrap().to_vec2().unwrap();
        assert_eq!(y, vec![vec![0.0, 12.0]]);
        let t = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        assert!(acc.get(&t).is_none());
    }
}
