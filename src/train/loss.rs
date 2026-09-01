//! Next-token cross entropy.

use crate::IGNORE_INDEX;
use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_nn::ops::log_softmax;

/// Positions scored per log_softmax call, so the f32 `[b, chunk, vocab]`
/// temporary stays small whatever the sequence length.
const CHUNK: usize = 512;

/// Mean negative log-likelihood of `labels[:, 1:]` under `logits[:, :-1]`,
/// skipping positions labelled [`IGNORE_INDEX`], and the number of positions
/// scored. Softmax runs in f32 whatever the logits dtype. With nothing to
/// score the loss is zero.
pub fn causal_lm_loss(logits: &Tensor, labels: &Tensor) -> Result<(Tensor, usize)> {
    let (_, t, _) = logits.dims3()?;
    let logits = logits.narrow(1, 0, t - 1)?;
    let labels = labels.narrow(1, 1, t - 1)?;
    let scored = labels.ne(IGNORE_INDEX)?;
    let count = scored.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
    if count == 0 {
        return Ok((logits.zeros_like()?.sum_all()?.to_dtype(DType::F32)?, 0));
    }
    let targets = scored
        .where_cond(&labels, &labels.zeros_like()?)?
        .unsqueeze(2)?;
    let weights = scored.to_dtype(DType::F32)?;
    let mut nll = None;
    for start in (0..t - 1).step_by(CHUNK) {
        let len = CHUNK.min(t - 1 - start);
        let logp = log_softmax(
            &logits.narrow(1, start, len)?.to_dtype(DType::F32)?,
            D::Minus1,
        )?;
        let picked = logp
            .gather(&targets.narrow(1, start, len)?, 2)?
            .squeeze(2)?;
        let part = (picked * weights.narrow(1, start, len)?)?.sum_all()?;
        nll = Some(match nll {
            None => part,
            Some(acc) => (acc + part)?,
        });
    }
    let nll = nll.expect("t > 1 with a scored position");
    Ok(((nll / -(count as f64))?, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    /// Straight loop over the same definition, in f64.
    fn reference(logits: &[Vec<Vec<f32>>], labels: &[Vec<i64>]) -> (f64, usize) {
        let (mut sum, mut n) = (0.0, 0);
        for (b, rows) in logits.iter().enumerate() {
            for t in 0..rows.len() - 1 {
                let y = labels[b][t + 1];
                if y == IGNORE_INDEX {
                    continue;
                }
                let row: Vec<f64> = rows[t].iter().map(|&x| x as f64).collect();
                let max = row.iter().cloned().fold(f64::MIN, f64::max);
                let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
                sum += lse - row[y as usize];
                n += 1;
            }
        }
        (if n == 0 { 0.0 } else { sum / n as f64 }, n)
    }

    fn example() -> (Vec<Vec<Vec<f32>>>, Vec<Vec<i64>>) {
        let mut v = 0.0f32;
        let logits = (0..2)
            .map(|_| {
                (0..4)
                    .map(|_| {
                        (0..5)
                            .map(|_| {
                                v = (v * 1.7 + 0.3) % 2.9;
                                v - 1.4
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let labels = vec![
            vec![IGNORE_INDEX, 3, IGNORE_INDEX, 1],
            vec![2, IGNORE_INDEX, 4, 0],
        ];
        (logits, labels)
    }

    #[test]
    fn matches_reference_with_ignored_positions() {
        let (logits, labels) = example();
        let (want, n) = reference(&logits, &labels);
        let dev = Device::Cpu;
        let l = Tensor::new(logits, &dev).unwrap();
        let y = Tensor::new(labels, &dev).unwrap();
        let (loss, count) = causal_lm_loss(&l, &y).unwrap();
        assert_eq!(count, n);
        assert_eq!(n, 4);
        let got = loss.to_scalar::<f32>().unwrap() as f64;
        assert!((got - want).abs() < 1e-5, "got {got} want {want}");
    }

    #[test]
    fn chunks_agree_with_one_pass() {
        let dev = Device::Cpu;
        let t = 2 * CHUNK + 7;
        let l = Tensor::randn(0f32, 1.0, (1, t, 11), &dev).unwrap();
        let ids: Vec<i64> = (0..t).map(|i| ((i * 5 + 3) % 11) as i64).collect();
        let y = Tensor::from_vec(ids, (1, t), &dev).unwrap();
        let (loss, count) = causal_lm_loss(&l, &y).unwrap();
        assert_eq!(count, t - 1);
        let logits: Vec<Vec<Vec<f32>>> = l.to_vec3().unwrap();
        let labels: Vec<Vec<i64>> = y.to_vec2().unwrap();
        let (want, _) = reference(&logits, &labels);
        let got = loss.to_scalar::<f32>().unwrap() as f64;
        assert!((got - want).abs() < 1e-4, "got {got} want {want}");
    }

    #[test]
    fn nothing_scored_is_zero() {
        let dev = Device::Cpu;
        let l = Tensor::zeros((2, 4, 5), DType::F32, &dev).unwrap();
        let y = Tensor::full(IGNORE_INDEX, (2, 4), &dev).unwrap();
        let (loss, count) = causal_lm_loss(&l, &y).unwrap();
        assert_eq!(count, 0);
        assert_eq!(loss.to_scalar::<f32>().unwrap(), 0.0);
    }
}
