//! Supervised fine-tuning loop.

use super::clip::{accumulate, clip_grad_norm};
use super::loss::causal_lm_loss;
use super::sched::lr_at;
use crate::data::collate;
use crate::{CausalLm, Config, Example};
use hanzo_ml::backprop::GradStore;
use hanzo_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

/// Train `model` on `train` per `cfg`, evaluating on `val` at the end of each
/// epoch and writing the adapter to `cfg.output_dir`.
pub fn sft(
    cfg: &Config,
    model: &dyn CausalLm,
    train: &[Example],
    val: &[Example],
    pad_id: u32,
) -> anyhow::Result<()> {
    fit(cfg, model, train, val, pad_id).map(|_| ())
}

/// Micro-batch losses, tokens and last gradient norm since the previous log line.
struct Window {
    losses: Vec<f32>,
    tokens: usize,
    norm: f64,
    since: Instant,
}

impl Window {
    fn new() -> Self {
        Self {
            losses: Vec::new(),
            tokens: 0,
            norm: 0.0,
            since: Instant::now(),
        }
    }

    fn mean(&self) -> f32 {
        self.losses.iter().sum::<f32>() / self.losses.len() as f32
    }

    fn log(&mut self, step: usize, total: usize, lr: f64) -> f32 {
        let loss = self.mean();
        let tps = self.tokens as f64 / self.since.elapsed().as_secs_f64();
        tracing::info!(
            step,
            total,
            loss,
            lr,
            grad_norm = self.norm,
            tokens_per_s = tps,
            "train"
        );
        *self = Self::new();
        loss
    }
}

/// The loop behind [`sft`]; returns the window-mean loss at every log line.
fn fit(
    cfg: &Config,
    model: &dyn CausalLm,
    train: &[Example],
    val: &[Example],
    pad_id: u32,
) -> anyhow::Result<Vec<f32>> {
    let vars = model.trainable_vars();
    let device = model.device();
    let out = Path::new(&cfg.output_dir);
    let micro = cfg.micro_batch_size;
    let accum = cfg.gradient_accumulation_steps;
    let per_epoch = train.len().div_ceil(micro * accum);
    let total = cfg
        .max_steps
        .unwrap_or(usize::MAX)
        .min(cfg.num_epochs * per_epoch);
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: cfg.weight_decay,
        },
    )?;
    let mut step = 0;
    let mut lr = 0.0;
    let mut window = Window::new();
    let mut logged = Vec::new();
    let mut done = total == 0;
    for epoch in 0..cfg.num_epochs {
        if done {
            break;
        }
        let mut order: Vec<usize> = (0..train.len()).collect();
        order.shuffle(&mut StdRng::seed_from_u64(cfg.seed + epoch as u64));
        let chunks: Vec<&[usize]> = order.chunks(micro).collect();
        let mut grads: Option<GradStore> = None;
        let mut pending = 0;
        for (i, ids) in chunks.iter().enumerate() {
            let examples: Vec<Example> = ids.iter().map(|&j| train[j].clone()).collect();
            let batch = collate(&examples, pad_id, device)?;
            let logits = model.forward(&batch.input_ids, batch.attention_mask.as_ref())?;
            let (loss, scored) = causal_lm_loss(&logits, &batch.labels)?;
            window.losses.push(loss.to_scalar::<f32>()?);
            window.tokens += batch.input_ids.elem_count();
            if scored > 0 {
                let g = (loss / accum as f64)?.backward()?;
                match grads.as_mut() {
                    Some(acc) => accumulate(acc, g, &vars)?,
                    None => grads = Some(g),
                }
            }
            pending += 1;
            if pending < accum && i + 1 < chunks.len() {
                continue;
            }
            if let Some(mut g) = grads.take() {
                window.norm = clip_grad_norm(&mut g, &vars, cfg.max_grad_norm)?;
                lr = lr_at(cfg, step, total);
                opt.set_learning_rate(lr);
                opt.step(&g)?;
            }
            pending = 0;
            step += 1;
            if step % cfg.logging_steps == 0 {
                logged.push(window.log(step, total, lr));
            }
            if cfg.save_steps.is_some_and(|s| step % s == 0) {
                model.save_adapter(&out.join(format!("checkpoint-{step}")))?;
            }
            if step >= total {
                done = true;
                break;
            }
        }
        if !val.is_empty() {
            let loss = evaluate(model, val, pad_id, micro)?;
            tracing::info!(epoch, step, loss, "eval");
        }
    }
    if !window.losses.is_empty() {
        logged.push(window.log(step, total, lr));
    }
    model.save_adapter(out)?;
    Ok(logged)
}

/// Token-weighted mean loss over `val`, no parameter update.
fn evaluate(
    model: &dyn CausalLm,
    val: &[Example],
    pad_id: u32,
    micro: usize,
) -> anyhow::Result<f32> {
    let (mut nll, mut n) = (0f64, 0usize);
    for examples in val.chunks(micro) {
        let batch = collate(examples, pad_id, model.device())?;
        let logits = model.forward(&batch.input_ids, batch.attention_mask.as_ref())?;
        let (loss, scored) = causal_lm_loss(&logits, &batch.labels)?;
        nll += loss.to_scalar::<f32>()? as f64 * scored as f64;
        n += scored;
    }
    Ok(if n == 0 { 0.0 } else { (nll / n as f64) as f32 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Device, Result, Tensor, Var};

    /// Embedding followed by one linear map: exactly a bigram model.
    struct Bigram {
        emb: Var,
        out: Var,
        dev: Device,
    }

    const VOCAB: usize = 16;

    impl Bigram {
        fn new(dev: &Device) -> Self {
            Self {
                emb: Var::randn(0f32, 0.5, (VOCAB, 8), dev).unwrap(),
                out: Var::randn(0f32, 0.5, (8, VOCAB), dev).unwrap(),
                dev: dev.clone(),
            }
        }
    }

    impl CausalLm for Bigram {
        fn forward(&self, ids: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
            let (b, t) = ids.dims2()?;
            let h = self.emb.index_select(&ids.flatten_all()?, 0)?;
            h.matmul(self.out.as_tensor())?.reshape((b, t, VOCAB))
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.emb.clone(), self.out.clone()]
        }
        fn device(&self) -> &Device {
            &self.dev
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn eos_token_id(&self) -> Option<u32> {
            None
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
        fn save_adapter(&self, dir: &Path) -> anyhow::Result<()> {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("adapter_model.safetensors"), b"")?;
            Ok(())
        }
    }

    /// Sequences that count upwards mod `VOCAB` from a varying start.
    fn counting(n: usize, len: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                let input_ids: Vec<u32> = (0..len).map(|k| ((i * 7 + k) % VOCAB) as u32).collect();
                let labels = input_ids.iter().map(|&x| x as i64).collect();
                Example { input_ids, labels }
            })
            .collect()
    }

    #[test]
    fn learns_the_counting_rule_and_saves() {
        let dir = std::env::temp_dir().join(format!("gym-sft-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg: Config = serde_yaml::from_str(&format!(
            "base_model: x\noutput_dir: {}\ndatasets: [{{path: p, type: completion}}]\n\
             micro_batch_size: 4\ngradient_accumulation_steps: 2\nnum_epochs: 10\nmax_steps: 40\n\
             learning_rate: 0.05\nlr_scheduler: constant\nlogging_steps: 5\nsave_steps: 10\n\
             max_grad_norm: 1.0\nseed: 7\n",
            dir.display()
        ))
        .unwrap();
        let dev = Device::Cpu;
        let model = Bigram::new(&dev);
        let train = counting(64, 8);
        let val = counting(8, 8);
        let windows = fit(&cfg, &model, &train, &val, 0).unwrap();
        assert_eq!(windows.len(), 8, "{windows:?}");
        let (first, last) = (windows[0], *windows.last().unwrap());
        assert!(last < 0.5 * first, "first {first} last {last}");
        for step in [10, 20, 30, 40] {
            let marker = dir
                .join(format!("checkpoint-{step}"))
                .join("adapter_model.safetensors");
            assert!(marker.is_file(), "{}", marker.display());
        }
        assert!(!dir.join("checkpoint-50").exists());
        assert!(dir.join("adapter_model.safetensors").is_file());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
