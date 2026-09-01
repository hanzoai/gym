//! What GRPO needs from a model: rollout, differentiable scoring, decoding.

use crate::grpo::sampler::Sampler;
use crate::{Cache, CausalLm};
use hanzo_ml::{DType, Tensor, Var, D};
use hanzo_nn::ops::log_softmax;
use tokenizers::Tokenizer;

pub trait Policy {
    fn trainable_vars(&self) -> Vec<Var>;

    /// `group_size` completions of at most `max_len` tokens for `prompt`, drawn
    /// with `sampler`. No autograd graph is kept.
    fn sample_group(
        &self,
        prompt: &[u32],
        group_size: usize,
        max_len: usize,
        sampler: &mut Sampler,
    ) -> anyhow::Result<Vec<Vec<u32>>>;

    /// `sum_t log p(c_t | prompt, c_<t)` as a scalar tensor differentiable
    /// w.r.t. [`Policy::trainable_vars`].
    fn sequence_logprob(&self, prompt: &[u32], completion: &[u32]) -> anyhow::Result<Tensor>;

    /// [`Policy::sequence_logprob`] for every completion of one group.
    fn group_logprob(
        &self,
        prompt: &[u32],
        completions: &[Vec<u32>],
    ) -> anyhow::Result<Vec<Tensor>> {
        completions
            .iter()
            .map(|c| self.sequence_logprob(prompt, c))
            .collect()
    }

    /// Text for `tokens`, used by text rewards. Empty when the policy has no tokenizer.
    fn decode(&self, _tokens: &[u32]) -> anyhow::Result<String> {
        Ok(String::new())
    }
}

/// A [`CausalLm`] as a GRPO policy.
///
/// Rollout runs through the model's KV cache: the prompt is prefilled once for
/// the whole group, then each generated token is one `forward_cached` over the
/// batch. Rows that hit eos keep receiving the pad token so the batch stays
/// rectangular; their later outputs are discarded.
pub struct LmPolicy<'a> {
    pub model: &'a dyn CausalLm,
    pub tokenizer: &'a Tokenizer,
    pub eos: Option<u32>,
}

impl Policy for LmPolicy<'_> {
    fn trainable_vars(&self) -> Vec<Var> {
        self.model.trainable_vars()
    }

    fn sample_group(
        &self,
        prompt: &[u32],
        group_size: usize,
        max_len: usize,
        sampler: &mut Sampler,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        anyhow::ensure!(
            !prompt.is_empty(),
            "grpo: a prompt needs at least one token"
        );
        let dev = self.model.device();
        let pad = self.eos.unwrap_or(0);
        let mut cache = Cache::default();
        let mut ids = Tensor::from_vec(prompt.repeat(group_size), (group_size, prompt.len()), dev)?;
        let mut out = vec![Vec::new(); group_size];
        let mut live = vec![true; group_size];
        for _ in 0..max_len {
            let logits = self.model.forward_cached(&ids, &mut cache)?;
            let last = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
            let mut next = Vec::with_capacity(group_size);
            for i in 0..group_size {
                let tok = if live[i] {
                    let tok = sampler.sample(&last.get(i)?)?;
                    out[i].push(tok);
                    live[i] &= Some(tok) != self.eos;
                    tok
                } else {
                    pad
                };
                next.push(tok);
            }
            if !live.contains(&true) {
                break;
            }
            ids = Tensor::from_vec(next, (group_size, 1), dev)?;
        }
        Ok(out)
    }

    fn sequence_logprob(&self, prompt: &[u32], completion: &[u32]) -> anyhow::Result<Tensor> {
        Ok(self
            .group_logprob(prompt, &[completion.to_vec()])?
            .remove(0))
    }

    /// One forward over the group, completions padded right with an attention mask.
    fn group_logprob(
        &self,
        prompt: &[u32],
        completions: &[Vec<u32>],
    ) -> anyhow::Result<Vec<Tensor>> {
        let dev = self.model.device();
        let p = prompt.len();
        anyhow::ensure!(p > 0, "grpo: a prompt needs at least one token");
        let g = completions.len();
        let t = p + completions.iter().map(Vec::len).max().unwrap_or(0);
        let pad = self.eos.unwrap_or(0);
        let mut ids = Vec::with_capacity(g * t);
        let mut mask = Vec::with_capacity(g * t);
        for c in completions {
            ids.extend_from_slice(prompt);
            ids.extend_from_slice(c);
            ids.resize(ids.len() + t - p - c.len(), pad);
            mask.extend(std::iter::repeat_n(1u8, p + c.len()));
            mask.extend(std::iter::repeat_n(0u8, t - p - c.len()));
        }
        let padded = mask.contains(&0);
        let ids = Tensor::from_vec(ids, (g, t), dev)?;
        let mask = if padded {
            Some(Tensor::from_vec(mask, (g, t), dev)?)
        } else {
            None
        };
        let logits = self
            .model
            .forward(&ids, mask.as_ref())?
            .to_dtype(DType::F32)?;
        let logp = log_softmax(&logits, D::Minus1)?;
        completions
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if c.is_empty() {
                    return Ok(Tensor::zeros((), DType::F32, dev)?);
                }
                let rows = logp.get(i)?.narrow(0, p - 1, c.len())?;
                let idx = Tensor::from_slice(c, (c.len(), 1), dev)?;
                Ok(rows.gather(&idx, 1)?.sum_all()?)
            })
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! A bigram table as a [`CausalLm`]: `logits[b, t, :] = table[input_ids[b, t], :]`.
    use super::*;
    use hanzo_ml::Device;

    pub struct Bigram {
        pub table: Var,
        pub eos: Option<u32>,
    }

    impl Bigram {
        pub fn new(rows: &[Vec<f32>], eos: Option<u32>) -> Self {
            let v = rows.len();
            let flat: Vec<f32> = rows.concat();
            Self {
                table: Var::from_vec(flat, (v, v), &Device::Cpu).unwrap(),
                eos,
            }
        }

        pub fn uniform(vocab: usize, eos: Option<u32>) -> Self {
            Self {
                table: Var::zeros((vocab, vocab), DType::F32, &Device::Cpu).unwrap(),
                eos,
            }
        }

        /// Softmax row `from` of the table.
        pub fn probs(&self, from: u32) -> Vec<f32> {
            let row = self.table.get(from as usize).unwrap();
            hanzo_nn::ops::softmax(&row, 0).unwrap().to_vec1().unwrap()
        }
    }

    impl CausalLm for Bigram {
        fn forward(&self, input_ids: &Tensor, _mask: Option<&Tensor>) -> hanzo_ml::Result<Tensor> {
            let (b, t) = input_ids.dims2()?;
            let v = self.vocab_size();
            self.table
                .index_select(&input_ids.flatten_all()?, 0)?
                .reshape((b, t, v))
        }
        fn forward_cached(
            &self,
            input_ids: &Tensor,
            cache: &mut Cache,
        ) -> hanzo_ml::Result<Tensor> {
            cache.len += input_ids.dim(1)?;
            self.forward(input_ids, None)
        }
        fn trainable_vars(&self) -> Vec<Var> {
            vec![self.table.clone()]
        }
        fn device(&self) -> &Device {
            self.table.device()
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn eos_token_id(&self) -> Option<u32> {
            self.eos
        }
        fn vocab_size(&self) -> usize {
            self.table.dim(0).unwrap()
        }
        fn save_adapter(&self, _dir: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Word-level tokenizer over `words`; id = index, `<eos>` is special.
    pub fn tokenizer(words: &[&str]) -> Tokenizer {
        let vocab: Vec<String> = words
            .iter()
            .enumerate()
            .map(|(i, w)| format!("{w:?}: {i}"))
            .collect();
        let eos = words
            .iter()
            .position(|&w| w == "<eos>")
            .expect("<eos> in vocab");
        let json = format!(
            r#"{{"version":"1.0","added_tokens":[{{"id":{eos},"content":"<eos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}}],
            "pre_tokenizer":{{"type":"Whitespace"}},
            "model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"{}"}}}}"#,
            vocab.join(","),
            words[0]
        );
        json.parse().expect("tokenizer json")
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{tokenizer, Bigram};
    use super::*;
    use crate::grpo::config::{GrpoConfig, Sampling};
    use hanzo_ml::Device;

    const BIG: f32 = 30.0;

    #[test]
    fn sequence_logprob_matches_hand_computation() -> anyhow::Result<()> {
        let rows = vec![
            vec![1.0, 2.0, 0.5],
            vec![0.0, -1.0, 3.0],
            vec![0.0, 0.0, 0.0],
        ];
        let model = Bigram::new(&rows, None);
        let tk = tokenizer(&["a", "b", "<eos>"]);
        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: None,
        };
        let got = policy.sequence_logprob(&[0], &[1, 2])?.to_scalar::<f32>()?;
        let lsm = |r: &[f32], i: usize| r[i] - r.iter().map(|x| x.exp()).sum::<f32>().ln();
        let want = lsm(&rows[0], 1) + lsm(&rows[1], 2);
        assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        Ok(())
    }

    #[test]
    fn batched_group_equals_single() -> anyhow::Result<()> {
        let rows = vec![
            vec![1.0, 2.0, 0.5, 0.1],
            vec![0.0, -1.0, 3.0, 0.2],
            vec![2.0, 0.0, 0.0, 1.0],
            vec![0.0; 4],
        ];
        let model = Bigram::new(&rows, Some(3));
        let tk = tokenizer(&["a", "b", "c", "<eos>"]);
        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: Some(3),
        };
        let group = vec![vec![1, 2, 3], vec![2], vec![0, 0, 1, 3]];
        let batched = policy.group_logprob(&[0, 1], &group)?;
        for (c, b) in group.iter().zip(&batched) {
            let single = policy.sequence_logprob(&[0, 1], c)?.to_scalar::<f32>()?;
            assert!((b.to_scalar::<f32>()? - single).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn logprob_is_differentiable() -> anyhow::Result<()> {
        let model = Bigram::uniform(3, None);
        let tk = tokenizer(&["a", "b", "<eos>"]);
        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: None,
        };
        let lp = policy.sequence_logprob(&[0], &[1])?;
        let grads = lp.backward()?;
        let g = grads
            .get(&model.table)
            .expect("gradient on table")
            .to_vec2::<f32>()?;
        // d/dlogit log softmax(row 0)[1] = onehot(1) - softmax = [-1/3, 2/3, -1/3]
        assert!((g[0][1] - 2.0 / 3.0).abs() < 1e-5 && (g[0][0] + 1.0 / 3.0).abs() < 1e-5);
        assert!(g[1].iter().all(|&x| x == 0.0));
        Ok(())
    }

    #[test]
    fn sample_group_stops_at_eos_and_max_len() -> anyhow::Result<()> {
        // 0 -> 1 -> eos(2), near-deterministically.
        let rows = vec![vec![0.0, BIG, 0.0], vec![0.0, 0.0, BIG], vec![0.0; 3]];
        let model = Bigram::new(&rows, Some(2));
        let tk = tokenizer(&["a", "b", "<eos>"]);
        let cfg = GrpoConfig {
            sampling: Sampling::Temperature,
            temperature: 1.0,
            ..GrpoConfig::default()
        };
        let mut sampler = Sampler::new(&cfg, 0);

        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: Some(2),
        };
        let group = policy.sample_group(&[0], 4, 5, &mut sampler)?;
        assert_eq!(group.len(), 4);
        assert!(group.iter().all(|c| c == &[1, 2]), "{group:?}");

        let none = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: None,
        };
        let group = none.sample_group(&[0], 3, 5, &mut sampler)?;
        assert_eq!(group.len(), 3);
        assert!(group.iter().all(|c| c.len() == 5), "{group:?}");
        assert_eq!(policy.decode(&[1, 2])?, "b");
        Ok(())
    }

    #[test]
    fn cached_greedy_rollout_matches_full_forward() -> anyhow::Result<()> {
        // 1 -> 4 -> 0 -> 3 -> 1 -> ..., eos (2) unreachable, so every row runs to max_len.
        let rows = vec![
            vec![0.0, 1.0, 0.0, 5.0, 2.0],
            vec![0.0, 0.0, 0.0, 1.0, 4.0],
            vec![0.0; 5],
            vec![0.0, 3.0, 0.0, 0.0, 1.0],
            vec![6.0, 0.0, 0.0, 1.0, 0.0],
        ];
        let model = Bigram::new(&rows, Some(2));
        let tk = tokenizer(&["a", "b", "<eos>", "c", "d"]);
        let policy = LmPolicy {
            model: &model,
            tokenizer: &tk,
            eos: Some(2),
        };
        let cfg = GrpoConfig {
            sampling: Sampling::Greedy,
            ..GrpoConfig::default()
        };
        let mut sampler = Sampler::new(&cfg, 0);
        let (prompt, max_len) = ([0u32, 1], 7);
        let group = policy.sample_group(&prompt, 3, max_len, &mut sampler)?;

        let mut seq = prompt.to_vec();
        for _ in 0..max_len {
            let ids = Tensor::from_slice(&seq, (1, seq.len()), &Device::Cpu)?;
            let last = model.forward(&ids, None)?.get(0)?.get(seq.len() - 1)?;
            seq.push(last.argmax(0)?.to_scalar::<u32>()?);
        }
        let want = seq[prompt.len()..].to_vec();
        assert_eq!(want, vec![4, 0, 3, 1, 4, 0, 3]);
        assert!(group.iter().all(|c| c == &want), "{group:?}");
        Ok(())
    }
}
