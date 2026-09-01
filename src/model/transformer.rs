//! The Llama-family decoder (Llama, Qwen3), built only from ops hanzo-ml can
//! differentiate: `rms_norm_slow`, `rope_slow`, `softmax`, `silu`, matmul.

use crate::model::lora::{linear, Linear, LoraLinear, Proj};
use crate::Cache;
use hanzo_ml::{DType, Device, Result, Tensor, Var, D};
use hanzo_nn::ops::{rms_norm_slow, silu, softmax};
use hanzo_nn::rotary_emb::rope_slow;
use hanzo_nn::VarBuilder;
use serde::Deserialize;

/// Hyperparameters of one checkpoint.
#[derive(Debug, Clone)]
pub struct Arch {
    pub vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub theta: f64,
    pub tied: bool,
    pub attn_bias: bool,
    /// Per-head RMSNorm on q and k (Qwen3).
    pub qk_norm: bool,
}

/// The `config.json` keys shared by the family.
#[derive(Deserialize)]
struct Json {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    rms_norm_eps: f64,
    #[serde(default = "ten_thousand")]
    rope_theta: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
}

fn ten_thousand() -> f64 {
    10_000.0
}

impl Arch {
    pub fn parse(v: &serde_json::Value, qk_norm: bool) -> anyhow::Result<Self> {
        let j: Json = serde_json::from_value(v.clone())?;
        Ok(Self {
            vocab: j.vocab_size,
            hidden: j.hidden_size,
            inter: j.intermediate_size,
            layers: j.num_hidden_layers,
            heads: j.num_attention_heads,
            kv_heads: j.num_key_value_heads.unwrap_or(j.num_attention_heads),
            head_dim: j.head_dim.unwrap_or(j.hidden_size / j.num_attention_heads),
            eps: j.rms_norm_eps,
            theta: j.rope_theta,
            tied: j.tie_word_embeddings,
            attn_bias: j.attention_bias,
            qk_norm,
        })
    }
}

/// LoRA settings applied while building.
#[derive(Debug, Clone)]
pub struct Lora {
    pub r: usize,
    pub alpha: f64,
    pub dropout: f64,
    /// Projection names (`q_proj`, `down_proj`, ...).
    pub targets: Vec<String>,
}

/// Projection paths under `model.layers.{i}`, in HF order.
const PROJS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// What one pass over a sequence holds fixed for every layer.
struct Pass<'a> {
    arch: &'a Arch,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor,
    /// Applies LoRA dropout.
    train: bool,
}

struct Layer {
    ln1: Tensor,
    ln2: Tensor,
    projs: [Proj; 7],
    qk_norm: Option<(Tensor, Tensor)>,
}

pub struct Transformer {
    pub arch: Arch,
    pub lora: Lora,
    /// Applies LoRA dropout in `forward`.
    pub train: bool,
    pub eos: Option<u32>,
    /// Checkpoint id, written to `adapter_config.json`.
    pub base: String,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    head: Tensor,
    inv_freq: Tensor,
    device: Device,
    dtype: DType,
}

fn proj(
    vb: &VarBuilder,
    name: &str,
    out: usize,
    inp: usize,
    bias: bool,
    lora: &Lora,
) -> Result<Proj> {
    let weight = vb.get((out, inp), &format!("{name}.weight"))?;
    let bias = bias
        .then(|| vb.get(out, &format!("{name}.bias")))
        .transpose()?;
    let short = name.rsplit('.').next().unwrap_or(name);
    if lora.targets.iter().any(|t| t == short) {
        let scale = lora.alpha / lora.r as f64;
        Ok(Proj::Lora(LoraLinear::new(
            weight,
            bias,
            lora.r,
            scale,
            lora.dropout,
        )?))
    } else {
        Ok(Proj::Base(Linear { weight, bias }))
    }
}

impl Layer {
    fn new(a: &Arch, vb: &VarBuilder, lora: &Lora) -> Result<Self> {
        let (h, kv, d) = (a.heads * a.head_dim, a.kv_heads * a.head_dim, a.hidden);
        let shapes = [
            (h, d),
            (kv, d),
            (kv, d),
            (d, h),
            (a.inter, d),
            (a.inter, d),
            (d, a.inter),
        ];
        let mut projs = Vec::with_capacity(7);
        for (i, name) in PROJS.iter().enumerate() {
            let bias = a.attn_bias && i < 4;
            projs.push(proj(vb, name, shapes[i].0, shapes[i].1, bias, lora)?);
        }
        let qk_norm = if a.qk_norm {
            Some((
                vb.get(a.head_dim, "self_attn.q_norm.weight")?,
                vb.get(a.head_dim, "self_attn.k_norm.weight")?,
            ))
        } else {
            None
        };
        Ok(Self {
            ln1: vb.get(a.hidden, "input_layernorm.weight")?,
            ln2: vb.get(a.hidden, "post_attention_layernorm.weight")?,
            projs: projs
                .try_into()
                .map_err(|_| hanzo_ml::Error::Msg("seven projections".into()))?,
            qk_norm,
        })
    }

    /// `kv` is this layer's cache slot: when given, attention runs over the
    /// cached keys and values followed by the new ones, and the concatenation
    /// is stored back.
    fn forward(
        &self,
        x: &Tensor,
        p: &Pass,
        kv: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (eps, train) = (p.arch.eps as f32, p.train);
        let h = rms_norm_slow(x, &self.ln1, eps)?;
        let x = (x + self.attn(&h, p, kv)?)?;
        let h = rms_norm_slow(&x, &self.ln2, eps)?;
        let [_, _, _, _, gate, up, down] = &self.projs;
        let h = (silu(&gate.forward(&h, train)?)? * up.forward(&h, train)?)?;
        x + down.forward(&h, train)?
    }

    fn attn(
        &self,
        x: &Tensor,
        p: &Pass,
        kv: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (a, cos, sin, mask, train) = (p.arch, &p.cos, &p.sin, &p.mask, p.train);
        let (b, t, _) = x.dims3()?;
        let (h, kv_heads, d) = (a.heads, a.kv_heads, a.head_dim);
        let [q, k, v, o, ..] = &self.projs;
        let split = |x: Tensor, n: usize| x.reshape((b, t, n, d))?.transpose(1, 2);
        let mut q = split(q.forward(x, train)?, h)?;
        let mut k = split(k.forward(x, train)?, kv_heads)?;
        let mut v = split(v.forward(x, train)?, kv_heads)?;
        if let Some((qn, kn)) = &self.qk_norm {
            q = rms_norm_slow(&q, qn, a.eps as f32)?;
            k = rms_norm_slow(&k, kn, a.eps as f32)?;
        }
        let q = rope_slow(&q, cos, sin)?.to_dtype(DType::F32)?;
        k = rope_slow(&k, cos, sin)?;
        if let Some(slot) = kv {
            if let Some((pk, pv)) = slot {
                k = Tensor::cat(&[&*pk, &k], 2)?;
                v = Tensor::cat(&[&*pv, &v], 2)?;
            }
            *slot = Some((k.clone(), v.clone()));
        }
        let k = repeat(&k, h / kv_heads)?.to_dtype(DType::F32)?;
        let v = repeat(&v, h / kv_heads)?.to_dtype(DType::F32)?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (d as f64).sqrt())?;
        let p = softmax(&scores.broadcast_add(mask)?, D::Minus1)?;
        let ctx = p.matmul(&v.contiguous()?)?.to_dtype(x.dtype())?;
        o.forward(&ctx.transpose(1, 2)?.reshape((b, t, h * d))?, train)
    }
}

/// Repeats each kv head `n` times along the head axis: `[b, kv, t, d]` to `[b, kv·n, t, d]`.
fn repeat(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    Tensor::cat(&vec![x; n], 2)?.reshape((b, kv * n, t, d))
}

impl Transformer {
    pub fn new(arch: Arch, lora: Lora, vb: &VarBuilder) -> Result<Self> {
        let embed = vb.get((arch.vocab, arch.hidden), "model.embed_tokens.weight")?;
        let head = if arch.tied {
            embed.clone()
        } else {
            vb.get((arch.vocab, arch.hidden), "lm_head.weight")?
        };
        let layers = (0..arch.layers)
            .map(|i| Layer::new(&arch, &vb.pp(format!("model.layers.{i}")), &lora))
            .collect::<Result<Vec<_>>>()?;
        let d = arch.head_dim;
        let inv_freq: Vec<f32> = (0..d)
            .step_by(2)
            .map(|i| 1.0 / arch.theta.powf(i as f64 / d as f64) as f32)
            .collect();
        Ok(Self {
            norm: vb.get(arch.hidden, "model.norm.weight")?,
            inv_freq: Tensor::from_vec(inv_freq, (1, d / 2), vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            embed,
            layers,
            head,
            arch,
            lora,
            train: true,
            eos: None,
            base: String::new(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Logits `[b, t, vocab]` over the full sequence. `attention_mask` is
    /// `[b, t]` u8 with 1 at real tokens.
    pub fn forward(&self, ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let t = ids.dim(1)?;
        let mask = self.mask(t, attention_mask)?;
        self.run(ids, self.pass(0, t, mask, self.train)?, None)
    }

    /// Logits for the new positions `ids` (`[b, t_new]`) continuing `cache`,
    /// which then covers `cache.len + t_new` positions. Every row of the batch
    /// must be the same length: there is no padding mask on this path.
    pub fn forward_cached(&self, ids: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let t = ids.dim(1)?;
        cache.layers.resize_with(self.layers.len(), || None);
        let mask = self.causal(t, cache.len)?;
        let logits = self.run(ids, self.pass(cache.len, t, mask, false)?, Some(cache))?;
        cache.len += t;
        Ok(logits)
    }

    /// The decoder over `ids` (`[b, t]`) through `p`, one cache slot per layer.
    fn run(&self, ids: &Tensor, p: Pass, mut cache: Option<&mut Cache>) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let a = &self.arch;
        let mut x = self
            .embed
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, t, a.hidden))?;
        for (i, l) in self.layers.iter().enumerate() {
            let kv = cache.as_deref_mut().map(|c| &mut c.layers[i]);
            x = l.forward(&x, &p, kv)?;
        }
        linear(
            &rms_norm_slow(&x, &self.norm, a.eps as f32)?,
            &self.head,
            None,
        )
    }

    /// Rotary tables for positions `past .. past + t` with `mask`.
    fn pass(&self, past: usize, t: usize, mask: Tensor, train: bool) -> Result<Pass<'_>> {
        let pos = Tensor::arange(past as u32, (past + t) as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((t, 1))?;
        let freqs = pos.matmul(&self.inv_freq)?;
        Ok(Pass {
            arch: &self.arch,
            cos: freqs.cos()?.to_dtype(self.dtype)?,
            sin: freqs.sin()?.to_dtype(self.dtype)?,
            mask,
            train,
        })
    }

    /// Additive f32 mask `[1, 1, t, past + t]`: query `i` sees every one of the
    /// `past` earlier keys and the new keys up to itself.
    fn causal(&self, t: usize, past: usize) -> Result<Tensor> {
        let n = past + t;
        let m: Vec<f32> = (0..t)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if j <= past + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_vec(m, (1, 1, t, n), &self.device)
    }

    /// Additive f32 mask: causal `[1, 1, t, t]`, or `[b, 1, t, t]` with `-inf`
    /// on padded keys.
    fn mask(&self, t: usize, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let causal = self.causal(t, 0)?;
        let Some(m) = attention_mask else {
            return Ok(causal);
        };
        let m = m.unsqueeze(1)?.unsqueeze(1)?;
        let keep = Tensor::zeros(m.dims(), DType::F32, &self.device)?;
        let drop = Tensor::full(f32::NEG_INFINITY, m.dims(), &self.device)?;
        causal.broadcast_add(&m.where_cond(&keep, &drop)?)
    }

    /// LoRA layers with their PEFT names (`base_model.model.model.layers.{i}.{path}`).
    pub fn adapters(&self) -> impl Iterator<Item = (String, &LoraLinear)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| {
            PROJS.iter().zip(&l.projs).filter_map(move |(p, proj)| {
                proj.lora()
                    .map(|l| (format!("base_model.model.model.layers.{i}.{p}"), l))
            })
        })
    }

    pub fn trainable_vars(&self) -> Vec<Var> {
        self.adapters()
            .flat_map(|(_, l)| [l.a.clone(), l.b.clone()])
            .collect()
    }
}
