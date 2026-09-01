//! LoRA on a frozen linear layer, as PEFT defines it.

use hanzo_ml::{DType, Result, Tensor, Var};
use hanzo_nn::Dropout;

/// `x·Wᵀ (+ b)` for `x` of shape `[b, t, in]` and `w` of shape `[out, in]`.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let (bs, t, i) = x.dims3()?;
    let y = x
        .reshape((bs * t, i))?
        .matmul(&w.t()?)?
        .reshape((bs, t, ()))?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

/// Frozen projection.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// Frozen projection plus a trainable low-rank update `scale·B·A`.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub a: Var,
    pub b: Var,
    pub scale: f64,
    pub dropout: Option<Dropout>,
}

impl LoraLinear {
    /// `A` is uniform in `±1/√in` (PEFT's kaiming init), `B` is zero, so the
    /// layer starts identical to the frozen one.
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        r: usize,
        scale: f64,
        dropout: f64,
    ) -> Result<Self> {
        let (out, inp) = weight.dims2()?;
        let bound = 1.0 / (inp as f32).sqrt();
        let a = Tensor::rand(-bound, bound, (r, inp), weight.device())?.to_dtype(weight.dtype())?;
        Ok(Self {
            a: Var::from_tensor(&a)?,
            b: Var::zeros((out, r), weight.dtype(), weight.device())?,
            scale,
            dropout: (dropout > 0.0).then(|| Dropout::new(dropout as f32)),
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let y = linear(x, &self.weight, self.bias.as_ref())?;
        let x = match &self.dropout {
            Some(d) => d.forward(x, train)?,
            None => x.clone(),
        };
        let d = linear(&linear(&x, &self.a, None)?, &self.b, None)?;
        y + (d * self.scale)?
    }

    /// `W + scale·B·A`, computed in f32 and returned in the weight's dtype.
    pub fn merged_weight(&self) -> Result<Tensor> {
        let d = (self
            .b
            .to_dtype(DType::F32)?
            .matmul(&self.a.to_dtype(DType::F32)?)?
            * self.scale)?;
        (self.weight.to_dtype(DType::F32)? + d)?.to_dtype(self.weight.dtype())
    }
}

/// A projection in the transformer: frozen, or frozen with a LoRA update.
#[derive(Debug, Clone)]
pub enum Proj {
    Base(Linear),
    Lora(LoraLinear),
}

impl Proj {
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        match self {
            Proj::Base(l) => linear(x, &l.weight, l.bias.as_ref()),
            Proj::Lora(l) => l.forward(x, train),
        }
    }

    pub fn lora(&self) -> Option<&LoraLinear> {
        match self {
            Proj::Lora(l) => Some(l),
            Proj::Base(_) => None,
        }
    }
}
