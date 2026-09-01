//! Training entry point: pick device and dtype, load text, data and model,
//! then hand off to SFT or GRPO.

mod clip;
mod loss;
mod sched;
mod trainer;

pub use clip::{accumulate, clip_grad_norm};
pub use loss::causal_lm_loss;
pub use sched::lr_at;
pub use trainer::sft;

use crate::config::{Precision, Rl};
use crate::Config;
use hanzo_ml::{DType, Device};

pub fn run(cfg: &Config) -> anyhow::Result<()> {
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let dtype = match cfg.bf16 {
        Precision::Auto if device.is_cpu() => DType::F32,
        Precision::Auto | Precision::On => DType::BF16,
        Precision::Off => DType::F32,
    };
    std::fs::create_dir_all(&cfg.output_dir)?;
    let text = crate::data::Text::load(cfg)?;
    let data = crate::data::load(cfg, &text)?;
    let model = crate::model::load(cfg, &device, dtype)?;
    let params: usize = model.trainable_vars().iter().map(|v| v.elem_count()).sum();
    tracing::info!(
        train = data.train.len(),
        val = data.val.len(),
        params,
        device = ?device,
        dtype = ?dtype,
        "run"
    );
    if cfg.rl == Some(Rl::Grpo) {
        let prompts = crate::data::load_prompts(cfg, &text)?;
        crate::grpo::run(cfg, model.as_ref(), &text, prompts)
    } else {
        sft(cfg, model.as_ref(), &data.train, &data.val, text.pad_id)
    }
}
