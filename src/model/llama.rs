//! Llama: `head_dim = hidden_size / num_attention_heads`, default rope only.

use super::transformer::Arch;

pub fn arch(v: &serde_json::Value) -> anyhow::Result<Arch> {
    anyhow::ensure!(
        v.get("rope_scaling").is_none_or(serde_json::Value::is_null),
        "rope_scaling is not supported; only default rope"
    );
    Arch::parse(v, false)
}
