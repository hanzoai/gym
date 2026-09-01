//! Qwen3: the Llama layout with per-head RMSNorm on q and k and an explicit `head_dim`.

use super::transformer::Arch;

pub fn arch(v: &serde_json::Value) -> anyhow::Result<Arch> {
    anyhow::ensure!(
        v.get("head_dim").is_some_and(|d| d.is_u64()),
        "qwen3 config.json needs head_dim"
    );
    Arch::parse(v, true)
}
