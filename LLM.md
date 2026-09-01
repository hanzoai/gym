# hanzo-gym

Rust trainer on hanzo-ml (candle fork, crates.io `hanzo-ml` / `hanzo-nn`).
Binary `gym`, lib `gym`. Standalone Cargo workspace (`[workspace]` in Cargo.toml
keeps it out of ~/work/hanzo's root workspace).

## Shape

- `src/lib.rs` — the contracts every module is written against: `CausalLm`
  (forward → logits, `trainable_vars`, `save_adapter`), `Example`, `Batch`
  (labels unshifted, `IGNORE_INDEX` = -100 at padding), `Prompt` (GRPO).
- `src/config.rs` — Axolotl-shaped YAML, `deny_unknown_fields`: a key this
  crate does not implement fails at load rather than being ignored.
- `src/hub.rs` — model files: local dir or Hugging Face repo via hf-hub.
- `src/model/` — LoRA (`lora.rs`), one trainable transformer for Qwen3 and
  Llama built from differentiable primitives only, PEFT adapter save/load/merge.
- `src/data/` — dataset sources (jsonl, json, parquet, HF datasets; HF split
  grammar), strategies `chat_template` / `alpaca` / `completion` with Axolotl's
  label semantics, `collate`, `load_prompts` for GRPO.
- `src/train/` — SFT loop: chunked masked CE, schedules, accumulation,
  clipping, AdamW, checkpoints; `run` picks device/dtype and dispatches to GRPO.
- `src/grpo/` — port of hanzo-ml's `hanzo-training/src/grpo` math with a real
  `LmPolicy` over `CausalLm`, verifiers (`exact_match`, `numeric_match`,
  `format:<re>`, `length:<n>`, `http:<url>`).

## Facts that shaped it

- hanzo-nn's fused `rms_norm`, `rope` and attention kernels have no backward;
  the model uses `rms_norm_slow`, `rope_slow`, `softmax_last_dim` and matmul.
- No KV cache in GRPO sampling yet (forward re-run per token). No QLoRA, full
  fine-tune, multi-GPU, DPO family, sample packing — rejected at config load.
- KL in GRPO is against the pre-update policy of the step (adapter cannot be
  disabled through `CausalLm`), not the frozen base.
- Hosted `/v1/ai/finetune/jobs` (hanzoai/ai) runs a Python transformers+PEFT
  image today; this binary is meant to take over `hanzo-ft-lora` once it
  covers the presets.

## Build / test

`cargo test` (CPU), `--features metal` or `--features cuda` for GPU. CI in
`.hanzo/workflows/ci.yml` (fmt, clippy -D warnings, tests; metal leg on
`hanzo-build-macos-arm64`). Publish: `cargo publish` as `hanzo-gym`.
