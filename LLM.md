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
  `LmPolicy` over `CausalLm`: rollouts through `forward_cached` (prompt
  prefilled once per group, one decode step per token), differentiable
  scoring through `forward`; verifiers (`exact_match`, `numeric_match`,
  `format:<re>`, `length:<n>`, `http:<url>`).
- `src/synth/` — `gym synth`: Data Designer-style dataset synthesis. Columns
  are samplers, Jinja expressions or LLM calls (text, structured, judge) run
  per row in dependency order against any OpenAI-compatible endpoint;
  validators drop or regenerate; rows stream to jsonl/parquet.
- `src/trainjob.rs` — `gym trainjob`: the Kubeflow TrainJob container entry.
  Reads the env contract from hanzoai/ai `cluster/finetune.go` (BASE_MODEL,
  METHOD, TASK, DATASET_DIR, …) into a `Config`; METHOD=lora only.
  `Dockerfile.cuda` + `.hanzo/workflows/image.yml` build
  `ghcr.io/hanzoai/gym:<tag>` on the forge runners.

## Facts that shaped it

- hanzo-nn's fused `rms_norm`, `rope` and attention kernels have no backward;
  the model uses `rms_norm_slow`, `rope_slow`, `softmax_last_dim` and matmul.
- `Cache`/`forward_cached` is the decode path; rows in a cached batch must
  have equal length (no padding mask there).
- Metal has no f64: reduce norms and scalars through f32 tensors, never
  `to_dtype(F64)`. No QLoRA, full fine-tune, multi-GPU, DPO family, sample
  packing — rejected at config load.
- KL in GRPO is against the pre-update policy of the step (adapter cannot be
  disabled through `CausalLm`), not the frozen base.
- Hosted `/v1/ai/finetune/jobs` (hanzoai/ai) submits a TrainJob against the
  `hanzo-ft-{lora,qlora,full}` ClusterTrainingRuntimes in
  `~/work/hanzo-ml/trainer/manifests/base/runtimes/hanzo_finetune.yaml`
  (image `ghcr.io/hanzoai/finetune-runtime`, Python). Pointing `hanzo-ft-lora`
  at `ghcr.io/hanzoai/gym:<tag>` with `CMD trainjob` is the switch, once the
  forge has built the image (the repo needs a git.hanzo.ai mirror first).
- Engine loads PEFT adapters at startup only (merged into weights);
  `/v1/models/unload` + `/reload` re-reads the same adapter path; it serves
  `logprobs`. Engine-served rollouts would be adapter-to-path + reload per step.

## Build / test

`cargo test` (CPU), `--features metal` or `--features cuda` for GPU. CI in
`.hanzo/workflows/ci.yml` (fmt, clippy -D warnings, tests; metal leg on
`hanzo-build-macos-arm64`). Publish: `cargo publish` as `hanzo-gym`.
