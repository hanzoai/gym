# Hanzo Gym

LoRA fine-tuning and GRPO for language models, in Rust on
[hanzo-ml](https://github.com/hanzoai/ml). One binary for CUDA, Metal and CPU.
Configs use Axolotl's keys, so a config from `zooai/gym` runs here unchanged
where the feature exists and fails at load where it does not.

```bash
cargo install hanzo-gym --features metal   # or --features cuda
gym train examples/qwen3-lora.yml   # Qwen3-0.6B, LoRA on q/v, 20 steps
gym merge examples/qwen3-lora.yml
```

```yaml
base_model: Qwen/Qwen3-0.6B
chat_template: qwen3
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    field_messages: conversations
    message_property_mappings: { role: from, content: value }
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_linear: true
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 1
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03
output_dir: ./outputs/qwen3-lora
```

The adapter is written in PEFT layout, so it loads in Hanzo Engine, vLLM and
`peft` without conversion. `rl: grpo` switches the run to GRPO with the reward
functions named under `trl:`.

## Scope

Implemented: LoRA on Qwen3 and Llama, SFT on `chat_template`, `alpaca` and
`completion` datasets, cosine/linear/constant schedules with warmup, gradient
accumulation and clipping, gradient checkpointing, GRPO with built-in and HTTP
verifiers, PEFT adapter save and merge.

Not implemented: QLoRA, full fine-tuning, multi-GPU, DPO/KTO/ORPO, sample
packing. A config that asks for one of these is rejected at load.

## Hosted / TrainJob

`ghcr.io/hanzoai/gym` (built from `Dockerfile.cuda` by
`.hanzo/workflows/image.yml`, tagged by git tag) is the trainer container of
the Kubeflow TrainJob that `POST /v1/ai/finetune/jobs` submits. Its command
is `gym trainjob`, which reads the run from the environment the broker
(hanzoai/ai `cluster/finetune.go`) sets, with the broker's defaults when a
variable is unset:

| Variable | Config key | Default |
|---|---|---|
| `BASE_MODEL` | `base_model`, when `MODEL_DIR` is not a directory | — |
| `MODEL_DIR` | `base_model`, the initializer's copy of the model | `/workspace/model` |
| `DATASET_DIR` | one dataset per `.jsonl` / `.json` / `.parquet` file under it | `/workspace/dataset` |
| `OUTPUT_DIR` | `output_dir` | `/workspace/output` |
| `TASK` | `chat` → `chat_template`, `instruct` → `alpaca`, `completion` → `completion` | — |
| `METHOD` | `lora` → `adapter: lora`, `lora_target_linear: true` | `qlora` |
| `EPOCHS` | `num_epochs` (whole number) | 3 |
| `LEARNING_RATE` | `learning_rate` | 2e-4 |
| `BATCH_SIZE` | `micro_batch_size` | 2 |
| `GRAD_ACCUM` | `gradient_accumulation_steps` | 8 |
| `MAX_SEQ_LEN` | `sequence_len` | 2048 |
| `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` | `lora_r`, `lora_alpha`, `lora_dropout` | 16, 32, 0.05 |
| `WARMUP_RATIO`, `WEIGHT_DECAY` | `warmup_ratio`, `weight_decay` | 0.03, 0.01 |
| `QUANT_4BIT`, `GRADIENT_CHECKPOINTING` | rejected when `true` | false |
| `HF_TOKEN` | read by hf-hub for gated repos; never logged | — |

`METHOD=lora` is what this image serves today (the `hanzo-ft-lora` runtime).
`qlora` and `full`, `QUANT_4BIT=true` and `GRADIENT_CHECKPOINTING=true` fail
at startup with a message naming the runtime or limit, so the TrainJob goes
`Failed` instead of training something other than what was asked. Precision
is `bf16: auto`. The resolved config is logged as YAML before training.

Apache-2.0.
