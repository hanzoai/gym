# Hanzo Gym

LoRA fine-tuning and GRPO for language models, in Rust on
[hanzo-ml](https://github.com/hanzoai/ml). One binary for CUDA, Metal and CPU.
Configs use Axolotl's keys, so a config from `zooai/gym` runs here unchanged
where the feature exists and fails at load where it does not.

```bash
cargo install hanzo-gym --features metal   # or --features cuda
gym train qwen3-lora.yml
gym merge qwen3-lora.yml
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

Apache-2.0.
