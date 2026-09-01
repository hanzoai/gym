use super::*;
use hanzo_ml::{DType, Device, Tensor};
use hanzo_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder};
use std::collections::HashMap;

const V: usize = 64;
const H: usize = 32;
const I: usize = 64;
const L: usize = 2;
const NH: usize = 4;
const KV: usize = 2;
const HD: usize = 8;
const T: usize = 6;
const ALL: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn arch(qk_norm: bool) -> Arch {
    Arch {
        vocab: V,
        hidden: H,
        inter: I,
        layers: L,
        heads: NH,
        kv_heads: KV,
        head_dim: HD,
        eps: 1e-6,
        theta: 10_000.0,
        tied: false,
        attn_bias: false,
        qk_norm,
    }
}

/// Random base weights under Hugging Face names.
fn weights(dev: &Device, qk_norm: bool) -> HashMap<String, Tensor> {
    let r = |s: &[usize]| Tensor::randn(0f32, 0.1, s, dev).unwrap();
    let g = |n: usize| Tensor::rand(0.5f32, 1.5, n, dev).unwrap();
    let mut m = HashMap::from([
        ("model.embed_tokens.weight".to_string(), r(&[V, H])),
        ("lm_head.weight".to_string(), r(&[V, H])),
        ("model.norm.weight".to_string(), g(H)),
    ]);
    let shapes = [
        ("self_attn.q_proj", [NH * HD, H]),
        ("self_attn.k_proj", [KV * HD, H]),
        ("self_attn.v_proj", [KV * HD, H]),
        ("self_attn.o_proj", [H, NH * HD]),
        ("mlp.gate_proj", [I, H]),
        ("mlp.up_proj", [I, H]),
        ("mlp.down_proj", [H, I]),
    ];
    for i in 0..L {
        let p = format!("model.layers.{i}");
        for (n, s) in shapes {
            m.insert(format!("{p}.{n}.weight"), r(&s));
        }
        m.insert(format!("{p}.input_layernorm.weight"), g(H));
        m.insert(format!("{p}.post_attention_layernorm.weight"), g(H));
        if qk_norm {
            m.insert(format!("{p}.self_attn.q_norm.weight"), g(HD));
            m.insert(format!("{p}.self_attn.k_norm.weight"), g(HD));
        }
    }
    m
}

fn model(w: &HashMap<String, Tensor>, qk_norm: bool, targets: &[&str]) -> Transformer {
    let lora = Lora {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        targets: targets.iter().map(|s| s.to_string()).collect(),
    };
    Transformer::new(
        arch(qk_norm),
        lora,
        &VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu),
    )
    .unwrap()
}

fn ids(dev: &Device) -> Tensor {
    let v: Vec<u32> = (0..2 * T).map(|i| ((i * 7 + 3) % V) as u32).collect();
    Tensor::from_vec(v, (2, T), dev).unwrap()
}

/// Random `B` so the LoRA path contributes.
fn perturb(m: &Transformer) {
    for (_, l) in m.adapters() {
        l.b.set(&Tensor::randn(0f32, 0.1, l.b.dims(), l.b.device()).unwrap())
            .unwrap();
    }
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap()
}

/// Next-token cross-entropy over every position.
fn loss(m: &Transformer, ids: &Tensor) -> Tensor {
    let logits = m.forward(ids, None).unwrap();
    let t = ids.dim(1).unwrap();
    let inp = logits.narrow(1, 0, t - 1).unwrap().flatten(0, 1).unwrap();
    let target = ids.narrow(1, 1, t - 1).unwrap().flatten_all().unwrap();
    cross_entropy(&inp, &target).unwrap()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("gym-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn qwen3_matches_reference() {
    use hanzo_transformers::models::qwen3::{Config, Model};
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let ids = ids(&dev);
    let ours = model(&w, true, &ALL).forward(&ids, None).unwrap();
    let cfg = Config {
        vocab_size: V,
        hidden_size: H,
        intermediate_size: I,
        num_hidden_layers: L,
        num_attention_heads: NH,
        head_dim: HD,
        attention_bias: false,
        num_key_value_heads: KV,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: hanzo_nn::Activation::Silu,
    };
    let mut theirs =
        Model::new(&cfg, VarBuilder::from_tensors(w.clone(), DType::F32, &dev)).unwrap();
    let hidden = theirs.forward(&ids, 0).unwrap();
    let logits = hidden
        .broadcast_matmul(&w["lm_head.weight"].t().unwrap())
        .unwrap();
    assert_eq!(ours.dims(), &[2, T, V]);
    assert!(
        max_diff(&ours, &logits) < 1e-4,
        "qwen3 diff {}",
        max_diff(&ours, &logits)
    );
}

#[test]
fn llama_matches_reference() {
    use hanzo_transformers::models::llama::{Cache, Config, Llama};
    let dev = Device::Cpu;
    let w = weights(&dev, false);
    let ids = ids(&dev);
    let ours = model(&w, false, &ALL).forward(&ids, None).unwrap();
    let cfg = Config {
        hidden_size: H,
        intermediate_size: I,
        vocab_size: V,
        num_hidden_layers: L,
        num_attention_heads: NH,
        num_key_value_heads: KV,
        use_flash_attn: false,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: 64,
        tie_word_embeddings: false,
    };
    let theirs = Llama::load(VarBuilder::from_tensors(w.clone(), DType::F32, &dev), &cfg).unwrap();
    let mut cache = Cache::new(false, DType::F32, &cfg, &dev).unwrap();
    // Their forward returns the last position only; every prefix covers every position.
    for n in 1..=T {
        let last = theirs
            .forward(&ids.narrow(1, 0, n).unwrap(), 0, &mut cache)
            .unwrap();
        let ours_n = ours.narrow(1, n - 1, 1).unwrap().squeeze(1).unwrap();
        assert!(
            max_diff(&ours_n, &last) < 1e-4,
            "llama position {} diff {}",
            n - 1,
            max_diff(&ours_n, &last)
        );
    }
}

#[test]
fn gradients_reach_only_lora() {
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let m = model(&w, true, &ALL);
    let grads = loss(&m, &ids(&dev)).backward().unwrap();
    let vars = m.trainable_vars();
    assert_eq!(vars.len(), 2 * ALL.len() * L);
    for v in &vars {
        assert!(grads.get(v).is_some());
    }
    assert!(grads.get(&w["lm_head.weight"]).is_none());
    assert!(grads
        .get(&w["model.layers.0.self_attn.q_proj.weight"])
        .is_none());
    assert_eq!(
        model(&w, true, &["q_proj", "v_proj"])
            .trainable_vars()
            .len(),
        2 * 2 * L
    );
}

#[test]
fn adamw_reduces_loss() {
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let m = model(&w, true, &ALL);
    let ids = ids(&dev);
    let mut opt = AdamW::new(
        m.trainable_vars(),
        ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        },
    )
    .unwrap();
    let before: f32 = loss(&m, &ids).to_scalar().unwrap();
    for _ in 0..30 {
        opt.backward_step(&loss(&m, &ids)).unwrap();
    }
    let after: f32 = loss(&m, &ids).to_scalar().unwrap();
    println!("loss {before} -> {after}");
    assert!(after < 0.7 * before, "loss {before} -> {after}");
}

#[test]
fn padding_leaves_real_positions_unchanged() {
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let m = model(&w, true, &ALL);
    perturb(&m);
    let ids = ids(&dev);
    let mask = Tensor::from_vec(vec![1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], (2, T), &dev).unwrap();
    let padded = m.forward(&ids, Some(&mask)).unwrap();
    for (row, len) in [(0, T), (1, T - 2)] {
        let alone = m
            .forward(
                &ids.narrow(0, row, 1).unwrap().narrow(1, 0, len).unwrap(),
                None,
            )
            .unwrap();
        let real = padded.narrow(0, row, 1).unwrap().narrow(1, 0, len).unwrap();
        assert!(
            max_diff(&alone, &real) < 1e-4,
            "row {row} diff {}",
            max_diff(&alone, &real)
        );
    }
}

#[test]
fn adapter_saves_loads_and_merges() {
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let m = model(&w, true, &ALL);
    perturb(&m);
    let ids = ids(&dev);
    let want = m.forward(&ids, None).unwrap();

    let dir = tmp("adapter");
    m.save_adapter(&dir).unwrap();
    let saved = hanzo_ml::safetensors::load(dir.join("adapter_model.safetensors"), &dev).unwrap();
    assert_eq!(saved.len(), 2 * ALL.len() * L);
    assert_eq!(
        saved["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].dims(),
        &[4, H]
    );
    assert_eq!(
        saved["base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"].dims(),
        &[H, 4]
    );
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!(cfg["peft_type"], "LORA");
    assert_eq!(cfg["r"], 4);
    assert_eq!(cfg["target_modules"].as_array().unwrap().len(), ALL.len());

    let fresh = model(&w, true, &ALL);
    load_adapter(&fresh, &dir).unwrap();
    assert_eq!(max_diff(&fresh.forward(&ids, None).unwrap(), &want), 0.0);

    let mut merged = w.clone();
    for (name, l) in m.adapters() {
        merged.insert(
            format!("{}.weight", name.trim_start_matches("base_model.model.")),
            l.merged_weight().unwrap(),
        );
    }
    let plain = model(&merged, true, &[]);
    assert!(plain.trainable_vars().is_empty());
    let got = plain.forward(&ids, None).unwrap();
    assert!(
        max_diff(&got, &want) < 1e-4,
        "merged diff {}",
        max_diff(&got, &want)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn build_and_merge_from_local_checkpoint() {
    let dev = Device::Cpu;
    let w = weights(&dev, true);
    let dir = tmp("checkpoint");
    hanzo_ml::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    let config = serde_json::json!({
        "model_type": "qwen3", "vocab_size": V, "hidden_size": H, "intermediate_size": I,
        "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": KV, "head_dim": HD,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "attention_bias": false, "eos_token_id": [151645, 151643],
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let yaml = |extra: &str| -> Config {
        serde_yaml::from_str(&format!(
            "base_model: {d}\noutput_dir: {d}/out\nlora_r: 4\nlora_alpha: 8\nlora_target_linear: true\n{extra}",
            d = dir.display()
        ))
        .unwrap()
    };
    let cfg = yaml("adapter: lora");
    assert!(build(
        &yaml("adapter: lora\ngradient_checkpointing: true"),
        &dev,
        DType::F32
    )
    .is_err());
    assert!(build(&yaml("adapter: none"), &dev, DType::F32).is_err());

    let m = build(&cfg, &dev, DType::F32).unwrap();
    assert_eq!(m.eos, Some(151645));
    assert_eq!(m.base, dir.display().to_string());
    perturb(&m);
    let ids = ids(&dev);
    let want = m.forward(&ids, None).unwrap();
    m.save_adapter(Path::new(&cfg.output_dir)).unwrap();

    merge(&cfg, None).unwrap();
    let out = dir.join("out/merged");
    assert!(out.join("config.json").exists() && out.join("tokenizer.json").exists());
    let merged = hanzo_ml::safetensors::load(out.join("model.safetensors"), &dev).unwrap();
    assert_eq!(merged.len(), w.len());
    let got = model(&merged, true, &[]).forward(&ids, None).unwrap();
    assert!(
        max_diff(&got, &want) < 1e-4,
        "merged diff {}",
        max_diff(&got, &want)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
