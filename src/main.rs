use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "gym",
    version,
    about = "Hanzo Gym — LoRA fine-tuning and GRPO on hanzo-ml"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train from a config (SFT, or GRPO when `rl: grpo`).
    Train { config: String },
    /// Tokenize the datasets and report token and example counts without training.
    Preprocess { config: String },
    /// Generate a synthetic dataset from a design (columns, samplers, LLM prompts, validators).
    Synth {
        design: String,
        /// Generate this many rows and print them instead of writing the dataset.
        #[arg(long)]
        preview: Option<usize>,
    },
    /// Train from the Kubeflow TrainJob environment (BASE_MODEL, METHOD, DATASET_DIR, …).
    Trainjob,
    /// Merge a trained LoRA adapter into the base weights.
    Merge {
        config: String,
        /// Directory to write merged safetensors into (default: <output_dir>/merged).
        #[arg(long)]
        out: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("gym=info".parse()?),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Train { config } => gym::train::run(&gym::Config::load(config)?),
        Cmd::Preprocess { config } => gym::data::preprocess(&gym::Config::load(config)?),
        Cmd::Synth { design, preview } => gym::synth::run(&design, preview),
        Cmd::Trainjob => gym::trainjob::run(),
        Cmd::Merge { config, out } => {
            gym::model::merge(&gym::Config::load(config)?, out.as_deref())
        }
    }
}
