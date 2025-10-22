use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use bbpe::config::{IngestConfig, TrainerConfig};
use bbpe::corpus::load_binary_corpus;
use bbpe::serialization;
use bbpe::{BinaryTokenizer, Trainer};
use clap::{ArgAction, Args, Parser, Subcommand};
use env_logger::Env;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serde_json::{self, json};

const DEFAULT_OUTPUT: &str = "tokenizer.json";

#[derive(Parser, Debug)]
#[command(author, version, about = "Binary BPE toolkit", long_about = None)]
struct Cli {
    /// Increase verbosity (-v, -vv)
    #[arg(short = 'v', long, global = true, action = ArgAction::Count)]
    verbose: u8,

    /// Decrease verbosity (-q, -qq)
    #[arg(short = 'q', long, global = true, action = ArgAction::Count)]
    quiet: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a new tokenizer from binary inputs
    Train(TrainArgs),
    /// Encode binary files with a trained tokenizer
    Encode(EncodeArgs),
    /// Decode token ids back into bytes
    Decode(DecodeArgs),
    /// Inspect tokenizer metadata
    Info(InfoArgs),
}

#[derive(Args, Debug)]
struct TrainArgs {
    /// Files or directories to ingest
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output path for tokenizer.json
    #[arg(short, long, value_name = "PATH", default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Target vocabulary size
    #[arg(long, value_name = "SIZE")]
    vocab_size: Option<usize>,

    /// Minimum frequency for merges
    #[arg(long, value_name = "COUNT")]
    min_frequency: Option<usize>,

    /// Allowed token lengths in bytes (repeat flag)
    #[arg(long = "allowed-length", value_name = "LEN")]
    allowed_lengths: Vec<usize>,

    /// Override chunk size (0 = whole file)
    #[arg(long, value_name = "BYTES")]
    chunk_size: Option<usize>,

    /// Disable per-iteration logging/progress
    #[arg(long)]
    no_progress: bool,

    /// Append additional special tokens
    #[arg(long = "special-token", value_name = "TOKEN")]
    special_tokens: Vec<String>,

    /// Maximum merge iterations
    #[arg(long, value_name = "COUNT")]
    max_merge_iterations: Option<usize>,

    /// Plateau frequency floor
    #[arg(long, value_name = "FREQ")]
    plateau_floor: Option<usize>,

    /// Plateau patience
    #[arg(long, value_name = "COUNT")]
    plateau_patience: Option<usize>,

    /// Plateau frequency divisor
    #[arg(long, value_name = "DIV")]
    plateau_divisor: Option<usize>,

    /// Enable plateau-based early stopping
    #[arg(long)]
    plateau_stop: bool,

    /// Emit pretty JSON
    #[arg(long)]
    pretty: bool,

    /// Limit Rayon worker threads
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// Disable recursive directory traversal
    #[arg(long)]
    no_recursive: bool,

    /// Follow symlinks during traversal
    #[arg(long)]
    follow_symlinks: bool,
}

#[derive(Args, Debug)]
struct EncodeArgs {
    /// Tokenizer JSON to load
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Binary inputs to encode
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Emit JSON lines instead of human-readable output
    #[arg(long)]
    json: bool,

    /// Include special tokens during encoding
    #[arg(long)]
    add_special_tokens: bool,

    /// Optional directory to write .tokens files
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct DecodeArgs {
    /// Tokenizer JSON to load
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Path to whitespace separated token ids
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Token ids to decode when --input is omitted
    #[arg(value_name = "ID", required_unless_present = "input")]
    tokens: Vec<u32>,

    /// Skip special tokens while decoding
    #[arg(long)]
    skip_special_tokens: bool,

    /// Output file for decoded bytes (defaults to stdout)
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct InfoArgs {
    /// Tokenizer JSON to inspect
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Emit machine-readable JSON summary
    #[arg(long)]
    json: bool,
}

#[derive(Deserialize)]
struct TokenizerFile {
    model: ModelSection,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(rename = "type")]
    kind: String,
    vocab: serde_json::Map<String, serde_json::Value>,
    merges: Vec<serde_json::Value>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Deserialize)]
struct AddedToken {
    content: String,
    #[serde(default)]
    special: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);

    match cli.command {
        Commands::Train(args) => run_train(args),
        Commands::Encode(args) => run_encode(args),
        Commands::Decode(args) => run_decode(args),
        Commands::Info(args) => run_info(args),
    }
}

fn init_logging(verbose: u8, quiet: u8) {
    use log::LevelFilter;

    let level = if quiet > 0 {
        match quiet {
            0 => LevelFilter::Info,
            1 => LevelFilter::Warn,
            _ => LevelFilter::Error,
        }
    } else {
        match verbose {
            0 => LevelFilter::Info,
            1 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or("info"));
    builder.format_timestamp_millis();
    builder.filter_level(level);
    let _ = builder.try_init();
}

fn run_train(args: TrainArgs) -> Result<()> {
    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("unable to configure Rayon thread pool")?;
    }

    let defaults = TrainerConfig::default();
    let mut cfg = TrainerConfig::builder();
    if let Some(vocab_size) = args.vocab_size {
        cfg = cfg.target_vocab_size(vocab_size);
    }
    if let Some(min_frequency) = args.min_frequency {
        cfg = cfg.min_frequency(min_frequency);
    }
    if !args.allowed_lengths.is_empty() {
        cfg = cfg.allowed_token_lengths(args.allowed_lengths.clone());
    }
    if !args.special_tokens.is_empty() {
        cfg = cfg.special_tokens(args.special_tokens.clone());
    }
    if args.plateau_floor.is_some()
        || args.plateau_patience.is_some()
        || args.plateau_divisor.is_some()
    {
        cfg = cfg.plateau_thresholds(
            args.plateau_floor
                .unwrap_or(defaults.plateau_frequency_floor),
            args.plateau_patience.unwrap_or(defaults.plateau_patience),
            args.plateau_divisor
                .unwrap_or(defaults.plateau_frequency_divisor),
        );
    }
    cfg = cfg.max_merge_iterations(args.max_merge_iterations);
    cfg = cfg.plateau_stop_enabled(args.plateau_stop);
    cfg = cfg.show_progress(!args.no_progress);
    let trainer_cfg = cfg.build()?;

    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size.unwrap_or(defaults_chunk_size()),
        recursive: !args.no_recursive,
        follow_symlinks: args.follow_symlinks,
    };

    let sequences = load_binary_corpus(&args.inputs, &ingest_cfg)
        .with_context(|| "failed to load binary corpus")?;
    let corpus_bytes: usize = sequences.iter().map(|seq| seq.len()).sum();
    info!(
        "loaded {} sequences totalling {:.2} MiB",
        sequences.len(),
        bytes_to_mebibytes(corpus_bytes)
    );

    let spinner = if args.no_progress {
        None
    } else {
        let pb = ProgressBar::new_spinner();
        let style = ProgressStyle::with_template("{spinner} training merges... {elapsed}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        pb.set_style(style);
        pb.enable_steady_tick(Duration::from_millis(80));
        Some(pb)
    };

    let trainer = Trainer::new(trainer_cfg.clone());
    let start = Instant::now();
    let artifacts = trainer.train_from_sequences(&sequences)?;
    drop(sequences);
    if let Some(pb) = spinner {
        pb.finish_with_message("training complete");
    }

    let elapsed = start.elapsed();
    let merges = artifacts.model.merges().len();
    let vocab_size = artifacts.model.vocab_size();
    let throughput = if elapsed.as_secs_f64() > 0.0 {
        bytes_to_mebibytes(corpus_bytes) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    artifacts
        .model
        .save_huggingface(&args.output)
        .with_context(|| format!("failed to save tokenizer to {}", args.output.display()))?;
    if args.pretty {
        let pretty = artifacts.model.to_huggingface_json(true)?;
        fs::write(&args.output, pretty)
            .with_context(|| format!("failed to pretty print {}", args.output.display()))?;
    }

    info!(
        "training complete: merges={merges} vocab={vocab_size} duration={elapsed:.2?} throughput={throughput:.2} MiB/s"
    );
    println!(
        "✅ wrote tokenizer with vocab {} ({} merges) to {}",
        vocab_size,
        merges,
        args.output.display()
    );
    println!(
        "   corpus {:.2} MiB | duration {:.2?} | throughput {:.2} MiB/s",
        bytes_to_mebibytes(corpus_bytes),
        elapsed,
        throughput
    );

    Ok(())
}

fn run_encode(args: EncodeArgs) -> Result<()> {
    let tokenizer = serialization::load_tokenizer(&args.tokenizer)
        .with_context(|| format!("failed to load tokenizer from {}", args.tokenizer.display()))?;
    let tokenizer = BinaryTokenizer::from_tokenizer(tokenizer)?;

    if let Some(dir) = &args.output_dir {
        if !dir.exists() {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create output directory {}", dir.display()))?;
        }
    }

    for path in &args.inputs {
        let mut file =
            File::open(path).with_context(|| format!("failed to open input {}", path.display()))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let tokens = tokenizer.encode_bytes(&data, args.add_special_tokens)?;
        if let Some(dir) = &args.output_dir {
            let mut out_path = dir.clone();
            let filename = path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "stdin".to_string());
            out_path.push(format!("{filename}.tokens"));
            let mut output = File::create(&out_path)
                .with_context(|| format!("failed to open {}", out_path.display()))?;
            write_token_sequence(&mut output, &tokens)?;
            println!("{} => {}", path.display(), out_path.display());
        } else if args.json {
            let record = json!({
                "path": path.display().to_string(),
                "tokens": tokens
            });
            println!("{}", serde_json::to_string(&record)?);
        } else {
            print!("{}:\t", path.display());
            for (idx, token) in tokens.iter().enumerate() {
                if idx > 0 {
                    print!(" ");
                }
                print!("{token}");
            }
            println!();
        }
    }

    Ok(())
}

fn run_decode(args: DecodeArgs) -> Result<()> {
    let tokenizer = serialization::load_tokenizer(&args.tokenizer)
        .with_context(|| format!("failed to load tokenizer from {}", args.tokenizer.display()))?;
    let tokenizer = BinaryTokenizer::from_tokenizer(tokenizer)?;

    let tokens = if let Some(input_path) = &args.input {
        let contents = fs::read_to_string(input_path)
            .with_context(|| format!("failed to read {}", input_path.display()))?;
        parse_token_list(&contents)?
    } else {
        args.tokens
    };

    let bytes = tokenizer.decode_to_bytes(&tokens, args.skip_special_tokens)?;

    if let Some(path) = &args.output {
        let mut file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("wrote {} bytes to {}", bytes.len(), path.display());
    } else {
        io::stdout().write_all(&bytes)?;
    }

    Ok(())
}

fn run_info(args: InfoArgs) -> Result<()> {
    let data = fs::read_to_string(&args.tokenizer)
        .with_context(|| format!("failed to read {}", args.tokenizer.display()))?;
    let parsed: TokenizerFile =
        serde_json::from_str(&data).context("failed to parse tokenizer.json")?;

    let vocab_size = parsed.model.vocab.len();
    let merges = parsed.model.merges.len();
    let special_tokens = parsed
        .added_tokens
        .iter()
        .filter(|token| token.special)
        .map(|token| token.content.clone())
        .collect::<Vec<_>>();
    let summary = json!({
        "path": args.tokenizer.display().to_string(),
        "model_type": parsed.model.kind,
        "vocab_size": vocab_size,
        "merges": merges,
        "byte_fallback": parsed.model.byte_fallback,
        "special_tokens": special_tokens,
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        let model_type = summary["model_type"].as_str().unwrap_or("unknown");
        let byte_fallback = parsed.model.byte_fallback;
        println!("Model type   : {model_type}");
        println!("Vocab size   : {vocab_size}");
        println!("Merges       : {merges}");
        println!("Byte fallback: {byte_fallback}");
        if summary["special_tokens"].is_array()
            && !summary["special_tokens"].as_array().unwrap().is_empty()
        {
            println!(
                "Special tokens: {}",
                summary["special_tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else {
            println!("Special tokens: (none)");
        }
    }

    Ok(())
}

#[must_use]
fn defaults_chunk_size() -> usize {
    IngestConfig::default().chunk_size
}

#[must_use]
fn bytes_to_mebibytes(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn write_token_sequence<W: Write>(writer: &mut W, tokens: &[u32]) -> Result<()> {
    for (idx, token) in tokens.iter().enumerate() {
        if idx > 0 {
            writer.write_all(b" ")?;
        }
        write!(writer, "{token}")?;
    }
    writer.write_all(b"\n")?;
    Ok(())
}

fn parse_token_list(text: &str) -> Result<Vec<u32>> {
    text.split_whitespace()
        .map(|part| {
            part.parse::<u32>()
                .map_err(|err| anyhow!("invalid token id `{part}`: {err}"))
        })
        .collect()
}
