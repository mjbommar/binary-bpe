use assert_cmd::Command;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

const PYTHON_ROUND_TRIP_SCRIPT: &str = r#"
import sys
from pathlib import Path
from tokenizers import Tokenizer

tokenizer_path, sample_path = sys.argv[1], sys.argv[2]
tokenizer = Tokenizer.from_file(tokenizer_path)
data = Path(sample_path).read_bytes()
text = data.decode('latin-1')
encoding = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False).encode('latin-1')
if decoded != data:
    raise SystemExit('round trip mismatch with Hugging Face')
"#;

fn temp_workspace() -> TempDir {
    tempfile::tempdir().expect("create tempdir")
}

fn run_command(cmd: &mut Command) {
    cmd.assert().success();
}

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn python_round_trip(tokenizer: &Path, sample: &Path) {
    let project_root = project_root();
    let cache_dir = project_root.join("target").join("uv-cache-tests");
    fs::create_dir_all(&cache_dir).expect("create uv cache dir");

    let mut cmd = Command::new("uv");
    cmd.current_dir(project_root)
        .env("UV_CACHE_DIR", &cache_dir)
        .args([
            "run",
            "--with",
            "tokenizers",
            "python",
            "-c",
            PYTHON_ROUND_TRIP_SCRIPT,
        ])
        .arg(tokenizer.as_os_str())
        .arg(sample.as_os_str());
    run_command(&mut cmd);
}

#[test]
fn train_encode_decode_round_trip() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("input.bin");
    let output_path = workspace.path().join("tokenizer.json");
    let decoded_path = workspace.path().join("decoded.bin");

    let data: Vec<u8> = (0..=255).cycle().take(4096).collect();
    fs::write(&input_path, &data).expect("write input");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-frequency",
        "2",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--chunk-size",
        "1024",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(output_path.exists(), "tokenizer.json was created");

    let mut encode = Command::cargo_bin("bbpe").expect("binary exists");
    let encode_output = encode
        .current_dir(workspace.path())
        .args([
            "--quiet",
            "encode",
            "-m",
            output_path.file_name().unwrap().to_str().unwrap(),
            input_path.file_name().unwrap().to_str().unwrap(),
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let encoded: Value =
        serde_json::from_slice(&encode_output).expect("encoded output is valid JSON");
    let tokens = encoded["tokens"]
        .as_array()
        .expect("tokens array")
        .iter()
        .map(|v| v.as_u64().expect("u64 token"))
        .collect::<Vec<_>>();
    assert!(!tokens.is_empty(), "some tokens produced");

    let token_args = tokens.iter().map(|tok| tok.to_string()).collect::<Vec<_>>();

    let mut decode = Command::cargo_bin("bbpe").expect("binary exists");
    let mut args = vec![
        "--quiet".to_string(),
        "decode".to_string(),
        "-m".to_string(),
        output_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string(),
        "--output".to_string(),
        decoded_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string(),
    ];
    args.push("--skip-special-tokens".to_string());
    args.extend(token_args);
    decode.current_dir(workspace.path()).args(args);
    run_command(&mut decode);

    let decoded = fs::read(&decoded_path).expect("read decoded output");
    assert_eq!(decoded, data);

    let mut info = Command::cargo_bin("bbpe").expect("binary exists");
    let info_output = info
        .current_dir(workspace.path())
        .args([
            "--quiet",
            "info",
            "-m",
            output_path.file_name().unwrap().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let info_text = String::from_utf8(info_output).expect("info output is UTF-8");
    assert!(
        info_text.contains("Total vocab"),
        "info output contained expected summary"
    );
}

#[test]
fn train_from_jsonl_field() {
    let workspace = temp_workspace();
    let jsonl_path = workspace.path().join("records.jsonl");
    let jsonl_content = r#"{"text":"alpha beta"}
{"text":"gamma"}
{"text":"delta epsilon"}
"#;
    fs::write(&jsonl_path, jsonl_content).expect("write jsonl");
    let tokenizer_path = workspace.path().join("jsonl-tokenizer.json");

    let jsonl_spec = format!(
        "{}:text",
        jsonl_path
            .file_name()
            .unwrap()
            .to_str()
            .expect("jsonl filename utf-8")
    );

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        "--jsonl",
        &jsonl_spec,
        "--vocab-size",
        "270",
        "--min-frequency",
        "1",
        "--chunk-size",
        "0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "-o",
        tokenizer_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(
        tokenizer_path.exists(),
        "tokenizer produced from jsonl input"
    );

    let sample_path = workspace.path().join("sample.txt");
    fs::write(&sample_path, b"alpha beta delta").expect("write sample");
    let mut encode = Command::cargo_bin("bbpe").expect("binary exists");
    let encode_output = encode
        .current_dir(workspace.path())
        .args([
            "--quiet",
            "encode",
            "-m",
            tokenizer_path.file_name().unwrap().to_str().unwrap(),
            sample_path.file_name().unwrap().to_str().unwrap(),
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let encoded: Value = serde_json::from_slice(&encode_output).expect("encoded output parses");
    assert!(
        encoded["tokens"]
            .as_array()
            .map(|arr| !arr.is_empty())
            .unwrap_or(false),
        "jsonl-trained tokenizer encodes text"
    );
}

#[test]
fn train_from_gzipped_jsonl_field() {
    let workspace = temp_workspace();
    let gz_path = workspace.path().join("records.jsonl.gz");
    let jsonl_content = r#"{"text":"foo bar"}
{"text":"baz qux"}
"#;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(jsonl_content.as_bytes())
        .expect("write gz payload");
    let compressed = encoder.finish().expect("finish gz encoding");
    fs::write(&gz_path, compressed).expect("write gz file");

    let tokenizer_path = workspace.path().join("jsonl-gz-tokenizer.json");
    let spec = format!(
        "{}:text",
        gz_path
            .file_name()
            .unwrap()
            .to_str()
            .expect("gz filename utf-8"),
    );

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        "--jsonl",
        &spec,
        "--vocab-size",
        "270",
        "--min-frequency",
        "1",
        "--chunk-size",
        "0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "-o",
        tokenizer_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(
        tokenizer_path.exists(),
        "tokenizer produced from gzipped jsonl input"
    );
}

#[test]
fn chunk_train_emits_combined_tokenizer_and_report() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("input.bin");
    let output_path = workspace.path().join("combined.json");
    let report_path = workspace.path().join("report.json");

    let data: Vec<u8> = (0..=63).cycle().take(2048).collect();
    fs::write(&input_path, &data).expect("write input");

    let run_chunk = |mode: &str, out: &PathBuf, report: &PathBuf| {
        let mut cmd = Command::cargo_bin("bbpe").expect("binary exists");
        cmd.current_dir(workspace.path()).args([
            "--quiet",
            "chunk-train",
            input_path.file_name().unwrap().to_str().unwrap(),
            "--vocab-size",
            "270",
            "--min-frequency",
            "2",
            "--chunk-size",
            "512",
            "--combine-mode",
            mode,
            "--disable-reasoning-tokens",
            "--output",
            out.file_name().unwrap().to_str().unwrap(),
            "--report",
            report.file_name().unwrap().to_str().unwrap(),
        ]);
        run_command(&mut cmd);
        assert!(out.exists(), "combined tokenizer was created for {mode}");
        assert!(report.exists(), "chunk report was created for {mode}");
        let report_bytes = fs::read(report).expect("read report");
        let report: Value =
            serde_json::from_slice(&report_bytes).expect("chunk report parses as JSON");
        assert!(
            report["total_chunks"].as_u64().unwrap_or(0) >= 1,
            "report recorded at least one chunk for {mode}"
        );
        assert_eq!(
            report["combine_mode"].as_str(),
            Some(mode),
            "combine mode recorded correctly for {mode}"
        );
        assert!(
            report["combine_stats"]["merges_realized"]
                .as_u64()
                .unwrap_or(0)
                > 0,
            "combiner realised merges for {mode}"
        );
    };

    run_chunk("first", &output_path, &report_path);

    let freq_output = workspace.path().join("combined-frequency.json");
    let freq_report = workspace.path().join("report-frequency.json");
    run_chunk("frequency", &freq_output, &freq_report);

    let entropy_output = workspace.path().join("combined-entropy.json");
    let entropy_report = workspace.path().join("report-entropy.json");
    run_chunk("entropy", &entropy_output, &entropy_report);
}

#[test]
fn chunk_train_deduplicates_when_requested() {
    let workspace = temp_workspace();
    let file_a = workspace.path().join("dup-a.bin");
    let file_b = workspace.path().join("dup-b.bin");

    let data = vec![42u8; 512];
    fs::write(&file_a, &data).expect("write dup-a");
    fs::write(&file_b, &data).expect("write dup-b");

    let unique_output = workspace.path().join("unique.json");
    let unique_report = workspace.path().join("unique-report.json");

    let mut unique_cmd = Command::cargo_bin("bbpe").expect("binary exists");
    unique_cmd.current_dir(workspace.path()).args([
        "--quiet",
        "chunk-train",
        file_a.file_name().unwrap().to_str().unwrap(),
        file_b.file_name().unwrap().to_str().unwrap(),
        "--chunk-size",
        "512",
        "--duplicates",
        "unique",
        "--no-progress",
        "--output",
        unique_output.file_name().unwrap().to_str().unwrap(),
        "--report",
        unique_report.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut unique_cmd);

    let unique_report_bytes = fs::read(&unique_report).expect("read unique report");
    let unique: Value =
        serde_json::from_slice(&unique_report_bytes).expect("parse unique report json");
    assert_eq!(
        unique["processed_chunks"].as_u64(),
        Some(2),
        "unique mode recorded processed chunk count"
    );
    assert_eq!(
        unique["total_chunks"].as_u64(),
        Some(1),
        "unique mode reduced combined chunk count"
    );
    assert_eq!(
        unique["duplicate_chunks_skipped"].as_u64(),
        Some(1),
        "unique mode skipped duplicate chunk"
    );
    assert_eq!(
        unique["duplicate_chunks_reused"].as_u64(),
        Some(0),
        "unique mode did not reuse duplicates"
    );
    assert_eq!(
        unique["duplicate_mode"].as_str(),
        Some("unique"),
        "duplicate mode recorded correctly for unique run"
    );

    let count_output = workspace.path().join("count.json");
    let count_report = workspace.path().join("count-report.json");

    let mut count_cmd = Command::cargo_bin("bbpe").expect("binary exists");
    count_cmd.current_dir(workspace.path()).args([
        "--quiet",
        "chunk-train",
        file_a.file_name().unwrap().to_str().unwrap(),
        file_b.file_name().unwrap().to_str().unwrap(),
        "--chunk-size",
        "512",
        "--duplicates",
        "count",
        "--no-progress",
        "--output",
        count_output.file_name().unwrap().to_str().unwrap(),
        "--report",
        count_report.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut count_cmd);

    let count_bytes = fs::read(&count_report).expect("read count report");
    let count: Value = serde_json::from_slice(&count_bytes).expect("parse count report json");
    assert_eq!(
        count["processed_chunks"].as_u64(),
        Some(2),
        "count mode recorded processed chunk count"
    );
    assert_eq!(
        count["total_chunks"].as_u64(),
        Some(2),
        "count mode retained both chunks"
    );
    assert_eq!(
        count["duplicate_chunks_reused"].as_u64(),
        Some(1),
        "count mode reused duplicate chunk"
    );
    assert_eq!(
        count["duplicate_chunks_skipped"].as_u64(),
        Some(0),
        "count mode did not skip duplicates"
    );
    assert_eq!(
        count["duplicate_mode"].as_str(),
        Some("count"),
        "duplicate mode recorded correctly for count run"
    );
}

#[test]
fn python_round_trip_preprocessors() {
    let ascii_data = "Alpha  beta\tgamma\nDelta ".repeat(32).into_bytes();
    run_preprocessor_python_round_trip("ascii-whitespace", &ascii_data);

    let unicode_data = "uno\u{2003}dos\u{202F}tres\u{00A0}cuatro\u{1680}cinco "
        .repeat(16)
        .into_bytes();
    run_preprocessor_python_round_trip("unicode-whitespace", &unicode_data);

    let null_data = "\0alpha\0beta\0\0gamma\0delta\0epsilon\0zeta\0"
        .repeat(16)
        .into_bytes();
    run_preprocessor_python_round_trip("null-delimited", &null_data);
}

fn run_preprocessor_python_round_trip(mode: &str, data: &[u8]) {
    let workspace = temp_workspace();
    let input_path = workspace.path().join(format!("{mode}.bin"));
    fs::write(&input_path, data).expect("write preprocessor sample");
    let tokenizer_path = workspace.path().join(format!("{mode}.json"));

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-frequency",
        "2",
        "--chunk-size",
        "512",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--preprocessor",
        mode,
        "-o",
        tokenizer_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);

    python_round_trip(&tokenizer_path, &input_path);
}

#[test]
fn train_supports_preprocessor_and_family_outputs() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("text.bin");
    let base_output = workspace.path().join("tokenizer.json");

    let data = b"foo  bar\tbaz\nqux\x00wumpus   llama".to_vec();
    fs::write(&input_path, &data).expect("write corpus");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "300",
        "--min-frequency",
        "1",
        "--chunk-size",
        "0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--preprocessor",
        "ascii-whitespace",
        "--family-size",
        "264",
        "--family-size",
        "280",
        "--family-template",
        "variants/family-{size}.json",
        "-o",
        base_output.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);

    let base_json = fs::read_to_string(&base_output).expect("read tokenizer");
    let parsed: Value = serde_json::from_str(&base_json).expect("parse tokenizer");
    assert!(
        parsed.get("pre_tokenizer").is_some(),
        "pre_tokenizer section is emitted when preprocessing is enabled"
    );

    for size in [264usize, 280usize] {
        let derived_path = workspace
            .path()
            .join("variants")
            .join(format!("family-{size}.json"));
        assert!(
            derived_path.exists(),
            "derived tokenizer for vocab {size} exists"
        );
        let data = fs::read_to_string(&derived_path).expect("read derived tokenizer");
        let json: Value = serde_json::from_str(&data).expect("parse derived tokenizer");
        let base_vocab = json["model"]
            .get("vocab")
            .and_then(|v| v.as_object())
            .map(|map| map.len())
            .unwrap_or_default();
        assert_eq!(
            base_vocab, size,
            "family vocab {size} preserves requested size"
        );
    }
}

#[test]
fn probabilistic_preprocessor_disables_pre_tokenizer() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("prob.bin");
    fs::write(&input_path, b"foo bar baz qux").expect("write corpus");
    let tokenizer_path = workspace.path().join("prob.json");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "300",
        "--min-frequency",
        "1",
        "--chunk-size",
        "0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--preprocessor",
        "ascii-whitespace",
        "--preprocessor-probability",
        "0.5",
        "--preprocessor-seed",
        "1337",
        "-o",
        tokenizer_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);

    let json = fs::read_to_string(&tokenizer_path).expect("read tokenizer");
    let parsed: Value = serde_json::from_str(&json).expect("parse tokenizer json");
    assert_eq!(
        parsed.get("pre_tokenizer"),
        Some(&Value::Null),
        "probabilistic preprocessors should serialize a null pre_tokenizer"
    );
}

#[test]
fn train_with_max_entropy_filter() {
    let workspace = temp_workspace();
    let low_entropy = workspace.path().join("low.bin");
    let high_entropy = workspace.path().join("high.bin");
    let output_path = workspace.path().join("tokenizer.json");

    // Low entropy: repeated bytes
    let low_data = vec![0u8; 1024];
    fs::write(&low_entropy, &low_data).expect("write low entropy file");

    // High entropy: random-looking pattern
    let high_data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    fs::write(&high_entropy, &high_data).expect("write high entropy file");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        low_entropy.file_name().unwrap().to_str().unwrap(),
        high_entropy.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--max-entropy",
        "7.0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--chunk-size",
        "512",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(
        output_path.exists(),
        "tokenizer was created with max-entropy filter"
    );
}

#[test]
fn train_with_min_entropy_filter() {
    let workspace = temp_workspace();
    let low_entropy = workspace.path().join("low.bin");
    let normal_entropy = workspace.path().join("normal.bin");
    let output_path = workspace.path().join("tokenizer.json");

    // Very low entropy: all zeros
    let low_data = vec![0u8; 1024];
    fs::write(&low_entropy, &low_data).expect("write low entropy file");

    // Normal entropy: mixed bytes
    let normal_data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    fs::write(&normal_entropy, &normal_data).expect("write normal entropy file");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        low_entropy.file_name().unwrap().to_str().unwrap(),
        normal_entropy.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-entropy",
        "0.5",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--chunk-size",
        "512",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(
        output_path.exists(),
        "tokenizer was created with min-entropy filter"
    );
}

#[test]
fn train_with_both_entropy_filters() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("input.bin");
    let output_path = workspace.path().join("tokenizer.json");

    // Medium entropy data - use fewer unique bytes (0..=31) to get entropy around 5 bits/byte
    let data: Vec<u8> = (0..=31).cycle().take(2048).collect();
    fs::write(&input_path, &data).expect("write input");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-entropy",
        "0.2",
        "--max-entropy",
        "7.0",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--chunk-size",
        "512",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    run_command(&mut train);
    assert!(
        output_path.exists(),
        "tokenizer was created with both entropy filters"
    );
}

#[test]
fn train_entropy_filter_invalid_range() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("input.bin");
    let output_path = workspace.path().join("tokenizer.json");

    let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    fs::write(&input_path, &data).expect("write input");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-entropy",
        "7.0",
        "--max-entropy",
        "0.5",
        "--no-progress",
        "--disable-reasoning-tokens",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    train.assert().failure();
}

#[test]
fn train_entropy_filter_rejects_all() {
    let workspace = temp_workspace();
    let input_path = workspace.path().join("input.bin");
    let output_path = workspace.path().join("tokenizer.json");

    // Normal entropy data
    let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    fs::write(&input_path, &data).expect("write input");

    let mut train = Command::cargo_bin("bbpe").expect("binary exists");
    train.current_dir(workspace.path()).args([
        "--quiet",
        "train",
        input_path.file_name().unwrap().to_str().unwrap(),
        "--vocab-size",
        "270",
        "--min-entropy",
        "7.5",
        "--max-entropy",
        "7.6",
        "--no-progress",
        "--disable-reasoning-tokens",
        "--chunk-size",
        "512",
        "-o",
        output_path.file_name().unwrap().to_str().unwrap(),
    ]);
    train.assert().failure();
}
