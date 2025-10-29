use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn temp_workspace() -> TempDir {
    tempfile::tempdir().expect("create tempdir")
}

fn run_command(cmd: &mut Command) {
    cmd.assert().success();
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
        info_text.contains("Vocab size"),
        "info output contained expected summary"
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
