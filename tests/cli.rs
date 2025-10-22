use assert_cmd::Command;
use serde_json::Value;
use std::fs;
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
