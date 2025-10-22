use std::hint::black_box;

use bbpe::{Trainer, TrainerConfig};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput};

fn build_sequences() -> Vec<Vec<u8>> {
    let mut data = Vec::with_capacity(1 << 20);
    for chunk in 0..1024 {
        data.extend_from_slice(&[
            (chunk & 0xFF) as u8,
            ((chunk >> 1) & 0xFF) as u8,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
        ]);
    }
    // Split into 4 KiB chunks
    data.chunks(4096).map(|chunk| chunk.to_vec()).collect()
}

fn bench_training(c: &mut Criterion) {
    let sequences = build_sequences();
    let total_bytes: usize = sequences.iter().map(|seq| seq.len()).sum();
    let cfg = TrainerConfig::builder()
        .special_tokens(Vec::<String>::new())
        .target_vocab_size(512)
        .min_frequency(2)
        .show_progress(false)
        .build()
        .expect("configuration");

    let mut group = c.benchmark_group("train_binary_corpus");
    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function(BenchmarkId::from_parameter("MiB_1"), |b| {
        b.iter(|| {
            let trainer = Trainer::new(cfg.clone());
            let artefacts = trainer.train_from_sequences(&sequences).expect("training");
            let _ = black_box(artefacts);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_training);
criterion_main!(benches);
