use std::sync::Arc;

use anna::index::ivf::Ivf;
use anna::test_util::{gen_floats, gen_vectors};
use anna::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const DIM: usize = 128;
const CLUSTER_NUM: usize = 128;
const DATASET_SIZE: usize = 1_000_000;

fn ivf_train(ivf: &mut Ivf) {
    let option = TrainOption {
        iteration_num: None,
        nlist: CLUSTER_NUM,
        metric_type: metric::MetricType::L2,
    };
    ivf.train(&option);
}

fn ivf_search(ivf: &Ivf) {
    let query = gen_floats(DIM);
    let deleted = roaring::RoaringBitmap::new();
    let option = SearchOption {
        nprobe: 8,
        topk: 10,
    };
    ivf.search(&query, &deleted, &option);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let accessor = Arc::new(gen_vectors(DATASET_SIZE, DIM, CLUSTER_NUM));
    let mut ivf = Ivf::new(accessor.clone());

    let mut group = c.benchmark_group("ivf");
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("ivf_train", DATASET_SIZE),
        &DATASET_SIZE,
        |b, _| b.iter(|| ivf_train(&mut ivf)),
    );

    group.bench_with_input(
        BenchmarkId::new("ivf_search", DATASET_SIZE),
        &DATASET_SIZE,
        |b, _| b.iter(|| ivf_search(&ivf)),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
