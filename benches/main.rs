use std::{fmt::Display, sync::Arc};

use anna::index::ivf::Ivf;
use anna::test_util::{gen_floats, gen_vectors};
use anna::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

struct Parameter {
    pub dataset_size: usize,

    pub dim: usize,

    pub train_option: TrainOption,
    pub search_option: SearchOption,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "params(dataset_size:{}, dim: {}, nlist: {}, nprobe: {}, topk: {})",
            self.dataset_size,
            self.dim,
            self.train_option.nlist,
            self.search_option.nprobe,
            self.search_option.topk
        ))
    }
}

static DEFAULT_TRAIN_OPTION: TrainOption = TrainOption {
    iteration_num: None,
    nlist: 256,
    metric_type: metric::MetricType::L2,
};

static DEFAULT_SEARCH_OPTION: SearchOption = SearchOption {
    nprobe: 10,
    topk: 10,
};

static PARAMETERS: &'static [Parameter] = &[
    Parameter {
        dataset_size: 1_000_000,
        dim: 128,
        train_option: DEFAULT_TRAIN_OPTION,
        search_option: DEFAULT_SEARCH_OPTION,
    },
    Parameter {
        dataset_size: 1_000_000,
        dim: 768,
        train_option: DEFAULT_TRAIN_OPTION,
        search_option: DEFAULT_SEARCH_OPTION,
    },
];

pub fn ivf_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf");
    group.sample_size(10);

    for parameter in PARAMETERS.iter() {
        group.bench_with_input(
            BenchmarkId::new("ivf_train", parameter),
            parameter,
            |b, parameter| {
                let accessor = Arc::new(gen_vectors(
                    parameter.dataset_size,
                    parameter.dim,
                    parameter.train_option.nlist,
                ));
                let mut ivf = Ivf::new(accessor.clone());

                b.iter(|| {
                    ivf.train(&parameter.train_option);
                })
            },
        );
    }
}

pub fn ivf_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf");
    group.sample_size(30);

    for parameter in PARAMETERS.iter() {
        group.bench_with_input(
            BenchmarkId::new("ivf_search", parameter),
            parameter,
            |b, parameter| {
                let accessor = Arc::new(gen_vectors(
                    parameter.dataset_size,
                    parameter.dim,
                    parameter.train_option.nlist,
                ));
                let mut ivf = Ivf::new(accessor.clone());

                ivf.train(&parameter.train_option);

                let query = gen_floats(parameter.dim);
                let deleted = roaring::RoaringBitmap::new();

                b.iter(|| {
                    ivf.search(&query, &deleted, &parameter.search_option);
                })
            },
        );
    }
}

criterion_group!(benches, ivf_train, ivf_search);
criterion_main!(benches);
