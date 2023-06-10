pub mod error;
pub mod index;
pub mod metric;

use std::{ops::Deref, sync::Arc};

use async_trait::async_trait;
pub struct TrainOption {
    iteration_num: Option<usize>,
    nlist: usize,
    metric_type: metric::MetricType, // ... index related options
}

pub struct SearchOption {
    nprobe: usize,
    topk: usize,
    // ... another index related options
}

// T could be f16, f32, f64, u8
#[async_trait]
pub trait AnnIndex {
    fn train(&mut self, option: &TrainOption);

    fn search(
        &self,
        query_vector: &[f32],
        bitset: &roaring::RoaringBitmap,
        option: &SearchOption,
    ) -> Vec<usize>;

    async fn serialize(&self, writer: impl std::io::Write) -> Result<(), error::Error>;

    async fn deserialize(reader: impl std::io::Read) -> Result<Self, error::Error>
    where
        Self: Sized;
}

#[async_trait]
pub trait VectorAccessor {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> &[f32];
}
