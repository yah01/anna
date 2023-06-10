pub mod error;
pub mod index;
pub mod metric;
mod test_util;
pub mod accessor;

use std::io;
use std::{ops::Deref, sync::Arc};

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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

    async fn serialize<T: AsyncWriteExt + Unpin + Send>(
        &self,
        writer: &mut tokio::io::BufWriter<T>,
    ) -> Result<(), io::Error>;

    async fn deserialize<T: AsyncReadExt + Unpin + Send>(
        &mut self,
        reader: &mut tokio::io::BufReader<T>,
    ) -> Result<(), io::Error>;
}

#[async_trait]
pub trait VectorAccessor: Send + Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> &[f32];
}
