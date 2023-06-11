#![feature(array_chunks)]
#![feature(portable_simd)]

pub mod accessor;
pub mod error;
pub mod index;
pub mod metric;
pub mod test_util;

use async_trait::async_trait;
use std::{io, pin::Pin};
use tokio::io::AsyncReadExt;

#[derive(Debug, Clone, Copy)]
pub struct TrainOption {
    pub iteration_num: Option<usize>,
    pub nlist: usize,
    pub metric_type: metric::MetricType, // ... index related options
}

#[derive(Debug, Clone, Copy)]
pub struct SearchOption {
    pub nprobe: usize,
    pub topk: usize,
    // ... another index related options
}

// T could be f16, f32, f64, u8
#[async_trait]
pub trait AnnIndex: Send + Sync {
    fn train(&mut self, option: &TrainOption);

    fn search(
        &self,
        query_vector: &[f32],
        deleted: &roaring::RoaringBitmap,
        option: &SearchOption,
    ) -> Vec<usize>;

    async fn serialize(
        &self,
        mut writer: Pin<Box<dyn tokio::io::AsyncWrite + Send>>,
    ) -> Result<(), io::Error>;

    async fn deserialize(
        &mut self,
        mut reader: Pin<Box<dyn tokio::io::AsyncRead + Send>>,
    ) -> Result<(), io::Error>;
}

#[async_trait]
pub trait VectorAccessor: Send + Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> &[f32];
}
