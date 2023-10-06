use bytes::{BufMut, BytesMut};
use datafusion::arrow::datatypes::ToByteSlice;

use crate::accessor::MemoryVectorAccessor;
use crate::VectorAccessor;

pub fn gen_vectors(n: usize, dim: usize, cluster_num: usize) -> impl VectorAccessor {
    let centroids: Vec<_> = (0..cluster_num).map(|_| gen_floats(dim)).collect();

    let mut vectors = BytesMut::with_capacity(n * dim * std::mem::size_of::<f32>());
    for i in 0..n {
        vectors.put_slice(centroids[i % cluster_num].to_byte_slice());
    }

    MemoryVectorAccessor::new(dim, vectors.freeze())
}

pub fn gen_floats(n: usize) -> Vec<f32> {
    (0..n).map(|_| rand::random()).collect()
}
