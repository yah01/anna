use crate::accessor::MemoryVectorAccessor;
use crate::VectorAccessor;

pub fn gen_vectors(n: usize, dim: usize) -> impl VectorAccessor {
    let cluster_num = 10;
    let centroids: Vec<_> = (0..cluster_num).map(|_| gen_floats(dim)).collect();

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors.extend_from_slice(&centroids[i % cluster_num]);
    }

    MemoryVectorAccessor::new(dim, vectors)
}

pub fn gen_floats(n: usize) -> Vec<f32> {
    (0..n).map(|_| rand::random()).collect()
}
