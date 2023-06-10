use rand::Rng;

use crate::accessor::MemoryVectorAccessor;
use crate::VectorAccessor;

pub fn gen_vectors(n: usize, dim: usize, cluster_num: usize) -> impl VectorAccessor {
    let centroids: Vec<_> = (0..cluster_num).map(|_| gen_floats(dim)).collect();

    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors.extend_from_slice(&centroids[i % cluster_num]);
    }

    MemoryVectorAccessor::new(dim, vectors)
}

pub fn gen_floats(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}
