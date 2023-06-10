use crate::VectorAccessor;

pub struct MemoryVectorAccessor {
    dim: usize,
    vectors: Vec<f32>,
}

impl MemoryVectorAccessor {
    pub fn new(dim: usize, vectors: Vec<f32>) -> Self {
        Self { dim, vectors }
    }
}

impl VectorAccessor for MemoryVectorAccessor {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    #[inline(always)]
    fn get(&self, index: usize) -> &[f32] {
        &self.vectors[index * self.dim..(index + 1) * self.dim]
    }
}
