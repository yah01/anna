use crate::VectorAccessor;
use datafusion::arrow::array::*;
use std::sync::Arc;

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

pub struct ArrowVectorAccessor {
    vectors: FixedSizeBinaryArray,
}

impl ArrowVectorAccessor {
    pub fn new(array: FixedSizeBinaryArray) -> Self {
        Self { vectors: array }
    }
}

impl VectorAccessor for ArrowVectorAccessor {
    fn dim(&self) -> usize {
        self.vectors.value_length() as usize / std::mem::size_of::<f32>()
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn get(&self, index: usize) -> &[f32] {
        let vec = self.vectors.value(index);
        unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const f32, self.dim()) }
    }
}
