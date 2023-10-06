use crate::VectorAccessor;
use bytes::Bytes;
use datafusion::arrow::array::*;

#[derive(Debug)]
pub struct MemoryVectorAccessor {
    dim: usize,
    vectors: Bytes,
}

impl MemoryVectorAccessor {
    pub fn new(dim: usize, vectors: Bytes) -> Self {
        Self { dim, vectors }
    }
}

impl VectorAccessor for MemoryVectorAccessor {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.vectors.len() / self.dim / std::mem::size_of::<f32>()
    }

    #[inline(always)]
    fn get(&self, index: usize) -> &[f32] {
        let size = self.dim * std::mem::size_of::<f32>();
        let vec = &self.vectors[index * size..(index + 1) * size];
        unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const f32, self.dim()) }
    }
}

#[derive(Debug)]
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
