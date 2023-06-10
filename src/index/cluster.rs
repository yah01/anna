use std::sync::Arc;

use crate::VectorAccessor;

#[derive(Debug)]
pub struct Cluster {
    pub centroid: Vec<f32>,
    pub elements: Vec<usize>,
}

impl Cluster {
    pub fn new() -> Cluster {
        Cluster {
            centroid: Vec::new(),
            elements: Vec::new(),
        }
    }

    pub fn with_centroid(centroid: &[f32]) -> Self {
        Self {
            centroid: Vec::from(centroid),
            elements: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn add(&mut self, id: usize) {
        self.elements.push(id);
    }

    pub fn split(&mut self) -> Self {
        let split_num = self.elements.len() / 2;
        let mut new = Self::new();
        new.elements = self
            .elements
            .drain((self.elements.len() - split_num)..)
            .collect();

        new
    }

    pub fn calc_centroid(&mut self, accessor: Arc<dyn VectorAccessor>) {
        if self.elements.len() == 0 {
            panic!("can't calculate centroid for empty cluster");
        }

        self.centroid.resize(accessor.dim(), 0f32);

        for id in self.elements.iter() {
            let vec = accessor.get(*id);
            for i in 0..accessor.dim() {
                self.centroid[i] += vec[i];
            }
        }

        for i in 0..accessor.dim() {
            self.centroid[i] /= self.elements.len() as f32;
        }
    }
}
