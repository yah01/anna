use std::sync::Arc;

use crate::{metric::l2_distance, VectorAccessor};

#[derive(Debug)]
pub struct Cluster {
    accessor: Arc<dyn VectorAccessor>,
    pub centroid: Vec<f32>,
    pub elements: Vec<usize>,
}

impl Cluster {
    pub fn new(accessor: Arc<dyn VectorAccessor>) -> Cluster {
        let dim = accessor.dim();
        Cluster {
            accessor,
            centroid: vec![0f32; dim],
            elements: Vec::new(),
        }
    }

    pub fn with_centroid(accessor: Arc<dyn VectorAccessor>, centroid: &[f32]) -> Self {
        Self {
            accessor,
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
        let mut new = Self::new(self.accessor.clone());
        new.elements = self
            .elements
            .drain((self.elements.len() - split_num)..)
            .collect();
        new
    }

    pub fn calc_centroid(&mut self) -> f32 {
        if self.elements.len() == 0 {
            panic!("can't calculate centroid for empty cluster");
        }

        self.centroid.fill(0f32);

        for id in self.elements.iter() {
            let vec = self.accessor.get(*id);
            for i in 0..self.accessor.dim() {
                self.centroid[i] += vec[i];
            }
        }

        for i in 0..self.accessor.dim() {
            self.centroid[i] /= self.elements.len() as f32;
        }

        let mut wcss = 0f32;
        for id in self.elements.iter() {
            let vec = self.accessor.get(*id);
            wcss += l2_distance(vec, &self.centroid);
        }

        wcss
    }
}
