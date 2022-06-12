extern crate ordered_float;

use std::collections::BinaryHeap;

use ordered_float::NotNan;

use crate::index::TopkIntermediate;

use super::Index;

#[derive(Clone, Debug)]
pub struct IndexFlatL2 {
    dim: usize,
    data: Vec<f32>,
}

impl IndexFlatL2 {
    pub fn new(dim: usize) -> Self {
        IndexFlatL2 {
            dim,
            data: Vec::new(),
        }
    }
}

impl Index for IndexFlatL2 {
    fn calc_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut distance = 0.0;
        for i in 0..self.dim {
            distance += (a[i] - b[i]).powi(2);
        }

        distance
    }

    fn train(&mut self, dataset: &[f32]) {}

    fn is_trained(&self) -> bool {
        true
    }

    fn add(&mut self, data: &[f32]) {
        assert!(data.len() % self.dim == 0, "dim not match");

        let n = data.len() / self.dim;
        self.data.extend_from_slice(data);
    }

    fn search(&self, queries: &[f32], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        assert!(queries.len() % self.dim == 0, "query dim not match");

        let n = self.data.len() / self.dim;
        let qn = queries.len() / self.dim;

        let mut ids = Vec::with_capacity(qn);
        let mut distances = Vec::with_capacity(qn);

        let mut query_topk = Vec::with_capacity(qn);
        for _ in 0..qn {
            query_topk.push(BinaryHeap::with_capacity(k));
        }

        for i in 0..qn {
            let query = &queries[i * self.dim..(i + 1) * self.dim];

            for id in 0..n {
                let data = &self.data[id * self.dim..(id + 1) * self.dim];

                let distance = self.calc_distance(query, data);

                let topk = &mut query_topk[i];
                if topk.len() < k {
                    topk.push(TopkIntermediate { id, distance });
                } else if topk.peek().unwrap().distance > distance {
                    topk.pop();
                    topk.push(TopkIntermediate { id, distance })
                }
            }
        }

        for topk in query_topk {
            let mut query_ids = Vec::with_capacity(k);
            let mut query_distances = Vec::with_capacity(k);
            for res in topk {
                query_ids.push(res.id);
                query_distances.push(res.distance);
            }

            ids.push(query_ids);
            distances.push(query_distances);
        }

        (ids, distances)
    }
}
