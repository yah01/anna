use std::cmp::*;
use std::collections::BinaryHeap;

use crate::IndexFlatL2;

use super::utils::{get_nearest_vector, get_topk, init_kmeans_centroids, vector_at, vector_at_mut};
use super::{Index, TopkIntermediate};

const kKmeansConvergePercentage: f32 = 0.01;

pub struct IndexIVFFlat {
    dim: usize,
    nlist: usize,
    nprobe: usize,
    is_trained: bool,
    data: Vec<f32>,
    cluster: Vec<Vec<usize>>,
    centroids: Vec<f32>,
}

impl IndexIVFFlat {
    pub fn new(dim: usize, nlist: usize) -> Self {
        let mut cluster = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            cluster.push(Vec::new());
        }

        IndexIVFFlat {
            dim,
            nlist,
            nprobe: 1,
            is_trained: false,
            data: Vec::new(),
            cluster,
            centroids: Vec::new(),
        }
    }
}

impl Index for IndexIVFFlat {
    fn calc_distance(a: &[f32], b: &[f32]) -> f32 {
        IndexFlatL2::calc_distance(a, b)
    }

    fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;

        let mut centroids = init_kmeans_centroids(self.nlist, self.dim, data);
        let mut belong = vec![0; n];
        let mut cluster_size = vec![0; self.nlist];

        let mut last_wcss = -1.0;
        loop {
            cluster_size.fill(0);
            let mut wcss = 0.0;

            for i in 0..n {
                let vector = vector_at(data, i, self.dim);

                let (centroid_idx, distance) =
                    get_nearest_vector(self.dim, vector, &centroids, IndexIVFFlat::calc_distance);

                belong[i] = centroid_idx;
                cluster_size[centroid_idx] += 1;
                wcss += distance;
            }

            if (last_wcss - wcss) / last_wcss <= kKmeansConvergePercentage && last_wcss >= 0.0 {
                println!("new wcss: {}", wcss);
                break;
            }
            last_wcss = wcss;
            println!("new wcss: {}", wcss);

            centroids.fill(0.);

            for i in 0..n {
                let vector = vector_at(data, i, self.dim);
                let centroid = vector_at_mut(&mut centroids, belong[i], self.dim);

                for j in 0..self.dim {
                    centroid[j] += vector[j] / cluster_size[belong[i]] as f32;
                }
            }
        }

        self.centroids = centroids;
        self.is_trained = true;
    }

    fn is_trained(&self) -> bool {
        return self.is_trained;
    }

    fn add(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;

        self.data.extend_from_slice(data);
        self.cluster.reserve(n);

        for i in 0..n {
            let (idx, _) = get_nearest_vector(
                self.dim,
                vector_at(data, i, self.dim),
                &self.centroids,
                Self::calc_distance,
            );

            self.cluster[idx].push(i);
        }
    }

    fn search(&self, queries: &[f32], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = self.data.len() / self.dim;
        let qn = queries.len() / self.dim;
        let nprobe = min(self.nprobe, self.nlist);

        let mut ids = Vec::with_capacity(qn);
        let mut distances = Vec::with_capacity(qn);
        let mut query_topk = Vec::with_capacity(qn);
        for _ in 0..qn {
            query_topk.push(BinaryHeap::with_capacity(k));
        }

        for i in 0..qn {
            let query = vector_at(queries, i, self.dim);
            let (top_clusters, _) = get_topk(
                self.dim,
                nprobe,
                query,
                &self.centroids,
                Self::calc_distance,
            );

            let topk = &mut query_topk[i];

            for c in top_clusters {
                for id in &self.cluster[c] {
                    let id = *id;
                    let distance =
                        IndexFlatL2::calc_distance(query, vector_at(&self.data, id, self.dim));

                    if topk.len() < k {
                        topk.push(TopkIntermediate { id, distance });
                    } else if topk.peek().unwrap().distance > distance {
                        topk.pop();
                        topk.push(TopkIntermediate { id, distance })
                    }
                }
            }
        }

        for topk in query_topk {
            let mut query_ids = Vec::with_capacity(k);
            let mut query_distances = Vec::with_capacity(k);
            for res in topk.into_sorted_vec() {
                query_ids.push(res.id);
                query_distances.push(res.distance);
            }

            ids.push(query_ids);
            distances.push(query_distances);
        }

        (ids, distances)
    }
}
