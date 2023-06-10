// Copyright 2023 yah01
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{collections::BinaryHeap, sync::Arc};

use ordered_float::NotNan;

use crate::*;

use super::util;

pub struct Ivf {
    vectors: Arc<dyn crate::VectorAccessor>,
    nlist: usize,
    clusters: Vec<Cluster>,
    metric_type: metric::MetricType,
}

impl crate::AnnIndex for Ivf {
    fn train(&mut self, option: &TrainOption) {
        self.metric_type = option.metric_type;
        self.clusters = util::rand_centroids(option.nlist, self.vectors.clone());

        let mut assign = vec![0; self.vectors.len()];
        for iter in 0..option.iteration_num.unwrap_or(2000) {
            let mut new_clusters: Vec<_> = (0..option.nlist)
                .map(|_| Cluster::empty(self.vectors.dim()))
                .collect();

            for i in 0..self.vectors.len() {
                let vec = self.vectors.get(i);
                let target = self
                    .clusters
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        self.metric_type
                            .distance(&a.centroid, vec)
                            .total_cmp(&self.metric_type.distance(&b.centroid, vec))
                    })
                    .map(|(index, _)| index)
                    .unwrap();

                assign[i] = target;
                new_clusters[target].add(vec);
            }

            for cluster in new_clusters.iter_mut() {
                cluster.calc_centroid();
            }

            self.clusters = new_clusters;
        }

        for (i, assign) in assign.iter().enumerate() {
            self.clusters[*assign].add_element(i);
        }
    }

    fn search(
        &self,
        query_vector: &[f32],
        bitset: &roaring::RoaringBitmap,
        option: &SearchOption,
    ) -> Vec<usize> {
        let mut cluster_distances: Vec<_> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.metric_type.distance(&c.centroid, query_vector)))
            .collect();

        cluster_distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        cluster_distances.truncate(option.nprobe);

        let mut topk: BinaryHeap<(NotNan<f32>, usize)> = BinaryHeap::with_capacity(option.topk);
        let clusters = cluster_distances.iter().map(|(i, _)| &self.clusters[*i]);
        for cluster in clusters {
            for i in cluster.elements() {
                if !bitset.contains(*i as u32) {
                    continue;
                }

                let distance = self
                    .metric_type
                    .distance(query_vector, self.vectors.get(*i));

                if topk.len() == option.topk {
                    if topk.peek().unwrap().0.total_cmp(&distance).is_gt() {
                        topk.pop();
                    } else {
                        continue;
                    }
                }
                topk.push((NotNan::new(distance).unwrap(), *i));
            }
        }

        topk.into_iter().map(|(_, i)| i).collect()
    }

    async fn serialize(&self, writer: impl std::io::Write) -> Result<(), error::Error> {}
}

pub struct Cluster {
    size: usize,
    centroid: Vec<f32>,
    elements: Vec<usize>,
}

impl Cluster {
    pub fn empty(dim: usize) -> Cluster {
        Cluster {
            size: 0,
            centroid: vec![0f32; dim],
            elements: Vec::new(),
        }
    }

    pub fn new(size: usize, centroid: &[f32]) -> Self {
        Self {
            size,
            centroid: Vec::from(centroid),
            elements: Vec::new(),
        }
    }

    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    pub fn add_element(&mut self, i: usize) {
        self.elements.push(i);
    }

    pub fn add(&mut self, vec: &[f32]) {
        self.size += 1;
        for i in 0..vec.len() {
            self.centroid[i] += vec[i];
        }
    }

    pub fn calc_centroid(&mut self) {
        for i in 0..self.centroid.len() {
            self.centroid[i] /= self.size as f32;
        }
    }
}
