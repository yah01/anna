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

use super::cluster::Cluster;
use crate::{metric::MetricType, TrainOption, VectorAccessor};
use std::{cmp, sync::Arc};

const MAX_CLUSTER_SIZE: usize = 256;

pub fn rand_centroids(n: usize, vectors: Arc<dyn crate::VectorAccessor>) -> Vec<Cluster> {
    let vec_num = vectors.len();
    (0..n)
        .into_iter()
        .map(|_| {
            Cluster::with_centroid(
                vectors.clone(),
                vectors.get(rand::random::<usize>() % vec_num),
            )
        })
        .collect()
}

pub fn train_clusters(
    metric_type: MetricType,
    vectors: Arc<dyn VectorAccessor>,
    option: &TrainOption,
) -> Vec<Cluster> {
    let mut clusters = rand_centroids(option.nlist, vectors.clone());

    let iter_num = option.iteration_num.unwrap_or(10);
    let train_size = cmp::min(option.nlist * MAX_CLUSTER_SIZE, vectors.len());
    let mut last_wcss = 0f32;
    for _ in 0..iter_num {
        let mut wcss = 0f32;
        let mut new_clusters: Vec<_> = (0..option.nlist)
            .map(|_| Cluster::new(vectors.clone()))
            .collect();

        for id in 0..train_size {
            let vec = vectors.get(id);
            let mut target = 0;
            let mut min_distance = metric_type.distance(&clusters[0].centroid, vec);

            for i in 1..clusters.len() {
                let distance = metric_type.distance(&clusters[i].centroid, vec);
                if distance < min_distance {
                    target = i;
                    min_distance = distance;
                }
            }

            new_clusters[target].add(id);
        }

        // split larger cluster to reach the expected number of clusters
        let mut new_clusters: Vec<_> = new_clusters.into_iter().filter(|c| c.len() > 0).collect();
        let mut split_num = clusters.len() - new_clusters.len();
        let mut splited_clusters = Vec::with_capacity(split_num);
        new_clusters.sort_by(|a, b| a.len().cmp(&b.len()));

        let mut max_cluster_idx = new_clusters.len() - 1;
        while splited_clusters.len() < split_num {
            splited_clusters.push(new_clusters[max_cluster_idx].split());
            if max_cluster_idx == 0 {
                new_clusters.append(&mut splited_clusters);
                split_num = clusters.len() - new_clusters.len();
                continue;
            }

            if new_clusters[max_cluster_idx - 1].len() > new_clusters[max_cluster_idx].len() {
                max_cluster_idx -= 1;
            }

            if let Some(first) = splited_clusters.first() {
                if first.len() > new_clusters[max_cluster_idx].len() {
                    new_clusters.append(&mut splited_clusters);
                    split_num = clusters.len() - new_clusters.len();
                }
            }
        }
        new_clusters.append(&mut splited_clusters);

        // calculate the centroid for each cluster
        for cluster in new_clusters.iter_mut() {
            wcss += cluster.calc_centroid();
        }

        if wcss > last_wcss {
            break;
        }
        clusters = new_clusters;
        if wcss >= last_wcss {
            break;
        }
        last_wcss = wcss;
    }

    // assign the vectors not in train set
    for id in train_size..vectors.len() {
        let vec = vectors.get(id);
        let mut target = 0;
        let mut min_distance = metric_type.distance(&clusters[0].centroid, vec);

        for i in 1..clusters.len() {
            let distance = metric_type.distance(&clusters[i].centroid, vec);
            if distance < min_distance {
                target = i;
                min_distance = distance;
            }
        }

        clusters[target].add(id);
    }
    clusters
}
