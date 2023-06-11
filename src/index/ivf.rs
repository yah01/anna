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
use super::util;
use crate::*;
use log::warn;
use ordered_float::NotNan;
use std::{collections::BinaryHeap, sync::Arc};
use std::{io, pin::Pin};
use tokio::io::AsyncWriteExt;

const VERSION: u16 = 1;

pub struct Ivf {
    vectors: Arc<dyn VectorAccessor>,
    clusters: Vec<Cluster>,
    metric_type: metric::MetricType,
}

impl Ivf {
    pub fn new(vectors: Arc<dyn VectorAccessor>) -> Self {
        Self {
            vectors: vectors,
            clusters: Vec::new(),
            metric_type: metric::MetricType::None,
        }
    }
}

#[async_trait]
impl crate::AnnIndex for Ivf {
    fn train(&mut self, option: &TrainOption) {
        self.metric_type = option.metric_type;
        self.clusters = util::train_clusters(self.metric_type, self.vectors.clone(), option);
    }

    fn search(
        &self,
        query_vector: &[f32],
        deleted: &roaring::RoaringBitmap,
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
            for i in &cluster.elements {
                if deleted.contains(*i as u32) {
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

        topk.iter().map(|(_, i)| *i).collect()
    }

    async fn serialize(
        &self,
        mut writer: Pin<Box<dyn tokio::io::AsyncWrite + Send>>,
    ) -> Result<(), io::Error> {
        // metadata part
        writer.write_u16_le(VERSION).await?;
        writer.write_u8(self.metric_type as u8).await?;
        writer.write_u32_le(self.vectors.dim() as u32).await?;
        writer.write_u32_le(self.clusters.len() as u32).await?;

        for cluster in &self.clusters {
            writer.write_u32_le(cluster.len() as u32).await?;
            for v in &cluster.centroid {
                writer.write_f32_le(*v).await?;
            }
            for v in &cluster.elements {
                writer.write_u32_le(*v as u32).await?;
            }
        }

        writer.flush().await
    }

    async fn deserialize(
        &mut self,
        mut reader: Pin<Box<dyn tokio::io::AsyncRead + Send>>,
    ) -> Result<(), io::Error> {
        let ivf_version = reader.read_u16_le().await?;
        if ivf_version != VERSION {
            warn!(
                "read newer version {} ivf index file, current version is {}",
                ivf_version, VERSION
            );
        }

        self.metric_type = metric::MetricType::from(reader.read_u8().await?);
        let dim = reader.read_u32_le().await? as usize;
        let nlist = reader.read_u32_le().await?;
        self.clusters = Vec::with_capacity(nlist as usize);
        for _ in 0..nlist {
            let mut cluster = Cluster::new();
            cluster.centroid.reserve(dim);
            let size = reader.read_u32_le().await? as usize;
            for _ in 0..dim as usize {
                cluster.centroid.push(reader.read_f32_le().await?);
            }

            cluster.elements.reserve(size);
            for _ in 0..size {
                cluster.add(reader.read_u32_le().await? as usize);
            }

            self.clusters.push(cluster);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::index::ivf::*;
    use crate::test_util::gen_vectors;
    use roaring::RoaringBitmap;
    use tokio::io::{BufReader, BufWriter};

    const DIM: usize = 32;
    const CLUSTER_NUM: usize = 32;
    const DATASET_SIZE: usize = CLUSTER_NUM * CLUSTER_NUM;

    #[tokio::test]
    async fn test_ivf() {
        let accessor = Arc::new(gen_vectors(DATASET_SIZE, DIM, CLUSTER_NUM));
        let mut ivf = Ivf::new(accessor.clone());

        let option = TrainOption {
            iteration_num: None,
            nlist: CLUSTER_NUM,
            metric_type: metric::MetricType::L2,
        };
        ivf.train(&option);

        let option = SearchOption {
            nprobe: CLUSTER_NUM / 2,
            topk: CLUSTER_NUM + 1,
        };

        let bitmap = RoaringBitmap::new();
        for i in 0..accessor.len() {
            let query = accessor.get(i);
            let result = ivf.search(query, &bitmap, &option);
            assert_eq!(
                result.len(),
                option.topk,
                "failed to search, cluster_num={}",
                ivf.clusters.len()
            );

            let mut close_count = 0;
            for id in &result {
                let distance = ivf.metric_type.distance(query, accessor.get(*id));
                if distance == 0f32 {
                    close_count += 1;
                }
            }
            assert_eq!(close_count, CLUSTER_NUM, "result: {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_ivf_serde() {
        let accessor = Arc::new(gen_vectors(DATASET_SIZE, DIM, CLUSTER_NUM));
        let mut ivf = Ivf::new(accessor.clone());

        let option = TrainOption {
            iteration_num: None,
            nlist: CLUSTER_NUM,
            metric_type: metric::MetricType::L2,
        };
        ivf.train(&option);

        let temp_dir = temp_dir::TempDir::new().unwrap();
        let path = temp_dir.path().join("ivf_serde.ivf");
        let ivf_file = tokio::fs::File::create(&path).await.unwrap();
        let buf = BufWriter::new(ivf_file);
        ivf.serialize(Box::pin(buf)).await.unwrap();

        let ivf_file = tokio::fs::File::open(&path).await.unwrap();
        let buf = BufReader::new(ivf_file);
        let mut ivf = Ivf::new(accessor.clone());
        ivf.deserialize(Box::pin(buf)).await.unwrap();

        assert_eq!(ivf.metric_type, metric::MetricType::L2);
        assert_eq!(ivf.clusters.len(), option.nlist);

        let option = SearchOption {
            nprobe: CLUSTER_NUM / 2,
            topk: CLUSTER_NUM + 1,
        };

        let bitmap = RoaringBitmap::new();
        for i in 0..accessor.len() {
            let query = accessor.get(i);
            let result = ivf.search(query, &bitmap, &option);
            assert_eq!(
                result.len(),
                option.topk,
                "failed to search, cluster_num={}",
                ivf.clusters.len()
            );

            let mut close_count = 0;
            for id in &result {
                let distance = ivf.metric_type.distance(query, accessor.get(*id));
                if distance == 0f32 {
                    close_count += 1;
                }
            }
            assert_eq!(close_count, CLUSTER_NUM, "result: {:?}", result);
        }
    }
}
