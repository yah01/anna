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

pub mod cluster;
mod graph;
pub mod hnsw;
pub mod ivf;
pub mod util;

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{AnnIndex, VectorAccessor};

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    IvfFlat,
    Hnsw,
}

pub fn new(typ: IndexType, accessor: Arc<dyn VectorAccessor>) -> Arc<RwLock<dyn AnnIndex>> {
    match typ {
        IndexType::IvfFlat => Arc::new(RwLock::new(ivf::Ivf::new(accessor))),
        // IndexType::Hnsw => Arc::new(ivf::Hnsw::new(accessor)),
        _ => unimplemented!("unsupported index type"),
    }
}
