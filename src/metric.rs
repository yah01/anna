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

use std::simd::{Simd, SimdFloat};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MetricType {
    None,
    L2,
}

impl MetricType {
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            MetricType::None => panic!("miss to set metric type"),
            MetricType::L2 => l2_distance(a, b),
        }
    }
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;

    let mut sum = a
        .array_chunks::<LANES>()
        .map(|&a| Simd::<_, LANES>::from_array(a))
        .zip(
            b.array_chunks::<LANES>()
                .map(|&b| Simd::<_, LANES>::from_array(b)),
        )
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .fold(Simd::<_, LANES>::splat(0.0), std::ops::Add::add)
        .reduce_sum();
    let remain = a.len() - (a.len() % LANES);
    sum += a[remain..]
        .iter()
        .zip(&b[remain..])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>();
    sum
}

impl From<u8> for MetricType {
    fn from(value: u8) -> Self {
        match value {
            1 => MetricType::L2,
            _ => MetricType::None,
        }
    }
}
