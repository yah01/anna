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
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

impl From<u8> for MetricType {
    fn from(value: u8) -> Self {
        match value {
            1 => MetricType::L2,
            _ => MetricType::None,
        }
    }
}
