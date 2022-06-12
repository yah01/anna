use std::collections::BinaryHeap;

use rand::Rng;

use super::TopkIntermediate;

pub fn init_kmeans_centroids(nlist: usize, dim: usize, data: &[f32]) -> Vec<f32> {
    let n = data.len() / dim;
    let mut rng = rand::thread_rng();

    let mut centroids = Vec::with_capacity(nlist * dim);

    for _ in 0..nlist {
        let idx = rng.gen::<usize>() % n;
        centroids.extend_from_slice(vector_at(data, idx, dim));
    }

    centroids
}

pub fn get_nearest_vector(
    dim: usize,
    vector: &[f32],
    data: &[f32],
    metric_fn: fn(&[f32], &[f32]) -> f32,
) -> (usize, f32) {
    let n = data.len() / dim;

    let mut idx = 0;
    let mut min_distance = metric_fn(vector, vector_at(data, 0, dim));
    for i in 1..n {
        let distance = metric_fn(vector, vector_at(data, i, dim));
        if min_distance > distance {
            idx = i;
            min_distance = distance;
        }
    }

    (idx, min_distance)
}

pub fn get_topk(
    dim: usize,
    k: usize,
    query: &[f32],
    data: &[f32],
    metric_fn: fn(&[f32], &[f32]) -> f32,
) -> (Vec<usize>, Vec<f32>) {
    let n = data.len() / dim;

    let mut topk = BinaryHeap::with_capacity(k);
    for id in 0..n {
        let distance = metric_fn(query, vector_at(data, id, dim));

        if topk.len() < k {
            topk.push(TopkIntermediate { id, distance });
        } else if topk.peek().unwrap().distance > distance {
            topk.pop();
            topk.push(TopkIntermediate { id, distance })
        }
    }

    let mut ids = Vec::with_capacity(k);
    let mut distances = Vec::with_capacity(k);
    for intermediate in topk {
        ids.push(intermediate.id);
        distances.push(intermediate.distance);
    }

    (ids, distances)
}

pub fn vector_at(data: &[f32], idx: usize, dim: usize) -> &[f32] {
    &data[idx * dim..(idx + 1) * dim]
}

pub fn vector_at_mut(data: &mut [f32], idx: usize, dim: usize) -> &mut [f32] {
    &mut data[idx * dim..(idx + 1) * dim]
}
