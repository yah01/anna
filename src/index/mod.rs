use ordered_float::Float;

pub use self::flat::*;

pub trait Index {
    fn calc_distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn train(&mut self, dataset: &[f32]);
    fn is_trained(&self) -> bool;
    fn add(&mut self, data: &[f32]);
    fn search(&self, queries: &[f32], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>);
}

struct TopkIntermediate {
    pub id: usize,
    pub distance: f32,
}

impl Ord for TopkIntermediate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

impl PartialOrd for TopkIntermediate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for TopkIntermediate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for TopkIntermediate {}

mod flat;
