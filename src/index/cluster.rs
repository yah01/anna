pub struct Cluster {
    pub size: usize,
    pub centroid: Vec<f32>,
    pub elements: Vec<usize>,
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