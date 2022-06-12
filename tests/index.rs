use anna::{Index, IndexFlatL2, IndexIVFFlat};
use rand::Rng;

#[test]
fn IndexFlatL2_test() {
    let n = 10000;
    let dim = 128;
    let nq = 10;
    let k = 100;

    let mut index = IndexFlatL2::new(dim);

    let data = generate_vector_data(n, dim);
    index.add(&data);

    // let query = generate_vector_data(nq, dim);
    let query = &data[..nq * dim];
    let (ids, distances) = index.search(&query, k);

    println!("ids={:?}, distances={:?}", ids, distances);
}

#[test]
fn IndexIVF_test() {
    let n = 10000;
    let dim = 128;
    let nq = 10;
    let k = 100;

    let mut index = IndexIVFFlat::new(dim, 100);

    let data = generate_vector_data(n, dim);
    index.train(&data);
    index.add(&data);

    // let query = generate_vector_data(nq, dim);
    let query = &data[..nq * dim];
    let (ids, distances) = index.search(&query, k);

    println!("ids={:?}, distances={:?}", ids, distances);
}

fn generate_vector_data(num: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(num * dim);
    for _ in 0..num * dim {
        data.push(rng.gen::<f32>());
    }

    data
}
