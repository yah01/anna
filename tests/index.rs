use anna::{Index, IndexFlatL2};
use rand::Rng;

#[test]
fn IndexFlatL2_test() {
    let n = 10000;
    let dim = 128;
    let nq = 1;
    let k = 5;

    let mut index = IndexFlatL2::new(dim);

    let data = generate_vector_data(n, dim);
    index.add(&data);

    // let query = generate_vector_data(nq, dim);
    let query = &data[..dim];
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
