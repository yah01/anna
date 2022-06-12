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

    let query = generate_vector_data(nq, dim);
    let (ids, diss) = index.search(&query, k);

    println!("ids={:?}, diss={:?}", ids, diss);
}

fn generate_vector_data(num: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(num * dim);
    for i in 0..num {
        for d in 0..dim {
            data.push(rng.gen::<f32>());
        }
    }

    data
}
