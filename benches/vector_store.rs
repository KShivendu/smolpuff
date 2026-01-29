use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use rand::Rng;
use smolpuff::VectorStore;
use std::sync::Arc;

const VECTOR_DIM: usize = 128;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn generate_random_metadata() -> serde_json::Value {
    let mut rng = rand::thread_rng();
    let categories = ["A", "B", "C"];
    let idx: usize = rng.gen_range(0..3);
    serde_json::json!({
        "title": format!("Document {}", rng.r#gen::<u32>()),
        "category": categories[idx],
        "score": rng.r#gen::<f32>()
    })
}

async fn setup_store() -> VectorStore {
    let object_store: Arc<dyn ObjectStore + 'static> = Arc::new(InMemory::new());
    VectorStore::open("/bench/vectors", object_store)
        .await
        .expect("Failed to open store")
}

async fn setup_store_with_vectors(num_vectors: usize) -> VectorStore {
    let store = setup_store().await;

    for i in 0..num_vectors {
        let vector = generate_random_vector(VECTOR_DIM);
        let metadata = Some(generate_random_metadata());
        store
            .add(&format!("doc{}", i), vector, metadata)
            .await
            .expect("Failed to add vector");
    }

    store
}

fn bench_write_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("write_latency");

    for &dim in &[64, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::new("single_write", dim), &dim, |b, &dim| {
            b.iter_custom(|iters| {
                rt.block_on(async {
                    let store = setup_store().await;
                    let start = std::time::Instant::now();

                    for i in 0..iters {
                        let vector = generate_random_vector(dim);
                        let metadata = Some(generate_random_metadata());
                        store
                            .add(&format!("doc{}", i), black_box(vector), black_box(metadata))
                            .await
                            .expect("Failed to add vector");
                    }

                    let elapsed = start.elapsed();
                    let _ = store.close().await;
                    elapsed
                })
            });
        });
    }

    group.finish();
}

fn bench_write_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("write_throughput");

    for &batch_size in &[100, 500, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_write", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let mut total_elapsed = std::time::Duration::ZERO;

                        for _ in 0..iters {
                            let store = setup_store().await;
                            let start = std::time::Instant::now();

                            for i in 0..batch_size {
                                let vector = generate_random_vector(VECTOR_DIM);
                                let metadata = Some(generate_random_metadata());
                                store
                                    .add(
                                        &format!("doc{}", i),
                                        black_box(vector),
                                        black_box(metadata),
                                    )
                                    .await
                                    .expect("Failed to add vector");
                            }

                            total_elapsed += start.elapsed();
                            let _ = store.close().await;
                        }

                        total_elapsed
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_query_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("query_latency");

    for &num_vectors in &[100, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("knn_query", num_vectors),
            &num_vectors,
            |b, &num_vectors| {
                let store = rt.block_on(setup_store_with_vectors(num_vectors));
                let query_vector = generate_random_vector(VECTOR_DIM);

                b.to_async(&rt).iter(|| async {
                    store
                        .query(black_box(&query_vector), black_box(10))
                        .await
                        .expect("Failed to query")
                });

                rt.block_on(async {
                    let _ = store.close().await;
                });
            },
        );
    }

    group.finish();
}

fn bench_query_varying_k(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("query_varying_k");
    let num_vectors = 1000;

    for &k in &[1, 5, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &k| {
            let store = rt.block_on(setup_store_with_vectors(num_vectors));
            let query_vector = generate_random_vector(VECTOR_DIM);

            b.to_async(&rt).iter(|| async {
                store
                    .query(black_box(&query_vector), black_box(k))
                    .await
                    .expect("Failed to query")
            });

            rt.block_on(async {
                let _ = store.close().await;
            });
        });
    }

    group.finish();
}

fn bench_query_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("query_throughput");

    for &num_vectors in &[100, 1000, 5000] {
        group.throughput(Throughput::Elements(100)); // 100 queries per iteration
        group.bench_with_input(
            BenchmarkId::new("queries_per_sec", num_vectors),
            &num_vectors,
            |b, &num_vectors| {
                let store = rt.block_on(setup_store_with_vectors(num_vectors));

                b.to_async(&rt).iter(|| async {
                    for _ in 0..100 {
                        let query_vector = generate_random_vector(VECTOR_DIM);
                        store
                            .query(black_box(&query_vector), black_box(10))
                            .await
                            .expect("Failed to query");
                    }
                });

                rt.block_on(async {
                    let _ = store.close().await;
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_write_latency,
    bench_write_throughput,
    bench_query_latency,
    bench_query_varying_k,
    bench_query_throughput,
);

criterion_main!(benches);
