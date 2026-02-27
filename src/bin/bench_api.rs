use rand::Rng;
use reqwest::Client;
use std::time::{Duration, Instant};

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

async fn create_ns(client: &Client, base: &str, ns: &str, dim: usize) {
    let resp = client
        .post(format!("{base}/v1/namespaces"))
        .json(&serde_json::json!({ "name": ns, "vector_dim": dim }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "Failed to create namespace {ns}");
}

async fn delete_ns(client: &Client, base: &str, ns: &str) {
    client
        .delete(format!("{base}/v1/namespaces/{ns}"))
        .send()
        .await
        .unwrap();
}

async fn populate(client: &Client, base: &str, ns: &str, dim: usize, count: usize) {
    create_ns(client, base, ns, dim).await;
    for i in 0..count {
        let vector = generate_random_vector(dim);
        client
            .post(format!("{base}/v1/namespaces/{ns}/write"))
            .json(&serde_json::json!({
                "id": format!("v{i}"),
                "vector": vector,
            }))
            .send()
            .await
            .unwrap();
    }
}

struct Stats {
    min: Duration,
    avg: Duration,
    p50: Duration,
    p99: Duration,
    max: Duration,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "min={:.2}ms  avg={:.2}ms  p50={:.2}ms  p99={:.2}ms  max={:.2}ms",
            self.min.as_secs_f64() * 1000.0,
            self.avg.as_secs_f64() * 1000.0,
            self.p50.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0,
            self.max.as_secs_f64() * 1000.0,
        )
    }
}

fn compute_stats(durations: &mut [Duration]) -> Stats {
    durations.sort();
    let n = durations.len();
    let total: Duration = durations.iter().sum();
    Stats {
        min: durations[0],
        avg: total / n as u32,
        p50: durations[n / 2],
        p99: durations[(n as f64 * 0.99) as usize],
        max: durations[n - 1],
    }
}

async fn bench_write_latency(client: &Client, base: &str) {
    println!("\n=== Write Latency (per-upsert round-trip) ===");
    let writes_per_dim = 50;

    for &dim in &[64, 128, 256, 512] {
        let ns = format!("bench_wl_{dim}");
        create_ns(client, base, &ns, dim).await;

        let mut durations = Vec::with_capacity(writes_per_dim);
        for i in 0..writes_per_dim {
            let vector = generate_random_vector(dim);
            let start = Instant::now();
            client
                .post(format!("{base}/v1/namespaces/{ns}/write"))
                .json(&serde_json::json!({
                    "id": format!("v{i}"),
                    "vector": vector,
                }))
                .send()
                .await
                .unwrap();
            durations.push(start.elapsed());
        }

        let stats = compute_stats(&mut durations);
        println!("  dim={dim:<4}  {stats}");
        delete_ns(client, base, &ns).await;
    }
}

async fn bench_write_throughput(client: &Client, base: &str) {
    println!("\n=== Write Throughput (bulk upsert) ===");
    let dim = 128;

    for &count in &[100, 500, 1000] {
        let ns = format!("bench_wt_{count}");
        create_ns(client, base, &ns, dim).await;

        let start = Instant::now();
        for i in 0..count {
            let vector = generate_random_vector(dim);
            client
                .post(format!("{base}/v1/namespaces/{ns}/write"))
                .json(&serde_json::json!({
                    "id": format!("v{i}"),
                    "vector": vector,
                }))
                .send()
                .await
                .unwrap();
        }
        let elapsed = start.elapsed();
        let ops_sec = count as f64 / elapsed.as_secs_f64();

        println!(
            "  count={count:<5}  total={:.2}ms  ops/sec={:.0}",
            elapsed.as_secs_f64() * 1000.0,
            ops_sec,
        );
        delete_ns(client, base, &ns).await;
    }
}

async fn bench_query_latency(client: &Client, base: &str) {
    println!("\n=== Query Latency (top_k=10) ===");
    let dim = 128;
    let top_k = 10;
    let queries = 50;

    for &store_size in &[100, 500, 1000] {
        let ns = format!("bench_ql_{store_size}");
        populate(client, base, &ns, dim, store_size).await;

        let mut durations = Vec::with_capacity(queries);
        for _ in 0..queries {
            let vector = generate_random_vector(dim);
            let start = Instant::now();
            client
                .post(format!("{base}/v1/namespaces/{ns}/query"))
                .json(&serde_json::json!({
                    "vector": vector,
                    "top_k": top_k,
                }))
                .send()
                .await
                .unwrap();
            durations.push(start.elapsed());
        }

        let stats = compute_stats(&mut durations);
        println!("  n={store_size:<5}  {stats}");
        delete_ns(client, base, &ns).await;
    }
}

async fn bench_query_varying_k(client: &Client, base: &str) {
    println!("\n=== Query Varying top_k (store_size=500) ===");
    let dim = 128;
    let store_size = 500;
    let queries = 50;

    let ns = "bench_qk";
    populate(client, base, ns, dim, store_size).await;

    for &top_k in &[1, 5, 10, 50] {
        let mut durations = Vec::with_capacity(queries);
        for _ in 0..queries {
            let vector = generate_random_vector(dim);
            let start = Instant::now();
            client
                .post(format!("{base}/v1/namespaces/{ns}/query"))
                .json(&serde_json::json!({
                    "vector": vector,
                    "top_k": top_k,
                }))
                .send()
                .await
                .unwrap();
            durations.push(start.elapsed());
        }

        let stats = compute_stats(&mut durations);
        println!("  top_k={top_k:<3}  {stats}");
    }

    delete_ns(client, base, ns).await;
}

#[tokio::main]
async fn main() {
    let base = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:3000".to_string());

    // Verify server is reachable
    let client = Client::new();
    match client.get(format!("{base}/")).send().await {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            eprintln!("Server at {base} returned status {}", resp.status());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Cannot reach server at {base}: {e}");
            eprintln!("Usage: bench_api [BASE_URL]  (default: http://127.0.0.1:3000)");
            std::process::exit(1);
        }
    }

    println!("smolpuff API benchmark");
    println!("Target: {base}");

    bench_write_latency(&client, &base).await;
    bench_write_throughput(&client, &base).await;
    bench_query_latency(&client, &base).await;
    bench_query_varying_k(&client, &base).await;

    println!("\nDone.");
}
