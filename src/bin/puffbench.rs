use clap::Parser;
use futures::stream::{self, StreamExt};
use rand::Rng;
use reqwest::Client;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "puffbench", about = "smolpuff API benchmark")]
struct Args {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    url: String,

    /// Workloads to run (comma-separated)
    /// Values: write-latency, write-throughput, query-latency, query-topk, all
    #[arg(long, value_delimiter = ',', default_value = "all")]
    workload: Vec<String>,

    /// Vector dimensions for write-latency
    #[arg(long = "dim", value_delimiter = ',', default_values_t = vec![64, 128, 256, 512])]
    dims: Vec<usize>,

    /// Write counts for write-throughput
    #[arg(long = "count", value_delimiter = ',', default_values_t = vec![100, 500, 1000])]
    counts: Vec<usize>,

    /// Store sizes for query-latency
    #[arg(long = "store-size", value_delimiter = ',', default_values_t = vec![100, 500, 1000])]
    store_sizes: Vec<usize>,

    /// Top-k values for query-topk
    #[arg(long = "top-k", value_delimiter = ',', default_values_t = vec![1, 5, 10, 50])]
    top_ks: Vec<usize>,

    /// Iterations per measurement point
    #[arg(long, default_value_t = 50)]
    iters: usize,

    /// Number of concurrent requests
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
}

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

async fn bench_write_latency(
    client: &Client,
    base: &str,
    dims: &[usize],
    iters: usize,
    concurrency: usize,
) {
    println!("\n=== Write Latency (per-upsert round-trip) ===");

    for &dim in dims {
        let ns = format!("bench_wl_{dim}");
        create_ns(client, base, &ns, dim).await;

        let mut durations: Vec<Duration> = stream::iter(0..iters)
            .map(|i| {
                let client = client.clone();
                let url = format!("{base}/v1/namespaces/{ns}/write");
                let vector = generate_random_vector(dim);
                async move {
                    let start = Instant::now();
                    client
                        .post(&url)
                        .json(&serde_json::json!({
                            "id": format!("v{i}"),
                            "vector": vector,
                        }))
                        .send()
                        .await
                        .unwrap();
                    start.elapsed()
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        let stats = compute_stats(&mut durations);
        println!("  dim={dim:<4}  {stats}");
        delete_ns(client, base, &ns).await;
    }
}

async fn bench_write_throughput(client: &Client, base: &str, counts: &[usize], concurrency: usize) {
    println!("\n=== Write Throughput (bulk upsert) ===");
    let dim = 128;

    for &count in counts {
        let ns = format!("bench_wt_{count}");
        create_ns(client, base, &ns, dim).await;

        let start = Instant::now();
        stream::iter(0..count)
            .map(|i| {
                let client = client.clone();
                let url = format!("{base}/v1/namespaces/{ns}/write");
                let vector = generate_random_vector(dim);
                async move {
                    client
                        .post(&url)
                        .json(&serde_json::json!({
                            "id": format!("v{i}"),
                            "vector": vector,
                        }))
                        .send()
                        .await
                        .unwrap();
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<()>>()
            .await;
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

async fn bench_query_latency(
    client: &Client,
    base: &str,
    store_sizes: &[usize],
    iters: usize,
    concurrency: usize,
) {
    println!("\n=== Query Latency (top_k=10) ===");
    let dim = 128;
    let top_k = 10;

    for &store_size in store_sizes {
        let ns = format!("bench_ql_{store_size}");
        populate(client, base, &ns, dim, store_size).await;

        let mut durations: Vec<Duration> = stream::iter(0..iters)
            .map(|_| {
                let client = client.clone();
                let url = format!("{base}/v1/namespaces/{ns}/query");
                let vector = generate_random_vector(dim);
                async move {
                    let start = Instant::now();
                    client
                        .post(&url)
                        .json(&serde_json::json!({
                            "vector": vector,
                            "top_k": top_k,
                        }))
                        .send()
                        .await
                        .unwrap();
                    start.elapsed()
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        let stats = compute_stats(&mut durations);
        println!("  n={store_size:<5}  {stats}");
        delete_ns(client, base, &ns).await;
    }
}

async fn bench_query_varying_k(
    client: &Client,
    base: &str,
    top_ks: &[usize],
    iters: usize,
    concurrency: usize,
) {
    println!("\n=== Query Varying top_k (store_size=500) ===");
    let dim = 128;
    let store_size = 500;

    let ns = "bench_qk";
    populate(client, base, ns, dim, store_size).await;

    for &top_k in top_ks {
        let mut durations: Vec<Duration> = stream::iter(0..iters)
            .map(|_| {
                let client = client.clone();
                let url = format!("{base}/v1/namespaces/{ns}/query");
                let vector = generate_random_vector(dim);
                async move {
                    let start = Instant::now();
                    client
                        .post(&url)
                        .json(&serde_json::json!({
                            "vector": vector,
                            "top_k": top_k,
                        }))
                        .send()
                        .await
                        .unwrap();
                    start.elapsed()
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        let stats = compute_stats(&mut durations);
        println!("  top_k={top_k:<3}  {stats}");
    }

    delete_ns(client, base, ns).await;
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let base = &args.url;

    let client = Client::new();
    match client.get(format!("{base}/")).send().await {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            eprintln!("Server at {base} returned status {}", resp.status());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Cannot reach server at {base}: {e}");
            std::process::exit(1);
        }
    }

    let run_all = args.workload.iter().any(|w| w == "all");
    let concurrency = args.concurrency;

    println!("smolpuff API benchmark");
    println!("Target: {base}  Concurrency: {concurrency}");

    if run_all || args.workload.iter().any(|w| w == "write-latency") {
        bench_write_latency(&client, base, &args.dims, args.iters, concurrency).await;
    }
    if run_all || args.workload.iter().any(|w| w == "write-throughput") {
        bench_write_throughput(&client, base, &args.counts, concurrency).await;
    }
    if run_all || args.workload.iter().any(|w| w == "query-latency") {
        bench_query_latency(&client, base, &args.store_sizes, args.iters, concurrency).await;
    }
    if run_all || args.workload.iter().any(|w| w == "query-topk") {
        bench_query_varying_k(&client, base, &args.top_ks, args.iters, concurrency).await;
    }

    println!("\nDone.");
}
