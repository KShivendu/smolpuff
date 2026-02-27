use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use reqwest::Client;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "puffbench", about = "smolpuff benchmark")]
struct Args {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    url: String,

    /// Number of vectors to write
    #[arg(long, default_value_t = 1000)]
    vectors: usize,

    /// Number of queries to run
    #[arg(long, default_value_t = 200)]
    queries: usize,

    /// Vector dimension
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Query top_k
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Namespace name
    #[arg(short = 'n', long, default_value = "bench")]
    namespace: String,

    /// Number of concurrent requests
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
}

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

struct Stats {
    p50: Duration,
    p99: Duration,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p50={:.1}ms p99={:.1}ms",
            self.p50.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0,
        )
    }
}

fn compute_stats(durations: &mut [Duration]) -> Stats {
    durations.sort();
    let n = durations.len();
    Stats {
        p50: durations[n / 2],
        p99: durations[(n as f64 * 0.99) as usize],
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let base = &args.url;
    let ns = &args.namespace;

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
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

    // Create namespace, cleaning up any leftover from a previous run
    let resp = client
        .post(format!("{base}/v1/namespaces"))
        .json(&serde_json::json!({ "name": ns, "vector_dim": args.dim }))
        .send()
        .await
        .unwrap();
    if resp.status() == 409 {
        eprint!("Cleaning up old namespace... ");
        // Delete can be slow (scans all vectors), so use a longer timeout
        let cleanup_client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap();
        cleanup_client
            .delete(format!("{base}/v1/namespaces/{ns}"))
            .send()
            .await
            .unwrap();
        eprintln!("done");
        let resp = client
            .post(format!("{base}/v1/namespaces"))
            .json(&serde_json::json!({ "name": ns, "vector_dim": args.dim }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "Failed to create namespace after cleanup"
        );
    } else {
        assert_eq!(resp.status(), 200, "Failed to create namespace");
    }

    let style = ProgressStyle::with_template(
        "{prefix:>10} [{bar:30}] {pos}/{len} ({per_sec}, eta {eta}) {msg}",
    )
    .unwrap()
    .progress_chars("=> ");

    // Write phase
    let pb = ProgressBar::new(args.vectors as u64)
        .with_prefix("Writing")
        .with_style(style.clone());
    pb.enable_steady_tick(Duration::from_millis(100));
    let mut write_durations: Vec<Duration> = stream::iter(0..args.vectors)
        .map(|i| {
            let client = client.clone();
            let url = format!("{base}/v1/namespaces/{ns}/write");
            let vector = generate_random_vector(args.dim);
            let pb = pb.clone();
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
                pb.inc(1);
                start.elapsed()
            }
        })
        .buffer_unordered(args.concurrency)
        .collect()
        .await;
    let write_stats = compute_stats(&mut write_durations);
    pb.finish_with_message(format!("done ({write_stats})"));

    // Query phase
    let pb = ProgressBar::new(args.queries as u64)
        .with_prefix("Querying")
        .with_style(style);
    pb.enable_steady_tick(Duration::from_millis(100));
    let mut query_durations: Vec<Duration> = stream::iter(0..args.queries)
        .map(|_| {
            let client = client.clone();
            let url = format!("{base}/v1/namespaces/{ns}/query");
            let vector = generate_random_vector(args.dim);
            let top_k = args.top_k;
            let pb = pb.clone();
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
                pb.inc(1);
                start.elapsed()
            }
        })
        .buffer_unordered(args.concurrency)
        .collect()
        .await;
    let query_stats = compute_stats(&mut query_durations);
    pb.finish_with_message(format!("done ({query_stats})"));

    // Delete namespace
    client
        .delete(format!("{base}/v1/namespaces/{ns}"))
        .send()
        .await
        .unwrap();

    // Summary
    println!(
        "puffbench: wrote {} vectors, ran {} queries (dim={}, top_k={}, concurrency={})",
        args.vectors, args.queries, args.dim, args.top_k, args.concurrency
    );
    println!("Write: {write_stats}  Query: {query_stats}");
    println!("See Grafana at http://localhost:3001 for details");
}
