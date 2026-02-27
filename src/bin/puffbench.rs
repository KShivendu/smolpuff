use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::signal;

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

    /// Query-only mode: skip create/write/delete, just query an existing namespace
    #[arg(long)]
    query_only: bool,

    /// Grafana URL for posting annotations
    #[arg(long, default_value = "http://localhost:3001")]
    grafana_url: String,

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

fn generate_run_id() -> String {
    let mut rng = rand::thread_rng();
    let id: u32 = rng.gen_range(0..0xFFFFFF);
    format!("{id:06x}")
}

async fn annotate(client: &Client, grafana_url: &str, text: &str, tags: &[&str]) {
    let _ = client
        .post(format!("{grafana_url}/api/annotations"))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "text": text,
            "tags": tags,
        }))
        .send()
        .await;
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
    let run_id = generate_run_id();

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

    // Set up Ctrl+C handler to annotate abort in Grafana
    let abort_client = client.clone();
    let abort_grafana = args.grafana_url.clone();
    let abort_run_id = run_id.clone();
    let abort_run_id = Arc::new(abort_run_id);
    let abort_run_id_clone = abort_run_id.clone();
    tokio::spawn(async move {
        signal::ctrl_c().await.ok();
        annotate(
            &abort_client,
            &abort_grafana,
            &format!("[{abort_run_id_clone}] Benchmark ABORTED"),
            &["puffbench", "abort"],
        )
        .await;
        std::process::exit(130);
    });

    // Annotate benchmark start in Grafana
    let grafana = &args.grafana_url;
    let start_text = if args.query_only {
        format!(
            "[{run_id}] Benchmark started: query-only, {} queries (ns={ns}, dim={}, top_k={}, concurrency={})",
            args.queries, args.dim, args.top_k, args.concurrency
        )
    } else {
        format!(
            "[{run_id}] Benchmark started: {} vectors, {} queries (ns={ns}, dim={}, top_k={}, concurrency={})",
            args.vectors, args.queries, args.dim, args.top_k, args.concurrency
        )
    };
    annotate(&client, grafana, &start_text, &["puffbench", "start"]).await;
    if args.query_only {
        eprintln!(
            "[{run_id}] query-only: {} queries (ns={ns}, dim={}, top_k={}, concurrency={})",
            args.queries, args.dim, args.top_k, args.concurrency
        );
    } else {
        eprintln!(
            "[{run_id}] {} vectors, {} queries (ns={ns}, dim={}, top_k={}, concurrency={})",
            args.vectors, args.queries, args.dim, args.top_k, args.concurrency
        );
    }

    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]);

    let bar_style = ProgressStyle::with_template(
        "{prefix:>10.bold} {bar:40.green/black} {pos:>6}/{len} | {per_sec:>12.cyan} | eta {eta:>3.yellow} {msg:.green.bold}",
    )
    .unwrap()
    .progress_chars("\u{2588}\u{2592}\u{2591}");

    // Write phase (skip in query-only mode)
    let mut write_stats = None;
    if !args.query_only {
        // Create namespace, cleaning up any leftover from a previous run
        let resp = client
            .post(format!("{base}/v1/namespaces"))
            .json(&serde_json::json!({ "name": ns, "vector_dim": args.dim }))
            .send()
            .await
            .unwrap();
        if resp.status() == 409 {
            let sp = ProgressBar::new_spinner().with_style(spinner_style.clone());
            sp.set_message("Cleaning up old namespace...");
            sp.enable_steady_tick(Duration::from_millis(80));
            let cleanup_client = Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap();
            cleanup_client
                .delete(format!("{base}/v1/namespaces/{ns}"))
                .send()
                .await
                .unwrap();
            sp.finish_with_message("Cleaned up old namespace");
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

        let pb = ProgressBar::new(args.vectors as u64)
            .with_prefix("Writing")
            .with_style(bar_style.clone());
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
        let ws = compute_stats(&mut write_durations);
        pb.finish_with_message(format!("done ({ws})"));
        write_stats = Some(ws);
    }

    // Query phase
    let pb = ProgressBar::new(args.queries as u64)
        .with_prefix("Querying")
        .with_style(bar_style);
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

    // Delete namespace (skip in query-only mode)
    if !args.query_only {
        let sp = ProgressBar::new_spinner().with_style(spinner_style);
        sp.set_message("Cleaning up...");
        sp.enable_steady_tick(Duration::from_millis(80));
        let cleanup_client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap();
        cleanup_client
            .delete(format!("{base}/v1/namespaces/{ns}"))
            .send()
            .await
            .unwrap();
        sp.finish_and_clear();
    }

    // Annotate benchmark end in Grafana
    let end_text = if let Some(ws) = &write_stats {
        format!("[{run_id}] Benchmark finished — Write: {ws} | Query: {query_stats}")
    } else {
        format!("[{run_id}] Benchmark finished — Query: {query_stats}")
    };
    annotate(&client, grafana, &end_text, &["puffbench", "end"]).await;

    // Summary
    if let Some(ws) = &write_stats {
        println!(
            "puffbench: wrote {} vectors, ran {} queries (dim={}, top_k={}, concurrency={})",
            args.vectors, args.queries, args.dim, args.top_k, args.concurrency
        );
        println!("Write: {ws}  Query: {query_stats}");
    } else {
        println!(
            "puffbench: ran {} queries on namespace \"{ns}\" (dim={}, top_k={}, concurrency={})",
            args.queries, args.dim, args.top_k, args.concurrency
        );
        println!("Query: {query_stats}");
    }
    println!("See Grafana at http://localhost:3001 for details");
}
