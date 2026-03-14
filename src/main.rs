use colored::Colorize;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Deserialize)]
struct Config {
    llama_bin: String,
    models_dir: String,
    servers: HashMap<String, ServerConfig>,
}

#[derive(Deserialize)]
struct ServerConfig {
    port: u16,
    model: String,
    gpu_layers: u32,
    ctx_size: u32,
    flash_attn: String,
    description: String,
    vram_mb: u32,
}

#[derive(Deserialize)]
struct ModelInventory {
    inventory: Vec<ModelEntry>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ModelEntry {
    name: String,
    quant: String,
    size_gb: f64,
    strengths: Vec<String>,
    glsl_trained: Option<bool>,
    notes: String,
}

#[derive(Deserialize)]
struct HealthResponse {
    status: Option<String>,
}

fn config_path() -> PathBuf {
    let exe = env::current_exe().unwrap_or_default();
    let dir = exe.parent().unwrap_or(Path::new("."));
    // Walk up from target/release or target/debug to find config.json
    for ancestor in dir.ancestors() {
        let candidate = ancestor.join("config.json");
        if candidate.exists() {
            return candidate;
        }
    }
    // Fallback: current directory
    PathBuf::from("config.json")
}

fn load_config() -> Config {
    let path = config_path();
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read config at {}", path.display()));
    serde_json::from_str(&data).expect("Failed to parse config.json")
}

fn project_dir() -> PathBuf {
    config_path().parent().unwrap_or(Path::new(".")).to_path_buf()
}

fn health(port: u16) -> bool {
    let url = format!("http://localhost:{}/health", port);
    match ureq::get(&url).timeout(Duration::from_secs(2)).call() {
        Ok(resp) => {
            if let Ok(h) = resp.into_json::<HealthResponse>() {
                h.status.as_deref() == Some("ok")
            } else {
                true
            }
        }
        Err(_) => false,
    }
}

fn start_server(name: &str, srv: &ServerConfig, config: &Config) {
    if health(srv.port) {
        println!("  [{}] already running on :{}", name.cyan(), srv.port);
        return;
    }

    let model_path = Path::new(&config.models_dir).join(&srv.model);
    if !model_path.exists() {
        eprintln!(
            "  {} model not found: {}",
            "ERROR:".red().bold(),
            model_path.display()
        );
        return;
    }

    let exe_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let exe = Path::new(&config.llama_bin).join(exe_name);
    if !exe.exists() {
        eprintln!(
            "  {} llama-server not found at {}",
            "ERROR:".red().bold(),
            exe.display()
        );
        return;
    }

    let log_dir = project_dir().join("logs");
    fs::create_dir_all(&log_dir).ok();
    let log_file = fs::File::create(log_dir.join(format!("{}.log", name)))
        .expect("Failed to create log file");

    println!(
        "  [{}] starting on :{} — {}",
        name.cyan(),
        srv.port,
        srv.description.dimmed()
    );
    println!(
        "    model: {}",
        srv.model.split('/').last().unwrap_or(&srv.model)
    );

    let child = Command::new(&exe)
        .arg("-m")
        .arg(&model_path)
        .arg("--port")
        .arg(srv.port.to_string())
        .arg("-ngl")
        .arg(srv.gpu_layers.to_string())
        .arg("-c")
        .arg(srv.ctx_size.to_string())
        .arg("-fa")
        .arg(&srv.flash_attn)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn();

    match child {
        Ok(c) => println!("    pid: {}", c.id()),
        Err(e) => {
            eprintln!("  {} failed to spawn: {}", "ERROR:".red().bold(), e);
            return;
        }
    }

    for _ in 0..30 {
        if health(srv.port) {
            println!("    status: {}", "ok".green());
            return;
        }
        thread::sleep(Duration::from_secs(1));
    }
    println!(
        "    {} didn't respond in 30s, check logs/{}.log",
        "WARNING:".yellow().bold(),
        name
    );
}

fn stop_server(name: &str, srv: &ServerConfig) {
    if health(srv.port) {
        println!("  [{}] stopping on :{}", name.cyan(), srv.port);
        let url = format!("http://localhost:{}/quit", srv.port);
        let _ = ureq::get(&url).timeout(Duration::from_secs(2)).call();
        println!("    {}", "stopped".green());
    } else {
        println!("  [{}] not running on :{}", name.dimmed(), srv.port);
    }
}

fn cmd_status(config: &Config) {
    println!();
    println!("  {}", "brainwave status".bold());
    println!("  {}", "=".repeat(56));
    println!();

    for (name, srv) in &config.servers {
        let running = health(srv.port);
        let state = if running {
            "RUNNING".green().bold().to_string()
        } else {
            "STOPPED".red().to_string()
        };
        let model_short = srv.model.split('/').last().unwrap_or(&srv.model);
        println!(
            "  {:<10} :{:<5}  {:<8}  {}MB  {}",
            name.cyan(),
            srv.port,
            state,
            srv.vram_mb,
            model_short
        );
        println!("  {:<10} {}", "", srv.description.dimmed());
        println!();
    }

    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            println!("  {}", "GPU".bold());
            println!("  ---");
            for line in text.trim().lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() == 4 {
                    println!(
                        "  {}  {}/{} MB  {}C",
                        parts[0], parts[1], parts[2], parts[3]
                    );
                }
            }
            println!();
        }
    }
}

fn cmd_start(config: &Config, target: Option<&str>) {
    println!();
    match target {
        Some(name) => {
            if let Some(srv) = config.servers.get(name) {
                start_server(name, srv, config);
            } else {
                eprintln!(
                    "  {} unknown server '{}'. Available: {}",
                    "ERROR:".red().bold(),
                    name,
                    config
                        .servers
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        None => {
            for (name, srv) in &config.servers {
                start_server(name, srv, config);
            }
        }
    }
    println!();
}

fn cmd_stop(config: &Config, target: Option<&str>) {
    println!();
    match target {
        Some(name) => {
            if let Some(srv) = config.servers.get(name) {
                stop_server(name, srv);
            } else {
                eprintln!("  {} unknown server '{}'", "ERROR:".red().bold(), name);
            }
        }
        None => {
            for (name, srv) in &config.servers {
                stop_server(name, srv);
            }
        }
    }
    println!();
}

fn cmd_models() {
    let path = project_dir().join("models.json");
    let data = match fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("  {} models.json not found", "ERROR:".red().bold());
            return;
        }
    };
    let inv: ModelInventory = serde_json::from_str(&data).expect("Failed to parse models.json");

    println!();
    println!(
        "  {:<33} {:>6}  {:<8}  {:>4}  {}",
        "Model".bold(),
        "Size".bold(),
        "Quant".bold(),
        "GLSL".bold(),
        "Notes".bold()
    );
    println!("  {}", "-".repeat(86));

    let mut models = inv.inventory;
    models.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb).unwrap());

    for m in &models {
        let glsl = if m.glsl_trained.unwrap_or(false) {
            "yes".green().to_string()
        } else {
            String::new()
        };
        let notes: String = m.notes.chars().take(40).collect();
        println!(
            "  {:<33} {:>5.1}G  {:<8}  {:>4}  {}",
            m.name, m.size_gb, m.quant, glsl,
            notes.dimmed()
        );
    }
    println!();
}

fn print_usage() {
    println!();
    println!(
        "  {} — local LLM inference stack",
        "brainwave".bold().cyan()
    );
    println!();
    println!("  Usage: {} [command] [server]", "bw".bold());
    println!();
    println!("  Commands:");
    println!("    {}         check what's running", "status".bold());
    println!("    {} [name]  start server(s)", "start".bold());
    println!("    {} [name]   stop server(s)", "stop".bold());
    println!("    {}        list model inventory", "models".bold());
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");
    let target = args.get(2).map(|s| s.as_str());

    let config = load_config();

    match cmd {
        "status" | "s" => cmd_status(&config),
        "start" => cmd_start(&config, target),
        "stop" => cmd_stop(&config, target),
        "models" | "m" => cmd_models(),
        "help" | "-h" | "--help" => print_usage(),
        _ => {
            eprintln!("  Unknown command: {}", cmd);
            print_usage();
        }
    }
}
