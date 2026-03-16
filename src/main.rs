use colored::Colorize;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

// ── Config (for offline commands) ──────────────────────────

#[derive(Deserialize)]
struct Config {
    port: u16,
    #[serde(default = "default_vram")]
    vram_budget_mb: u32,
    #[serde(default)]
    models: HashMap<String, ModelDef>,
    #[serde(default)]
    connectors: HashMap<String, ConnectorConfig>,
    // Legacy fields (ignored)
    #[serde(default)]
    llama_bin: String,
    #[serde(default)]
    models_dir: String,
}

fn default_vram() -> u32 { 14000 }

#[derive(Deserialize)]
#[allow(dead_code)]
struct ModelDef {
    path: String,
    description: String,
    vram_mb: u32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ConnectorConfig {
    #[serde(rename = "type")]
    conn_type: String,
    script: String,
    #[serde(default)]
    args: Vec<String>,
    description: String,
    shm_name: String,
    #[serde(default)]
    camera: u32,
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

// ── Daemon communication ───────────────────────────────────

#[derive(Deserialize)]
struct DaemonStatus {
    status: String,
    port: u16,
    vram_used_mb: u32,
    vram_budget_mb: u32,
    loaded_models: Vec<LoadedModelInfo>,
    available_models: Vec<String>,
}

#[derive(Deserialize)]
struct LoadedModelInfo {
    name: String,
    port: u16,
    vram_mb: u32,
    idle_secs: u64,
}

fn config_path() -> PathBuf {
    let exe = env::current_exe().unwrap_or_default();
    let dir = exe.parent().unwrap_or(Path::new("."));
    for ancestor in dir.ancestors() {
        let candidate = ancestor.join("config.json");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("config.json")
}

fn load_config() -> Config {
    let path = config_path();
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read config at {}", path.display()));
    serde_json::from_str(&data).expect("Failed to parse config.json")
}

fn project_dir() -> PathBuf {
    config_path()
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

fn daemon_url(config: &Config, path: &str) -> String {
    format!("http://127.0.0.1:{}{}", config.port, path)
}

fn daemon_running(config: &Config) -> bool {
    ureq::get(&daemon_url(config, "/health"))
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok()
}

// ── Connector helpers ──────────────────────────────────────

fn connector_pid_path(name: &str) -> PathBuf {
    project_dir().join("logs").join(format!("{}.pid", name))
}

fn connector_running(name: &str) -> bool {
    let pid_path = connector_pid_path(name);
    if let Ok(pid_str) = fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            if let Ok(output) = Command::new("tasklist")
                .args(["/FI", &format!("PID eq {}", pid), "/NH"])
                .output()
            {
                let text = String::from_utf8_lossy(&output.stdout);
                return text.contains(&pid.to_string());
            }
        }
    }
    false
}

fn start_connector(name: &str, conn: &ConnectorConfig) {
    if connector_running(name) {
        println!("  [{}] already running", name.cyan());
        return;
    }

    let script_path = project_dir().join(&conn.script);
    if !script_path.exists() {
        eprintln!("  {} script not found: {}", "ERROR:".red().bold(), script_path.display());
        return;
    }

    let log_dir = project_dir().join("logs");
    fs::create_dir_all(&log_dir).ok();
    let log_file = fs::File::create(log_dir.join(format!("{}.log", name)))
        .expect("Failed to create log file");

    println!("  [{}] starting — {}", name.cyan(), conn.description.dimmed());

    let mut cmd_args = vec![script_path.to_string_lossy().to_string()];
    cmd_args.extend(conn.args.clone());
    if conn.camera > 0 {
        cmd_args.push("--camera".to_string());
        cmd_args.push(conn.camera.to_string());
    }

    match Command::new("python")
        .args(&cmd_args)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
    {
        Ok(c) => {
            let pid = c.id();
            fs::write(connector_pid_path(name), pid.to_string()).ok();
            println!("    pid: {} — {}", pid, "launched".green());
        }
        Err(e) => eprintln!("  {} {}", "ERROR:".red().bold(), e),
    }
}

fn stop_connector(name: &str) {
    let pid_path = connector_pid_path(name);
    if let Ok(pid_str) = fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            println!("  [{}] stopping pid {}", name.cyan(), pid);
            let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/F"]).output();
            fs::remove_file(&pid_path).ok();
            println!("    {}", "stopped".green());
            return;
        }
    }
    println!("  [{}] not running", name.dimmed());
}

// ── Commands ───────────────────────────────────────────────

fn cmd_status(config: &Config) {
    println!();
    println!("  {}", "brainwave".bold().cyan());
    println!("  {}", "=".repeat(56));
    println!();

    // Daemon status
    if daemon_running(config) {
        match ureq::get(&daemon_url(config, "/status"))
            .timeout(Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                if let Ok(status) = resp.into_json::<DaemonStatus>() {
                    println!(
                        "  daemon     :{:<5}  {}    {}/{}MB VRAM",
                        status.port,
                        "RUNNING".green().bold(),
                        status.vram_used_mb,
                        status.vram_budget_mb,
                    );
                    println!();

                    if status.loaded_models.is_empty() {
                        println!("  {} — models load on demand", "no models loaded".dimmed());
                    } else {
                        println!("  {}", "Loaded Models".bold());
                        for m in &status.loaded_models {
                            println!(
                                "  {:<10} :{:<5}  {}MB  idle {}s",
                                m.name.cyan(),
                                m.port,
                                m.vram_mb,
                                m.idle_secs
                            );
                        }
                    }
                    println!();

                    println!("  {}", "Available".bold());
                    for name in &status.available_models {
                        let loaded = status.loaded_models.iter().any(|m| m.name == *name);
                        let cfg = config.models.get(name);
                        let desc = cfg.map(|c| c.description.as_str()).unwrap_or("");
                        let vram = cfg.map(|c| c.vram_mb).unwrap_or(0);
                        if loaded {
                            println!("  {:<10} {}MB  {} {}", name.green(), vram, desc.dimmed(), "[loaded]".green());
                        } else {
                            println!("  {:<10} {}MB  {}", name, vram, desc.dimmed());
                        }
                    }
                    println!();
                }
            }
            Err(e) => eprintln!("  {} {}", "ERROR:".red().bold(), e),
        }
    } else {
        println!("  daemon     :{:<5}  {}", config.port, "STOPPED".red());
        println!("  run {} to start", "brainwave".bold());
        println!();
    }

    // Connectors
    if !config.connectors.is_empty() {
        println!("  {}", "Connectors".bold());
        for (name, conn) in &config.connectors {
            let state = if connector_running(name) {
                "RUNNING".green().bold().to_string()
            } else {
                "STOPPED".red().to_string()
            };
            println!("  {:<10} {}  shm:{}", name.cyan(), state, conn.shm_name);
        }
        println!();
    }

    // GPU
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            println!("  {}", "GPU".bold());
            for line in text.trim().lines() {
                let p: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if p.len() == 4 {
                    println!("  {}  {}/{} MB  {}C", p[0], p[1], p[2], p[3]);
                }
            }
            println!();
        }
    }
}

fn cmd_up(config: &Config) {
    if daemon_running(config) {
        println!("  brainwave already running on :{}", config.port);
        return;
    }

    println!("  starting brainwave daemon on :{}...", config.port);

    let exe = env::current_exe().unwrap_or_default();
    let daemon_exe = exe.parent().unwrap_or(Path::new(".")).join(
        if cfg!(windows) { "brainwave.exe" } else { "brainwave" }
    );

    if !daemon_exe.exists() {
        eprintln!("  {} daemon binary not found at {}", "ERROR:".red().bold(), daemon_exe.display());
        eprintln!("  run: cargo build --release");
        return;
    }

    let log_dir = project_dir().join("logs");
    fs::create_dir_all(&log_dir).ok();
    let log_file = fs::File::create(log_dir.join("daemon.log"))
        .expect("Failed to create log file");

    match Command::new(&daemon_exe)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn()
    {
        Ok(c) => {
            println!("  pid: {} — waiting for health...", c.id());
            for _ in 0..10 {
                std::thread::sleep(Duration::from_secs(1));
                if daemon_running(config) {
                    println!("  {}", "brainwave is up".green().bold());
                    return;
                }
            }
            println!("  {} check logs/daemon.log", "WARNING: slow start".yellow());
        }
        Err(e) => eprintln!("  {} {}", "ERROR:".red().bold(), e),
    }
}

fn cmd_down(config: &Config) {
    if !daemon_running(config) {
        println!("  brainwave not running");
        return;
    }
    println!("  stopping brainwave...");
    match ureq::post(&daemon_url(config, "/shutdown"))
        .timeout(Duration::from_secs(5))
        .send_string("{}")
    {
        Ok(_) => println!("  {}", "brainwave stopped".green()),
        Err(_) => println!("  {}", "brainwave stopped".green()), // connection reset = it shut down
    }
}

fn cmd_start(config: &Config, target: Option<&str>) {
    println!();
    match target {
        Some(name) => {
            if config.connectors.contains_key(name) {
                start_connector(name, &config.connectors[name]);
            } else if config.models.contains_key(name) {
                // Pre-load a model by sending a lightweight request
                if !daemon_running(config) {
                    eprintln!("  {} daemon not running. Run {} first", "ERROR:".red().bold(), "bw up".bold());
                    return;
                }
                println!("  [{}] pre-loading...", name.cyan());
                let body = serde_json::json!({
                    "model": name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                });
                match ureq::post(&daemon_url(config, "/v1/chat/completions"))
                    .set("Content-Type", "application/json")
                    .timeout(Duration::from_secs(120))
                    .send_string(&body.to_string())
                {
                    Ok(_) => println!("  [{}] {}", name.cyan(), "loaded".green()),
                    Err(e) => eprintln!("  {} {}", "ERROR:".red().bold(), e),
                }
            } else {
                let all: Vec<&str> = config.models.keys().chain(config.connectors.keys())
                    .map(|s| s.as_str()).collect();
                eprintln!("  {} unknown: '{}'. Available: {}", "ERROR:".red().bold(), name, all.join(", "));
            }
        }
        None => {
            // Start all connectors (models load on demand)
            for (name, conn) in &config.connectors {
                start_connector(name, conn);
            }
        }
    }
    println!();
}

fn cmd_stop(config: &Config, target: Option<&str>) {
    println!();
    match target {
        Some(name) => {
            if config.connectors.contains_key(name) {
                stop_connector(name);
            } else if config.models.contains_key(name) {
                if !daemon_running(config) {
                    println!("  daemon not running");
                    return;
                }
                let body = serde_json::json!({"model": name});
                match ureq::post(&daemon_url(config, "/unload"))
                    .set("Content-Type", "application/json")
                    .send_string(&body.to_string())
                {
                    Ok(_) => println!("  [{}] {}", name.cyan(), "unloaded".green()),
                    Err(e) => eprintln!("  {} {}", "ERROR:".red().bold(), e),
                }
            } else {
                eprintln!("  {} unknown: '{}'", "ERROR:".red().bold(), name);
            }
        }
        None => {
            for (name, _) in &config.connectors {
                stop_connector(name);
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
        "Model".bold(), "Size".bold(), "Quant".bold(), "GLSL".bold(), "Notes".bold()
    );
    println!("  {}", "-".repeat(86));

    let mut models = inv.inventory;
    models.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb).unwrap());

    for m in &models {
        let glsl = if m.glsl_trained.unwrap_or(false) { "yes".green().to_string() } else { String::new() };
        let notes: String = m.notes.chars().take(40).collect();
        println!(
            "  {:<33} {:>5.1}G  {:<8}  {:>4}  {}",
            m.name, m.size_gb, m.quant, glsl, notes.dimmed()
        );
    }
    println!();
}

fn print_usage() {
    println!();
    println!("  {} — local ML inference daemon", "brainwave".bold().cyan());
    println!();
    println!("  Usage: {} [command] [target]", "bw".bold());
    println!();
    println!("  Daemon:");
    println!("    {}             start the daemon", "up".bold());
    println!("    {}           stop the daemon", "down".bold());
    println!("    {}         what's running + loaded models", "status".bold());
    println!();
    println!("  Models:");
    println!("    {} <name>   pre-load a model", "start".bold());
    println!("    {} <name>    unload a model", "stop".bold());
    println!("    {}         list model inventory", "models".bold());
    println!();
    println!("  Connectors:");
    println!("    {} vision  start vision connector", "start".bold());
    println!("    {} vision   stop vision connector", "stop".bold());
    println!();
    println!("  Models load automatically on first request.");
    println!("  Idle models unload after timeout (configurable).");
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");
    let target = args.get(2).map(|s| s.as_str());

    let config = load_config();

    match cmd {
        "status" | "s" => cmd_status(&config),
        "up" => cmd_up(&config),
        "down" => cmd_down(&config),
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
