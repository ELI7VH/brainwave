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
    #[serde(default)]
    connectors: HashMap<String, ConnectorConfig>,
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

#[derive(Deserialize)]
struct HealthResponse {
    status: Option<String>,
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

// ── Connector helpers ──────────────────────────────────────

fn connector_pid_path(name: &str) -> PathBuf {
    project_dir().join("logs").join(format!("{}.pid", name))
}

fn connector_running(name: &str) -> bool {
    let pid_path = connector_pid_path(name);
    if let Ok(pid_str) = fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            // Check if process exists (Windows: tasklist)
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
        eprintln!(
            "  {} script not found: {}",
            "ERROR:".red().bold(),
            script_path.display()
        );
        return;
    }

    let log_dir = project_dir().join("logs");
    fs::create_dir_all(&log_dir).ok();
    let log_file = fs::File::create(log_dir.join(format!("{}.log", name)))
        .expect("Failed to create log file");

    println!(
        "  [{}] starting — {}",
        name.cyan(),
        conn.description.dimmed()
    );
    println!("    shm: {}", conn.shm_name);

    let mut cmd_args = vec![script_path.to_string_lossy().to_string()];
    cmd_args.extend(conn.args.clone());
    if conn.camera > 0 {
        cmd_args.push("--camera".to_string());
        cmd_args.push(conn.camera.to_string());
    }

    let child = Command::new("python")
        .args(&cmd_args)
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn();

    match child {
        Ok(c) => {
            let pid = c.id();
            println!("    pid: {}", pid);
            fs::write(connector_pid_path(name), pid.to_string()).ok();
            println!("    status: {}", "launched".green());
        }
        Err(e) => {
            eprintln!("  {} failed to spawn: {}", "ERROR:".red().bold(), e);
        }
    }
}

fn stop_connector(name: &str) {
    let pid_path = connector_pid_path(name);
    if let Ok(pid_str) = fs::read_to_string(&pid_path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            println!("  [{}] stopping pid {}", name.cyan(), pid);
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"])
                .output();
            fs::remove_file(&pid_path).ok();
            println!("    {}", "stopped".green());
            return;
        }
    }
    println!("  [{}] not running", name.dimmed());
}

// ── Server helpers ─────────────────────────────────────────

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

// ── Commands ───────────────────────────────────────────────

fn cmd_status(config: &Config) {
    println!();
    println!("  {}", "brainwave status".bold());
    println!("  {}", "=".repeat(56));

    // LLM servers
    println!();
    println!("  {}", "LLM Servers".bold());
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

    // Connectors
    if !config.connectors.is_empty() {
        println!("  {}", "Connectors".bold());
        for (name, conn) in &config.connectors {
            let running = connector_running(name);
            let state = if running {
                "RUNNING".green().bold().to_string()
            } else {
                "STOPPED".red().to_string()
            };
            println!(
                "  {:<10} {:<6} {:<8}  shm:{}",
                name.cyan(),
                "",
                state,
                conn.shm_name
            );
            println!("  {:<10} {}", "", conn.description.dimmed());
            println!();
        }
    }

    // GPU
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
            } else if let Some(conn) = config.connectors.get(name) {
                start_connector(name, conn);
            } else {
                let all_names: Vec<&str> = config
                    .servers
                    .keys()
                    .chain(config.connectors.keys())
                    .map(|s| s.as_str())
                    .collect();
                eprintln!(
                    "  {} unknown: '{}'. Available: {}",
                    "ERROR:".red().bold(),
                    name,
                    all_names.join(", ")
                );
            }
        }
        None => {
            for (name, srv) in &config.servers {
                start_server(name, srv, config);
            }
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
            if let Some(srv) = config.servers.get(name) {
                stop_server(name, srv);
            } else if config.connectors.contains_key(name) {
                stop_connector(name);
            } else {
                eprintln!("  {} unknown: '{}'", "ERROR:".red().bold(), name);
            }
        }
        None => {
            for (name, srv) in &config.servers {
                stop_server(name, srv);
            }
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
        "  {} — local inference + ML connector stack",
        "brainwave".bold().cyan()
    );
    println!();
    println!("  Usage: {} [command] [target]", "bw".bold());
    println!();
    println!("  Commands:");
    println!("    {}         check what's running", "status".bold());
    println!(
        "    {} [name]  start server/connector (all if omitted)",
        "start".bold()
    );
    println!(
        "    {} [name]   stop server/connector (all if omitted)",
        "stop".bold()
    );
    println!("    {}        list model inventory", "models".bold());
    println!();
    println!("  Targets: servers (code, naming) or connectors (vision)");
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
