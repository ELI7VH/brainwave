use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Response, Server};

// ── Config ─────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
struct Config {
    llama_bin: String,
    models_dir: String,
    port: u16,
    backend_port_start: u16,
    idle_timeout_secs: u64,
    vram_budget_mb: u32,
    models: HashMap<String, ModelConfig>,
    #[serde(default)]
    connectors: HashMap<String, ConnectorConfig>,
}

#[derive(Deserialize, Clone)]
struct ModelConfig {
    path: String,
    gpu_layers: u32,
    ctx_size: u32,
    flash_attn: bool,
    description: String,
    vram_mb: u32,
}

#[derive(Deserialize, Clone)]
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

// ── Backend state ──────────────────────────────────────────

struct LoadedModel {
    name: String,
    port: u16,
    process: Child,
    vram_mb: u32,
    last_used: Instant,
}

struct State {
    config: Config,
    loaded: Vec<LoadedModel>,
    next_port: u16,
    project_dir: PathBuf,
}

impl State {
    fn new(config: Config, project_dir: PathBuf) -> Self {
        let next_port = config.backend_port_start;
        State {
            config,
            loaded: Vec::new(),
            next_port,
            project_dir,
        }
    }

    fn vram_used(&self) -> u32 {
        self.loaded.iter().map(|m| m.vram_mb).sum()
    }

    fn find_loaded(&mut self, name: &str) -> Option<&mut LoadedModel> {
        self.loaded.iter_mut().find(|m| m.name == name)
    }

    fn allocate_port(&mut self) -> u16 {
        let port = self.next_port;
        self.next_port += 1;
        if self.next_port > self.config.backend_port_start + 99 {
            self.next_port = self.config.backend_port_start;
        }
        port
    }

    fn evict_for_vram(&mut self, needed_mb: u32) {
        let budget = self.config.vram_budget_mb;
        while self.vram_used() + needed_mb > budget && !self.loaded.is_empty() {
            // Evict least recently used
            let lru_idx = self
                .loaded
                .iter()
                .enumerate()
                .min_by_key(|(_, m)| m.last_used)
                .map(|(i, _)| i)
                .unwrap();

            let mut model = self.loaded.remove(lru_idx);
            eprintln!(
                "  [{}] evicting (LRU) to free {}MB VRAM",
                model.name.yellow(),
                model.vram_mb
            );
            let _ = model.process.kill();
            let _ = model.process.wait();
        }
    }

    fn load_model(&mut self, name: &str) -> Result<u16, String> {
        // Already loaded?
        if let Some(m) = self.find_loaded(name) {
            m.last_used = Instant::now();
            return Ok(m.port);
        }

        let model_cfg = self
            .config
            .models
            .get(name)
            .ok_or_else(|| format!("unknown model: {}", name))?
            .clone();

        // Evict if needed
        self.evict_for_vram(model_cfg.vram_mb);

        let port = self.allocate_port();
        let model_path = Path::new(&self.config.models_dir).join(&model_cfg.path);
        if !model_path.exists() {
            return Err(format!("model file not found: {}", model_path.display()));
        }

        let exe_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let exe = Path::new(&self.config.llama_bin).join(exe_name);

        let log_dir = self.project_dir.join("logs");
        fs::create_dir_all(&log_dir).ok();
        let log_file = fs::File::create(log_dir.join(format!("backend-{}.log", name)))
            .map_err(|e| format!("log file: {}", e))?;

        eprintln!(
            "  [{}] loading on :{} ({}MB VRAM)",
            name.cyan(),
            port,
            model_cfg.vram_mb
        );

        let mut args = vec![
            "-m".to_string(),
            model_path.to_string_lossy().to_string(),
            "--port".to_string(),
            port.to_string(),
            "-ngl".to_string(),
            model_cfg.gpu_layers.to_string(),
            "-c".to_string(),
            model_cfg.ctx_size.to_string(),
        ];
        if model_cfg.flash_attn {
            args.push("--flash-attn".to_string());
            args.push("on".to_string());
        }

        let child = Command::new(&exe)
            .args(&args)
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file))
            .spawn()
            .map_err(|e| format!("spawn failed: {}", e))?;

        eprintln!("  [{}] pid {} — waiting for health...", name.cyan(), child.id());

        self.loaded.push(LoadedModel {
            name: name.to_string(),
            port,
            process: child,
            vram_mb: model_cfg.vram_mb,
            last_used: Instant::now(),
        });

        // Wait for health (up to 60s for large models)
        for _ in 0..60 {
            thread::sleep(Duration::from_secs(1));
            if backend_health(port) {
                eprintln!("  [{}] ready on :{}", name.green(), port);
                return Ok(port);
            }
        }

        // Failed to start — clean up
        if let Some(idx) = self.loaded.iter().position(|m| m.name == name) {
            let mut m = self.loaded.remove(idx);
            let _ = m.process.kill();
            let _ = m.process.wait();
        }
        Err(format!("{}: backend didn't respond in 60s", name))
    }

    fn unload_idle(&mut self) {
        let timeout = Duration::from_secs(self.config.idle_timeout_secs);
        let now = Instant::now();

        let mut to_remove = Vec::new();
        for (i, m) in self.loaded.iter().enumerate() {
            if now.duration_since(m.last_used) > timeout {
                to_remove.push(i);
            }
        }

        for i in to_remove.into_iter().rev() {
            let mut model = self.loaded.remove(i);
            eprintln!(
                "  [{}] unloading (idle {}s)",
                model.name.yellow(),
                self.config.idle_timeout_secs
            );
            let _ = model.process.kill();
            let _ = model.process.wait();
        }
    }

    fn shutdown(&mut self) {
        for m in &mut self.loaded {
            eprintln!("  [{}] shutting down", m.name.yellow());
            let _ = m.process.kill();
            let _ = m.process.wait();
        }
        self.loaded.clear();
    }
}

fn backend_health(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);
    ureq::get(&url)
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok()
}

// ── Request types ──────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    #[serde(flatten)]
    rest: serde_json::Value,
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    port: u16,
    vram_used_mb: u32,
    vram_budget_mb: u32,
    loaded_models: Vec<LoadedModelInfo>,
    available_models: Vec<String>,
}

#[derive(Serialize)]
struct LoadedModelInfo {
    name: String,
    port: u16,
    vram_mb: u32,
    idle_secs: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ── Config loading ─────────────────────────────────────────

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

fn project_dir() -> PathBuf {
    config_path()
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

// ── HTTP handling ──────────────────────────────────────────

fn json_response<T: Serialize>(status: u16, body: &T) -> Response<std::io::Cursor<Vec<u8>>> {
    let json = serde_json::to_string(body).unwrap();
    let cursor = std::io::Cursor::new(json.into_bytes());
    Response::new(
        tiny_http::StatusCode(status),
        vec![
            Header::from_bytes("Content-Type", "application/json").unwrap(),
            Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap(),
        ],
        cursor,
        None,
        None,
    )
}

fn handle_status(state: &State) -> Response<std::io::Cursor<Vec<u8>>> {
    let now = Instant::now();
    let loaded: Vec<LoadedModelInfo> = state
        .loaded
        .iter()
        .map(|m| LoadedModelInfo {
            name: m.name.clone(),
            port: m.port,
            vram_mb: m.vram_mb,
            idle_secs: now.duration_since(m.last_used).as_secs(),
        })
        .collect();

    let available: Vec<String> = state.config.models.keys().cloned().collect();

    json_response(
        200,
        &StatusResponse {
            status: "running".to_string(),
            port: state.config.port,
            vram_used_mb: state.vram_used(),
            vram_budget_mb: state.config.vram_budget_mb,
            loaded_models: loaded,
            available_models: available,
        },
    )
}

fn handle_chat(
    state: &mut State,
    body: &[u8],
) -> Response<std::io::Cursor<Vec<u8>>> {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(400, &ErrorResponse {
                error: format!("invalid JSON: {}", e),
            });
        }
    };

    let model_name = match &req.model {
        Some(name) => name.clone(),
        None => {
            return json_response(400, &ErrorResponse {
                error: "model field is required".to_string(),
            });
        }
    };

    // Load model (or get existing port)
    let backend_port = match state.load_model(&model_name) {
        Ok(port) => port,
        Err(e) => {
            return json_response(503, &ErrorResponse {
                error: format!("failed to load {}: {}", model_name, e),
            });
        }
    };

    // Proxy to backend
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", backend_port);
    let body_str = String::from_utf8_lossy(body);

    match ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(120))
        .send_string(&body_str)
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_body = resp.into_string().unwrap_or_default();
            let cursor = std::io::Cursor::new(resp_body.into_bytes());
            Response::new(
                tiny_http::StatusCode(status),
                vec![
                    Header::from_bytes("Content-Type", "application/json").unwrap(),
                    Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap(),
                ],
                cursor,
                None,
                None,
            )
        }
        Err(e) => json_response(502, &ErrorResponse {
            error: format!("backend error: {}", e),
        }),
    }
}

fn handle_models(state: &State) -> Response<std::io::Cursor<Vec<u8>>> {
    // OpenAI-compatible /v1/models response
    let models: Vec<serde_json::Value> = state
        .config
        .models
        .iter()
        .map(|(name, cfg)| {
            let loaded = state.loaded.iter().any(|m| m.name == *name);
            serde_json::json!({
                "id": name,
                "object": "model",
                "owned_by": "brainwave",
                "description": cfg.description,
                "vram_mb": cfg.vram_mb,
                "loaded": loaded,
            })
        })
        .collect();

    let body = serde_json::json!({
        "object": "list",
        "data": models,
    });

    json_response(200, &body)
}

fn handle_unload(
    state: &mut State,
    body: &[u8],
) -> Response<std::io::Cursor<Vec<u8>>> {
    #[derive(Deserialize)]
    struct UnloadReq {
        model: String,
    }

    let req: UnloadReq = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => {
            return json_response(400, &ErrorResponse {
                error: "expected {\"model\": \"name\"}".to_string(),
            });
        }
    };

    if let Some(idx) = state.loaded.iter().position(|m| m.name == req.model) {
        let mut model = state.loaded.remove(idx);
        eprintln!("  [{}] unloading (manual)", model.name.yellow());
        let _ = model.process.kill();
        let _ = model.process.wait();
        json_response(200, &serde_json::json!({"unloaded": req.model}))
    } else {
        json_response(404, &ErrorResponse {
            error: format!("{} is not loaded", req.model),
        })
    }
}

// ── Main ───────────────────────────────────────────────────

fn main() {
    let path = config_path();
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("config not found at {}", path.display()));
    let config: Config = serde_json::from_str(&data).expect("invalid config.json");

    let listen = format!("0.0.0.0:{}", config.port);
    let server = Server::http(&listen).unwrap_or_else(|e| {
        eprintln!("  {} failed to bind {}: {}", "ERROR:".red().bold(), listen, e);
        std::process::exit(1);
    });

    eprintln!();
    eprintln!("  {}", "brainwave".bold().cyan());
    eprintln!("  listening on :{}", config.port);
    eprintln!("  models: {}", config.models.keys().cloned().collect::<Vec<_>>().join(", "));
    eprintln!("  VRAM budget: {} MB", config.vram_budget_mb);
    eprintln!("  idle timeout: {}s", config.idle_timeout_secs);
    eprintln!("  backends: :{}-:{}", config.backend_port_start, config.backend_port_start + 99);
    eprintln!();

    let state = Arc::new(Mutex::new(State::new(config.clone(), project_dir())));

    // Idle reaper thread
    let reaper_state = Arc::clone(&state);
    let idle_check_interval = Duration::from_secs(30);
    thread::spawn(move || loop {
        thread::sleep(idle_check_interval);
        if let Ok(mut s) = reaper_state.lock() {
            s.unload_idle();
        }
    });

    // Ctrl+C handler
    let shutdown_state = Arc::clone(&state);
    setup_shutdown(shutdown_state);

    for mut request in server.incoming_requests() {
        let path = request.url().to_string();
        let method = request.method().clone();

        // Read body
        let mut body = Vec::new();
        let _ = request.as_reader().read_to_end(&mut body);

        let response = {
            let mut s = state.lock().unwrap();

            match (method, path.as_str()) {
                // Status
                (Method::Get, "/status") | (Method::Get, "/") => handle_status(&s),

                // OpenAI-compatible endpoints
                (Method::Post, "/v1/chat/completions") => handle_chat(&mut s, &body),
                (Method::Get, "/v1/models") => handle_models(&s),

                // Brainwave-specific
                (Method::Post, "/unload") => handle_unload(&mut s, &body),

                // Shutdown
                (Method::Post, "/shutdown") => {
                    eprintln!("  shutdown requested");
                    s.shutdown();
                    let resp = json_response(200, &serde_json::json!({"status": "shutdown"}));
                    let _ = request.respond(resp);
                    std::process::exit(0);
                }

                // Health check
                (Method::Get, "/health") => {
                    json_response(200, &serde_json::json!({"status": "ok"}))
                }

                // CORS preflight
                (Method::Options, _) => {
                    let cursor = std::io::Cursor::new(Vec::new());
                    Response::new(
                        tiny_http::StatusCode(204),
                        vec![
                            Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap(),
                            Header::from_bytes("Access-Control-Allow-Methods", "GET, POST, OPTIONS").unwrap(),
                            Header::from_bytes("Access-Control-Allow-Headers", "Content-Type, Authorization").unwrap(),
                        ],
                        cursor,
                        None,
                        None,
                    )
                }

                _ => json_response(404, &ErrorResponse {
                    error: format!("not found: {}", request.url()),
                }),
            }
        };

        let _ = request.respond(response);
    }
}

fn setup_shutdown(_state: Arc<Mutex<State>>) {
    // Child processes are cleaned up when brainwave exits (Windows behavior).
    // The OS terminates child processes in the same job object.
    // For graceful shutdown, use POST /shutdown endpoint.
}
