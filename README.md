# brainwave

Local ML inference hub. Manages LLM servers, vision connectors, and any future ML pipeline — all from a single 2.5MB Rust binary.

![brainwave](hero.png)

## What it does

brainwave is the single entry point for all ML inference on this machine. Any system that needs ML capabilities (LLM completions, computer vision, audio classification, embeddings) talks to brainwave. No Electron, no UI frameworks, no bloat.

```
                        ┌──────────────────────────────┐
                        │         brainwave            │
                        │  config-driven process mgr   │
                        └──────┬───────────┬───────────┘
                               │           │
                 ┌─────────────┤           ├─────────────┐
                 ▼             ▼           ▼             ▼
          ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ llama.cpp  │ │ llama.cpp│ │ vision   │ │ future   │
          │ :1234 code │ │ :1235    │ │ mediapipe│ │ connector│
          └─────┬──────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
                │             │            │             │
         OpenAI API    OpenAI API    shared memory   your protocol
                │             │            │             │
          ┌─────┘      ┌─────┘      ┌─────┘       ┌─────┘
          ▼             ▼            ▼              ▼
       wavehaze     chat-daddy   wavehaze       anything
       shader gen   auto-naming  GL uniforms
```

## Quick Start

```bash
bw status          # what's running
bw start           # start all servers + connectors
bw start code      # start just the code LLM
bw start vision    # start vision connector
bw stop            # stop everything
bw models          # list model inventory
```

## Connector Protocol

Any system can request ML capabilities from brainwave. Two protocols depending on latency requirements.

### LLM Inference (OpenAI-compatible HTTP)

Every LLM server exposes the standard OpenAI chat completions API:

```
POST http://<host>:{port}/v1/chat/completions
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

Servers are defined in `config.json` under `servers`. Each has:
- `port` — HTTP port
- `model` — path to GGUF file (relative to `models_dir`)
- `gpu_layers` — layers to offload to GPU
- `ctx_size` — context window in tokens

### Vision / Sensor Data (Shared Memory)

For real-time data (30+ Hz), brainwave uses **named shared memory** — no HTTP overhead, zero-copy reads.

**Protocol:**
1. Connector creates a named shared memory region
2. Writes a magic header + structured data at defined offsets
3. Consumer polls the `sequence` field to detect updates
4. On shutdown, connector zeros the magic → consumer detects disconnect

**Header spec for any connector:**

```c
struct ConnectorHeader {
    uint32_t magic;            // unique 4-byte ID per connector type
    uint32_t version;          // protocol version
    uint32_t features;         // bitmask: what data sections are populated
    uint32_t flags;            // reserved
    volatile int64_t sequence; // increment every update (~30 Hz)
    uint64_t timestamp_us;     // microseconds since start
    char source_name[32];      // connector identity string
    char source_version[16];   // version string
};
// Total header: 80 bytes. Pad to 256 for future fields.
// Data sections follow at fixed offsets declared per connector.
```

**Rules:**
- Magic must be unique per connector type (e.g., `0x56495332` for vision)
- Sequence must be volatile / atomically incremented
- Consumer detects connector by polling for the named shared memory region
- Consumer detects disconnect when magic is zeroed
- All coordinates normalized 0..1 unless otherwise specified
- Region size declared and fixed per connector type

**Current connectors:**

| Name | SHM Name | Magic | Data | Consumer |
|------|----------|-------|------|----------|
| vision | `WaveHaze_Vision` | `VIS2` | hand/pose landmarks, segmentation mask (512 KB) | wavehaze |

### Adding a New Connector

1. Create `connectors/<name>/` with your script or binary
2. Add entry to `config.json` under `connectors`:
   ```json
   {
     "type": "python",
     "script": "connectors/<name>/your_script.py",
     "args": ["--flag"],
     "description": "What it does",
     "shm_name": "YourShmName",
     "camera": 0
   }
   ```
3. `bw start <name>` / `bw stop <name>` — brainwave manages the lifecycle

### Connector Roadmap

| Connector | Protocol | Purpose |
|-----------|----------|---------|
| audio-classify | SHM | Genre/mood/instrument detection from audio stream |
| embeddings | HTTP | Text/image embeddings for similarity search |
| tts | HTTP | Text-to-speech via local model |
| whisper | HTTP | Speech-to-text via whisper.cpp |
| img-gen | HTTP | Image generation via stable-diffusion.cpp |

## Model Inventory

`bw models` lists all available GGUF files with size, quantization, and capability tags.

## Hardware

- RTX 5060 Ti (16 GB VRAM)
- Ryzen 7 5700X
- 32 GB RAM

## Build

```bash
cargo build --release
# Binary: target/release/bw.exe (2.5 MB)
```
