# brainwave

Local LLM inference stack. No Electron, no UI, just llama.cpp servers with config-driven model management.

## Stack

- **llama.cpp** server binaries at `D:\llama-cpp\`
- **GGUF models** at `D:\.lmstudio\models\`
- Config-driven server definitions in `config.json`
- Full model inventory in `models.json`

## Quick Start

```bash
./start.sh          # start all servers
./start.sh code     # start just the code server
./status.sh         # check what's running
./stop.sh           # stop all
```

## Servers

| Name | Port | Model | VRAM | Purpose |
|------|------|-------|------|---------|
| code | 1234 | CodeQwen 7B Q4 | ~5 GB | Code gen, GLSL, technical |
| naming | 1235 | Qwen2.5 0.5B Q8 | ~825 MB | Chat naming, classification |

All servers expose an OpenAI-compatible API at `http://localhost:{port}/v1/chat/completions`.

## Hardware

- RTX 5060 Ti (16 GB VRAM)
- Ryzen 7 5700X
- 32 GB RAM
