---
inclusion: always
---

# FactoryPulse Project Build Instructions

## GitHub Repository
https://github.com/varunsivasamy/factorypulse

## Commit Convention
```
[M1-FactoryPulse] <type>: <description>
```
Types: `feat` | `config` | `docs` | `data` | `test` | `fix`

## Phase 1 — Milestone 1: Architecture & Project Setup
**11 commits total → tagged as `v0.1.0-M1`**

| Step | Deliverable | Commit Type |
|------|-------------|-------------|
| 1.1 | Repository init + .gitignore | config |
| 1.2 | Complete folder structure scaffold | feat |
| 1.3 | README, architecture docs, ADRs | docs |
| 1.4 | Environment config (.env, MQTT) | config |
| 1.5 | Database schema (TimescaleDB) | data |
| 1.6 | Docker Compose (5 infra containers) | config |
| 1.7 | Service Dockerfiles + requirements | config |
| 1.8 | ML package setup (pyproject.toml) | config |
| 1.9 | Makefile (all targets) | config |
| 1.10 | DVC init + dataset download script | data |
| 1.11 | Seed script (demo data) | data |

## .gitignore Categories
- Python: `__pycache__/` `*.py[cod]` `*.egg-info/` `dist/` `.venv/`
- Data (DVC): `data/raw/` `data/processed/` `*.h5` `*.pt` `*.onnx` `*.pkl`
- Environment: `.env` `*.log`
- IDE/OS: `.vscode/` `.idea/` `.DS_Store` `Thumbs.db`
- Docker: `docker-compose.override.yml`

## Rules
- Always use the exact commit message format: `[M1-FactoryPulse] <type>: <description>`
- User will provide content for each commit — use that content exactly
- After all 11 commits, tag as `v0.1.0-M1`
- Push to https://github.com/varunsivasamy/factorypulse
