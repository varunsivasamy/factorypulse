# FactoryPulse 🏭

### Real-Time Predictive Maintenance & Equipment Intelligence Platform

FactoryPulse is an end-to-end **Industrial AI / Industry 4.0 platform** for monitoring industrial equipment, detecting developing faults, predicting **Remaining Useful Life (RUL)**, and assisting maintenance engineers through an **agentic GenAI copilot**.

The platform combines **real-time sensor streaming, time-series analytics, deep learning, predictive maintenance, computer vision, RAG, and agentic AI** into a production-oriented microservice architecture.

> **Goal:** Move industrial maintenance from *"repair after failure"* to *"predict, diagnose, and act before failure."*

---

## 🚀 Key Capabilities

* 📡 Real-time multi-channel machine telemetry ingestion
* 🔍 Near-real-time anomaly detection
* 🧠 LSTM Autoencoder + Isolation Forest anomaly detection
* 📉 Remaining Useful Life (RUL) prediction using Transformer models
* 🏷️ Failure-mode classification using LightGBM
* 🌡️ Infrared thermography-based hotspot detection
* ❤️ Machine health scoring from 0–100
* 🚨 Intelligent severity-based alerting
* 📚 RAG-based maintenance knowledge system
* 🤖 LangGraph-powered maintenance copilot
* 📝 AI-generated maintenance work-order drafts
* 🔎 Hybrid vector + keyword document retrieval
* 📊 Fleet and machine monitoring dashboard
* 🧪 ML experiment tracking with MLflow
* 📦 Dataset and model versioning with DVC
* 📈 Model/data drift monitoring
* 🔐 JWT authentication with role-based access
* 📝 Complete audit logging
* 🐳 Dockerized microservice architecture
* 📈 Prometheus-based monitoring
* 🔄 CI/CD-ready architecture

---

# 🏭 Problem

Traditional industrial maintenance generally follows one of two approaches:

### Reactive Maintenance

> Machine fails → technician investigates → machine is repaired.

Problems:

* Unexpected downtime
* Expensive emergency repairs
* Safety risks
* Production losses
* Difficult root-cause analysis

### Preventive Maintenance

> Replace/service equipment after a fixed number of hours or cycles.

Problems:

* Healthy components may be replaced unnecessarily
* Developing faults can occur between maintenance intervals
* Maintenance resources are wasted
* Fixed schedules do not represent actual machine health

FactoryPulse introduces a third approach:

### Predictive + Intelligent Maintenance

```text
Sensor Data
    ↓
Machine Health Monitoring
    ↓
Anomaly Detection
    ↓
RUL Prediction
    ↓
Failure Diagnosis
    ↓
Knowledge Retrieval
    ↓
AI Maintenance Copilot
    ↓
Actionable Work Order
```

---

# 🎯 Objectives

FactoryPulse is designed around the following measurable objectives:

| Objective                            |                               Target |
| ------------------------------------ | -----------------------------------: |
| Early fault detection                | ≥ 48 hours before functional failure |
| C-MAPSS FD001 RUL RMSE               |                          ≤ 20 cycles |
| False-alarm reduction                |          ≥ 60% vs threshold baseline |
| Sensor → alert latency               |                          ≤ 5 seconds |
| Inference p95 latency                |                             < 150 ms |
| Copilot first-token latency          |                          < 3 seconds |
| Copilot grounded claims              |                                ≥ 90% |
| Simulated maintenance-cost reduction |                                ≥ 25% |
| Load-test throughput                 |                    200+ messages/sec |

> Actual benchmark results will be reported here as models and datasets are evaluated.

---

# 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │  Machine Simulator   │
                         │ MQTT / HTTP Telemetry│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Ingestion Service   │
                         │ Validation / DLQ    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     TimescaleDB     │
                         │   Time-Series Data  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                ┌─────────────────┐    ┌─────────────────┐
                │ Feature Pipeline│    │ Thermal Pipeline│
                │ Rolling Stats   │    │ OpenCV          │
                │ FFT / Kurtosis  │    │ Hotspot Detection│
                └────────┬────────┘    └────────┬────────┘
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Inference Service  │
                         │                     │
                         │ Anomaly Detection   │
                         │ RUL Prediction      │
                         │ Failure Classification│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Alert Engine     │
                         │ Severity / Hysteresis│
                         │ Deduplication       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                ┌─────────────────┐    ┌─────────────────┐
                │ Streamlit       │    │ Webhook / Slack │
                │ Operations UI   │    │ Notifications   │
                └─────────────────┘    └─────────────────┘


       Maintenance Manuals
       Fault-Code Documents
       Maintenance Logs
                 │
                 ▼
       ┌─────────────────────┐
       │ Document Processing │
       │ Chunking + Embedding│
       └──────────┬──────────┘
                  ▼
       ┌─────────────────────┐
       │      Qdrant         │
       │ Hybrid Retrieval    │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ LangGraph Copilot   │
       │                     │
       │ Machine Health      │
       │ Recent Anomalies    │
       │ RUL                  │
       │ Maintenance History │
       │ Manual Search       │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Work Order Draft    │
       │ + Citations         │
       └─────────────────────┘


 ML Training Pipeline

 Datasets → DVC → Feature Engineering
                 ↓
        Model Training
                 ↓
              MLflow
                 ↓
        Model Registry
                 ↓
       Staging → Production
```

---

# 🧩 Core Components

## 1. Sensor Data Simulator

The simulator reproduces real-world industrial telemetry without requiring physical factory equipment.

It supports:

* Multiple machines
* Configurable sampling frequency
* Multiple sensor channels
* Machine-specific behavior
* Gradual degradation
* Fault injection
* Network reconnection
* MQTT/HTTP transmission

Example telemetry:

```json
{
  "machine_id": "MACHINE_001",
  "timestamp": "2026-08-27T10:30:21Z",
  "vibration": 4.82,
  "temperature": 76.4,
  "pressure": 102.3,
  "rpm": 1842,
  "current": 8.71
}
```

---

# 📡 2. Streaming Ingestion

The ingestion service receives machine telemetry and performs validation before storing it.

### Validation

* Payload schema validation
* Data type validation
* Sensor range validation
* Timestamp validation
* Timestamp monotonicity
* Machine/sensor existence validation

Invalid records are sent to a **Dead-Letter Table** instead of being silently discarded.

```text
Telemetry
    ↓
Schema Validation
    ↓
Range Validation
    ↓
Timestamp Validation
    ↓
 ┌───────────────┐
 │               │
Valid          Invalid
 │               │
 ▼               ▼
TimescaleDB    Dead Letter
```

---

# 🗄️ 3. Time-Series Storage

FactoryPulse uses **TimescaleDB** for high-frequency sensor telemetry.

The database stores:

* Machine metadata
* Sensor metadata
* Raw telemetry
* Processed features
* Anomaly scores
* RUL predictions
* Failure probabilities
* Alerts
* Maintenance history
* Audit events

TimescaleDB hypertables are used for efficient time-series storage and querying.

---

# ⚙️ 4. Feature Engineering

The feature pipeline converts raw sensor streams into ML-ready features.

### Statistical Features

* Mean
* Standard deviation
* Minimum
* Maximum
* Median
* Variance
* RMS
* Rolling statistics

### Frequency-Domain Features

* FFT
* Frequency bands
* Band energy
* Spectral kurtosis

### Temporal Features

* Exponential moving average
* Trend
* Rate of change
* Lag features
* Rolling windows

Features are:

```text
Raw Telemetry
      ↓
Windowing
      ↓
Feature Extraction
      ↓
Normalization
      ↓
Feature Store / Redis
      ↓
ML Inference
```

Redis is used for low-latency access to online features.

---

# 🤖 5. Anomaly Detection

FactoryPulse uses an ensemble of:

### LSTM Autoencoder

The model learns normal machine behavior.

```text
Normal Sensor Window
        ↓
     Encoder
        ↓
   Latent Vector
        ↓
     Decoder
        ↓
Reconstructed Window
```

The anomaly score is based on reconstruction error:

```text
Anomaly Score = Reconstruction Error
```

A large reconstruction error indicates that the current machine behavior differs significantly from learned normal behavior.

### Isolation Forest

Isolation Forest provides an additional unsupervised anomaly signal.

The two signals are combined:

```text
LSTM AE Score
      +
Isolation Forest Score
      ↓
Ensemble Anomaly Score
      ↓
Adaptive Machine Threshold
      ↓
Alert
```

Thresholds are adaptive rather than being fixed globally.

---

# 📉 6. Remaining Useful Life Prediction

FactoryPulse predicts the **Remaining Useful Life (RUL)** of equipment.

The primary model is a **Transformer Encoder implemented in PyTorch**.

```text
Sensor Window
      ↓
Feature Embedding
      ↓
Positional Encoding
      ↓
Transformer Encoder
      ↓
Pooling / Representation
      ↓
Regression Head
      ↓
Predicted RUL
```

An **XGBoost model** is maintained as a baseline.

### Example

```text
Machine: ENGINE_024

Current Cycle: 180
Predicted RUL: 34 cycles

Estimated Status:
⚠️ Maintenance Recommended
```

---

# 🏷️ 7. Failure-Mode Classification

The system classifies probable failure modes using **LightGBM**.

Supported AI4I failure categories include:

* Tool wear
* Heat dissipation failure
* Power failure
* Overstrain failure

Example:

```text
Failure Probability

Heat Dissipation     ████████████████  72%
Overstrain           ██████             18%
Power Failure        ██                  7%
Tool Wear            █                    3%
```

---

# 🌡️ 8. Infrared Thermography

FactoryPulse includes a thermal-inspection module to detect abnormal heat patterns.

The pipeline uses:

* OpenCV
* Adaptive thresholding
* Region growing
* Ambient-temperature comparison
* ΔT scoring
* Hotspot detection

Example pipeline:

```text
Thermal Image
      ↓
Preprocessing
      ↓
Temperature/Intensity Analysis
      ↓
Adaptive Threshold
      ↓
Hotspot Segmentation
      ↓
Region Analysis
      ↓
ΔT vs Ambient
      ↓
Thermal Anomaly Score
```

The thermal score is fused with other machine-health signals.

Hotspot overlays can be attached to alerts as visual evidence.

---

# ❤️ 9. Machine Health Score

Every machine receives a health score from **0–100**.

The score combines:

* Anomaly trend
* Predicted RUL
* Failure probabilities
* Open alerts
* Thermal anomaly score

Example:

```text
Machine: PUMP-102

Health Score: 61 / 100

Anomaly Trend:     Elevated
Predicted RUL:     72 cycles
Thermal Status:    Warning
Open Alerts:       1
```

---

# 🚨 10. Alert Engine

The alert engine transforms ML predictions into actionable alerts.

Alert levels:

| Severity | Meaning                         |
| -------- | ------------------------------- |
| INFO     | Minor deviation                 |
| WARN     | Developing abnormal behavior    |
| CRITICAL | Immediate maintenance attention |

The engine implements:

* Alert deduplication
* Severity ranking
* Hysteresis
* Alert persistence
* Model-output snapshots
* Acknowledge/resolve workflow
* Webhook notifications

Hysteresis prevents alerts from repeatedly switching between states when a machine is close to a threshold.

---

# 🧠 11. GenAI Maintenance Copilot

FactoryPulse includes an agentic maintenance assistant powered by **LangGraph**.

The copilot does not rely solely on the LLM's internal knowledge.

Instead, it combines:

```text
Live Machine Data
       +
ML Predictions
       +
Maintenance History
       +
Equipment Manuals
       +
Fault-Code Documentation
       ↓
    RAG System
       ↓
LangGraph Agent
       ↓
Grounded Response
```

---

# 📚 RAG Knowledge Base

Maintenance documents are processed through:

```text
Documents
   ↓
Document Loader
   ↓
Cleaning
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
```

Sources include:

* Equipment manuals
* Maintenance procedures
* Fault-code documentation
* Historical maintenance logs
* Troubleshooting guides

Qdrant supports the retrieval layer using **dense + keyword/hybrid search**.

---

# 🛠️ Copilot Tools

The LangGraph agent has access to tools such as:

```text
get_machine_health()
get_recent_anomalies()
get_rul()
search_manuals()
get_maintenance_history()
```

This allows the copilot to combine **real-time operational data** with documented maintenance knowledge.

---

# 💬 Example Copilot Interaction

### Technician

```text
Why is MACHINE_017 showing a critical alert?
```

### Copilot

```text
MACHINE_017 is currently showing abnormal vibration and
temperature behavior.

The vibration anomaly score has increased consistently over
the last 4 hours, while the predicted RUL has decreased to
approximately 18 cycles.

The most likely issue is bearing degradation.

Recommended actions:
1. Inspect bearing housing temperature.
2. Check vibration around the bearing frequency band.
3. Verify lubrication condition.
4. Follow the bearing inspection procedure in the
   maintenance manual.

Safety:
Isolate the machine according to the documented lockout/tagout
procedure before physical inspection.

Sources:
[Maintenance Manual – Bearing Inspection]
[Fault Code F-204]
[Maintenance Log – MACHINE_017]
```

The system performs a **groundedness check** before returning the response.

Unsupported claims trigger regeneration or an explicit:

```text
Not found in available documentation.
```

---

# 📝 AI Work-Order Generation

When a critical alert occurs, the copilot can generate a maintenance work-order draft.

Example:

```text
WORK ORDER

Machine:
PUMP-102

Severity:
CRITICAL

Symptoms:
- Increasing vibration
- Elevated bearing temperature
- Declining predicted RUL

Probable Cause:
Bearing degradation

Recommended Inspection:
- Inspect bearing housing
- Check lubrication
- Inspect vibration spectrum

Required Parts:
- Bearing assembly
- Recommended lubricant

Safety Steps:
- Shut down equipment
- Apply lockout/tagout procedure
- Verify zero-energy state

Evidence:
- Sensor anomaly
- RUL prediction
- Thermal hotspot
- Maintenance documentation

Status:
Pending Engineer Approval
```

The final work order requires human approval.

---

# 📊 12. Streamlit Operations Console

The Streamlit dashboard provides an operational view of the entire machine fleet.

### Fleet Overview

```text
┌─────────────────────────────────────────────┐
│              FACTORYPULSE                   │
├─────────────────────────────────────────────┤
│ Machines       Healthy     Warning Critical │
│   120             91          21       8    │
├─────────────────────────────────────────────┤
│                                             │
│ Machine       Health     RUL       Status    │
│ M-001          92       184       HEALTHY   │
│ M-002          74        91       WARNING   │
│ M-003          38        18       CRITICAL  │
│                                             │
└─────────────────────────────────────────────┘
```

Machine detail pages include:

* Live sensor charts
* Historical trends
* Anomaly overlays
* RUL curves
* Failure probabilities
* Thermal hotspot images
* Alert history
* Maintenance history
* Copilot interface

---

# 🔐 13. Authentication & Authorization

FactoryPulse uses JWT-based authentication.

### Roles

| Role       | Capabilities                                       |
| ---------- | -------------------------------------------------- |
| Technician | View machines, alerts, copilot, acknowledge alerts |
| Engineer   | Technician permissions + models, assets, analytics |
| Admin      | Full system and user management                    |

Passwords are securely hashed using **bcrypt**.

Role-gated API endpoints prevent unauthorized operations.

---

# 📝 14. Audit Logging

Important actions are recorded in an audit trail.

Examples:

```text
User acknowledged alert
User resolved alert
Engineer approved work order
Model promoted to production
Machine configuration changed
Copilot recommendation approved
```

This provides traceability for maintenance decisions.

---

# 🧪 Machine Learning Lifecycle

FactoryPulse follows an ML lifecycle based on **DVC + MLflow**.

```text
Dataset
   ↓
DVC Versioning
   ↓
Data Processing
   ↓
Feature Engineering
   ↓
Training
   ↓
Evaluation
   ↓
MLflow Tracking
   ↓
Model Registry
   ↓
Staging
   ↓
Validation
   ↓
Production
```

Candidate models can also be evaluated in **shadow mode** before becoming production models.

---

# 📦 Public Datasets

FactoryPulse is designed to be completely reproducible without access to factory equipment.

### NASA C-MAPSS

Used primarily for:

* RUL prediction
* Degradation modeling
* Sequence learning

### NASA IMS Bearing Dataset

Used for:

* Bearing degradation
* Vibration analysis
* Anomaly detection

### AI4I 2020 Predictive Maintenance Dataset

Used for:

* Failure classification
* Predictive maintenance experiments
* Failure-mode analysis

Dataset processing converts these sources into a common FactoryPulse schema.

---

# 🗂️ Canonical Data Schema

FactoryPulse normalizes different datasets into a common representation:

```text
machine_id
timestamp
machine_class
sensor_id
sensor_type
value
operating_condition
failure_status
failure_mode
rul
```

This allows different datasets and simulated telemetry to use the same downstream pipeline.

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI / REST
* Pydantic
* SQLAlchemy
* PostgreSQL
* TimescaleDB
* Redis

## Machine Learning

* PyTorch
* Scikit-learn
* XGBoost
* LightGBM
* NumPy
* Pandas
* SciPy

## GenAI

* LangGraph
* LangChain
* Qdrant
* Embedding models
* LLM provider

## Computer Vision

* OpenCV
* NumPy

## MLOps

* MLflow
* DVC
* Git
* CI/CD

## Frontend / Operations

* Streamlit
* Plotly

## Messaging

* MQTT
* HTTP/Webhooks

## Infrastructure

* Docker
* Docker Compose
* Prometheus

---

# 📁 Project Structure

```text
factorypulse/
│
├── services/
│   │
│   ├── simulator/
│   │   ├── src/
│   │   └── tests/
│   │
│   ├── ingestion/
│   │   ├── src/
│   │   └── tests/
│   │
│   ├── feature_pipeline/
│   │   ├── src/
│   │   └── tests/
│   │
│   ├── inference/
│   │   ├── src/
│   │   └── tests/
│   │
│   ├── alert_engine/
│   │   ├── src/
│   │   └── tests/
│   │
│   ├── copilot/
│   │   ├── src/
│   │   └── tests/
│   │
│   └── api/
│       ├── src/
│       └── tests/
│
├── ml/
│   │
│   ├── anomaly_detection/
│   ├── rul_prediction/
│   ├── failure_classification/
│   ├── thermal_detection/
│   ├── training/
│   └── evaluation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── rag/
│   ├── ingestion/
│   ├── retrieval/
│   ├── tools/
│   └── evaluation/
│
├── dashboard/
│   └── streamlit_app/
│
├── database/
│   ├── migrations/
│   └── schemas/
│
├── tests/
│
├── docs/
│   ├── architecture/
│   ├── adr/
│   └── api/
│
├── docker/
│
├── .github/
│   └── workflows/
│
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── .env.example
└── README.md
```

---

# ⚡ Performance Requirements

FactoryPulse targets:

```text
Ingestion throughput:
≥ 200 messages/sec

Inference:
p95 < 150 ms

Sensor → Alert:
< 5 seconds

Copilot first token:
< 3 seconds
```

The ingestion layer is designed to be horizontally scalable through stateless service instances and batched database writes.

---

# 📈 Monitoring

Every service exposes health endpoints:

```text
/health
/ready
```

Prometheus metrics include:

* Message throughput
* Processing latency
* Inference latency
* Error rate
* Queue depth
* Model prediction counts
* Alert counts
* API latency

Structured JSON logs are used for centralized observability.

---

# 📊 Model Drift Detection

FactoryPulse monitors changes in production data using:

* PSI
* Kolmogorov–Smirnov tests
* Feature distribution monitoring
* Prediction distribution monitoring

Drift reports are logged to MLflow.

```text
Production Data
      ↓
Distribution Monitoring
      ↓
PSI / KS Test
      ↓
Drift Detected?
   ↙        ↘
 Yes         No
 ↓            ↓
Retraining   Continue
Pipeline
```

---

# 🔄 CI/CD

The project is designed for automated CI/CD.

Pipeline:

```text
Git Push
   ↓
Lint
   ↓
Type Check
   ↓
Unit Tests
   ↓
Integration Tests
   ↓
Build Docker Images
   ↓
Security Checks
   ↓
Deploy
```

Core business logic targets **80%+ unit-test coverage**.

---

# 🐳 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/factorypulse.git

cd factorypulse
```

## 2. Configure environment variables

```bash
cp .env.example .env
```

Configure:

```env
DATABASE_URL=
REDIS_URL=
QDRANT_URL=
MLFLOW_TRACKING_URI=
JWT_SECRET=
LLM_API_KEY=
MQTT_BROKER_URL=
```

Never commit `.env`.

---

## 3. Start infrastructure

```bash
docker compose up -d
```

Expected services include:

```text
TimescaleDB
Redis
Qdrant
MLflow
MQTT Broker
Prometheus
FactoryPulse Services
```

---

# ▶️ Start the Simulator

```bash
python -m services.simulator
```

The simulator publishes telemetry to the configured MQTT broker.

Example:

```text
MACHINE_001 → vibration=4.2 temperature=71.3 rpm=1820
MACHINE_002 → vibration=7.8 temperature=84.1 rpm=1762
```

---

# 🧠 Train Models

Example:

```bash
dvc repro
```

This executes the reproducible ML pipeline.

Experiments can be viewed through MLflow:

```text
MLflow UI
    ↓
Experiments
    ↓
Runs
    ↓
Metrics
    ↓
Artifacts
    ↓
Model Registry
```

---

# 📊 Launch Dashboard

```bash
streamlit run dashboard/streamlit_app/app.py
```

The dashboard provides:

* Fleet health
* Machine telemetry
* Anomaly scores
* RUL predictions
* Alerts
* Thermal evidence
* Copilot
* Model monitoring

---

# 🧪 Testing

Run unit tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Run linting:

```bash
ruff check .
```

Run type checking:

```bash
mypy .
```

---

# 🔍 Example End-to-End Flow

Consider a bearing beginning to degrade.

```text
1. Machine vibration slowly increases
              ↓
2. Simulator publishes telemetry
              ↓
3. Ingestion validates the data
              ↓
4. TimescaleDB stores the telemetry
              ↓
5. Feature pipeline calculates vibration features
              ↓
6. LSTM Autoencoder detects increasing reconstruction error
              ↓
7. Isolation Forest confirms abnormal behavior
              ↓
8. Transformer predicts decreasing RUL
              ↓
9. Thermal pipeline detects abnormal hotspot
              ↓
10. Health score drops from 86 → 54
              ↓
11. Alert engine creates WARN/CRITICAL alert
              ↓
12. Dashboard displays machine degradation
              ↓
13. Technician asks the Copilot for diagnosis
              ↓
14. LangGraph retrieves manuals + fault codes
              ↓
15. Agent combines retrieved knowledge with live ML results
              ↓
16. Groundedness check validates the answer
              ↓
17. Copilot generates a work-order draft
              ↓
18. Engineer reviews and approves it
```

---

# 🎯 Success Metrics

FactoryPulse will be evaluated across four dimensions.

### Machine Learning

* RUL RMSE
* MAE
* Precision
* Recall
* F1-score
* False alarm rate
* Early warning time

### System Performance

* Messages/sec
* API latency
* Inference p95
* End-to-end alert latency
* Copilot first-token latency

### GenAI

* Retrieval precision
* Citation coverage
* Groundedness
* Answer relevance
* Unsupported-claim rate

### Business Simulation

* Maintenance cost
* Downtime
* Prevented failures
* Number of unnecessary maintenance operations
* Mean time between failures

---

# 🧮 Maintenance Cost Simulation

FactoryPulse can compare different maintenance policies:

```text
                    ┌─────────────────────┐
                    │ Maintenance Policy  │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       Reactive          Calendar-Based     FactoryPulse
       Maintenance        Maintenance       Predictive
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                     Discrete Event Simulation
                               ↓
                     Cost / Downtime Analysis
```

The objective is to demonstrate reduced:

* Unplanned downtime
* Emergency repair cost
* Unnecessary component replacement
* Production loss

---

# 🔒 Security

FactoryPulse follows basic production-security practices:

* JWT authentication
* Role-based authorization
* bcrypt password hashing
* Environment-based secrets
* Input validation
* API authorization
* Audit logging
* No secrets committed to Git
* Model version pinning

---

# ⚠️ Important Disclaimer

FactoryPulse is an **engineering and research prototype** designed for demonstrating predictive-maintenance and industrial AI concepts.

It uses public datasets and simulated telemetry and should **not be used as the sole basis for safety-critical industrial decisions** without validation against real equipment, domain-specific requirements, and appropriate industrial safety procedures.

---

# 🗺️ Roadmap

### Phase 1 — Data Foundation

* [x] Canonical telemetry schema
* [ ] Dataset ingestion
* [ ] DVC pipeline
* [ ] Sensor simulator
* [ ] TimescaleDB integration

### Phase 2 — ML

* [ ] Feature engineering
* [ ] LSTM Autoencoder
* [ ] Isolation Forest
* [ ] Transformer RUL model
* [ ] XGBoost baseline
* [ ] LightGBM failure classifier

### Phase 3 — Real-Time Intelligence

* [ ] Streaming inference
* [ ] Adaptive thresholds
* [ ] Health scoring
* [ ] Alert engine
* [ ] Webhook notifications

### Phase 4 — Thermal Intelligence

* [ ] Thermal dataset integration
* [ ] Synthetic hotspot generator
* [ ] OpenCV hotspot detection
* [ ] Thermal anomaly scoring
* [ ] Multi-modal health fusion

### Phase 5 — GenAI

* [ ] Document ingestion
* [ ] Qdrant indexing
* [ ] Hybrid retrieval
* [ ] LangGraph agent
* [ ] Machine-data tools
* [ ] Citation generation
* [ ] Groundedness evaluation
* [ ] Work-order generation

### Phase 6 — Production Engineering

* [ ] JWT authentication
* [ ] RBAC
* [ ] Audit logging
* [ ] Prometheus monitoring
* [ ] Drift detection
* [ ] CI/CD
* [ ] Load testing
* [ ] Docker Compose deployment

---

# 🌟 Why FactoryPulse?

FactoryPulse demonstrates how multiple modern engineering and AI technologies can work together in a single real-world system:

```text
                 FactoryPulse
                      │
       ┌──────────────┼──────────────┐
       │              │              │
   Data/Streaming     ML           GenAI
       │              │              │
     MQTT          PyTorch       LangGraph
     Redis         XGBoost       Qdrant
 TimescaleDB      LightGBM        RAG
       │              │              │
       └──────────────┼──────────────┘
                      │
                 MLOps / DevOps
                      │
              Docker + MLflow
              DVC + Prometheus
                      │
                      ▼
          Predictive Maintenance
```

The project brings together:

**Data Engineering + Time-Series Analytics + Deep Learning + Computer Vision + MLOps + RAG + Agentic AI + Backend Engineering + Observability**

into one end-to-end Industrial AI platform.

---

# 👨‍💻 Author

**Your Name**

AI/ML Engineer | Backend Engineer | Generative AI

### Areas demonstrated

* Python
* Backend Engineering
* Machine Learning
* Deep Learning
* Time-Series Analysis
* Predictive Maintenance
* Computer Vision
* RAG
* Agentic AI
* MLOps
* Docker
* System Design

---

# 📄 License

This project is intended for educational, research, and demonstration purposes.

Add your preferred license here, for example:

```text
MIT License
```
