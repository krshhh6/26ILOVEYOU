# 🛢️ SpillSense — Maritime Intelligence C2 Platform

<div align="center">

[![SIH 2026](https://img.shields.io/badge/SIH-2026-blue.svg?style=for-the-badge&logo=target&logoColor=white)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-SIH26143-orange.svg?style=for-the-badge)](https://www.sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Disaster%20Management%20%2F%20MDA-red.svg?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](#-license)

<br/>

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19.2+-61DAFB.svg?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF.svg?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_WASM-Edge_AI-005CED.svg?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PostGIS-4169E1.svg?style=flat-square&logo=postgresql&logoColor=white)](https://postgis.net/)
[![Copernicus](https://img.shields.io/badge/Copernicus-Sentinel_1_%26_CMEMS-002D62.svg?style=flat-square)](https://dataspace.copernicus.eu/)
[![ISRO Bhoonidhi](https://img.shields.io/badge/ISRO-Bhoonidhi_STAC_API-FF9933.svg?style=flat-square)](https://bhoonidhi.nrsc.gov.in/)

**Autonomous Spaceborne Radar AI Detection • 4D Hydrodynamic Lagrangian Drift Reversal • Dark Vessel AIS Forensic Attribution • Tamper-Proof Cryptographic Court Dossiers**

[Executive Summary](#-executive-summary) • [4-Stage Pipeline](#-how-it-works--the-4-stage-pipeline) • [Tactical Features](#-tactical-c2-command--control-features) • [System Architecture](#-system-architecture) • [Quick Start](#-quick-start--local-deployment) • [Judge FAQs](#-faq--anticipated-judge-questions) • [Team](#-team-bug-stalkers)

</div>

---

## 📌 Table of Contents

- [Executive Summary](#-executive-summary)
- [The National Maritime Crisis](#-the-national-maritime-crisis)
- [Our Solution — In One Line](#-our-solution--in-one-line)
- [How It Works — The 4-Stage Pipeline](#-how-it-works--the-4-stage-pipeline)
  - [Stage 1: Multi-Sensor Spaceborne Radar Fusion & Bragg Damping](#stage-1--multi-sensor-spaceborne-radar-fusion--bragg-damping)
  - [Stage 2: In-Browser Edge AI Deep Learning Segmentation](#stage-2--in-browser-edge-ai-deep-learning-segmentation)
  - [Stage 3: 4D Lagrangian Hydrodynamic Drift Backtracking & Forecasting](#stage-3--4d-lagrangian-hydrodynamic-drift-backtracking--forecasting)
  - [Stage 4: Dark Vessel Forensic Attribution & AIS Blackout Diagnostic](#stage-4--dark-vessel-forensic-attribution--ais-blackout-diagnostic)
  - [Stage 5: Cryptographic Evidence Locker & Court Dossier](#stage-5--cryptographic-evidence-locker--court-dossier)
- [Tactical C2 Command & Control Features](#-tactical-c2-command--control-features)
  - [Core Feature Matrix](#core-feature-matrix)
  - [Pan-India Sovereign Basins](#8-sovereign-pan-india-monitoring-basins)
  - [ECDIS Hydrographic Engine & Interactive Tools](#ecdis-hydrographic-engine--interactive-tools)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Quick Start & Local Deployment](#-quick-start--local-deployment)
  - [Prerequisites](#prerequisites)
  - [Option 1: One-Click Instant Local Launch](#option-1-one-click-instant-local-launch-powershell)
  - [Option 2: Standalone Static Dashboard](#option-2-standalone-static-dashboard-zero-dependencies)
  - [Option 3: Full-Stack Manual Installation](#option-3-full-stack-manual-installation)
- [REST & WebSocket API Reference](#-rest--websocket-api-reference)
- [Operational Impact & Maritime Value](#-operational-impact--maritime-value)
- [Regulatory & Legal Compliance](#-regulatory--legal-compliance)
- [Honest Limitations & Defensible Engineering](#-honest-limitations--defensible-engineering)
- [Engineering Roadmap](#-engineering-roadmap)
- [FAQ — Anticipated Judge Questions](#-faq--anticipated-judge-questions)
- [Team BUG STALKERS](#-team-bug-stalkers)
- [License](#-license)

---

## 🌊 Executive Summary

India commands an Exclusive Economic Zone (EEZ) exceeding **2.37 million km²**, traversed by over **70% of the world's seaborne crude petroleum trade**. Commercial tankers voyaging from the Persian Gulf across the Arabian Sea, around Cape Comorin, and through the Bay of Bengal to the Malacca Strait routinely pass through fragile coastal ecosystems and fisheries.

When illicit bilge dumping or accidental spills occur, traditional maritime monitoring struggles with:
1. **Cloud cover and nocturnal blindness** of optical satellites.
2. **False alarms** caused by natural look-alikes (algal blooms, low-wind sea slicks, biogenic films).
3. **The Static Location Fallacy**: ocean currents and winds transport surface slicks rapidly; the coordinates of detection are never the original point of discharge.
4. **Dark Vessel Evasion**: rogue vessels intentionally power off AIS (Automatic Identification System) transponders during illicit tank-washing.
5. **Lack of Court Admissibility**: unstructured satellite screenshots fail to meet legal thresholds for international maritime prosecution under UNCLOS and the Indian Merchant Shipping Act, 1958.

**SpillSense** solves all five bottlenecks as a single, end-to-end tactical command-and-control platform.

---

## 🚨 The National Maritime Crisis

```
┌───────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Operational Bottleneck        │ Real-World Maritime Impact                                  │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Satellite Look-Alikes      │ Manual SAR interpretation mistakes calm water & algae for   │
│    & False Positives          │ oil, wasting critical Coast Guard emergency response hours. │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. The Static Location        │ Winds & currents transport surface slicks dozens of nautical│
│    Fallacy                    │ miles. Responders searching the detection point find nothing│
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Dark Vessel AIS Blackouts  │ Polluters deactivate AIS transponders to dump oily bilge in │
│    & Evasive Manifolds        │ the dark, escaping liability across sovereign EEZ waters.   │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Evidentiary Chain of       │ Raw data fails court scrutiny. Authorities lack tamper-proof│
│    Custody Inadmissibility    │ audit trails compliant with Section 356 of Indian MSA 1958. │
└───────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 💡 Our Solution — In One Line

> Point a satellite at the ocean → SpillSense tells you **what it is, where it really came from, who probably did it, and hands you court-ready proof** — all inside one tactical command dashboard.

---

## ⚙️ How It Works — The 4-Stage Pipeline

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  STAGE 1: DETECT    │     │ STAGE 2: TRACE BACK │     │ STAGE 3: ATTRIBUTE  │     │   STAGE 4: PROVE    │
│  Spaceborne SAR/    │ ──▶ │ 4D Hydrodynamic     │ ──▶ │ Spatiotemporal AIS  │ ──▶ │ SHA-256 Forensic    │
│  Optical Fusion     │     │ Lagrangian Backtrack│     │ Dark Vessel Correl. │     │ Court Dossier       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### Stage 1 — Multi-Sensor Spaceborne Radar Fusion & Bragg Damping

- **All-Weather, Day/Night Acquisition**: Ingests **Copernicus Sentinel-1 IW GRD** (C-band SAR) and **ISRO EOS-04 C-SAR** granules through the **ISRO NRSC Bhoonidhi OpenSearch STAC 1.0.0 API**.
- **Radiometric Calibration & Lee Speckle Filtering**: Normalizes radar backscatter ($\sigma^0$ in dB) and applies a 5×5 Lee filter to suppress granular speckle while preserving sharp slick boundaries.
- **Dual-Band Bragg Wave Damping Physics**: Capitalizes on **NASA-ISRO NISAR Dual-Frequency SAR (L-band 1.26 GHz + S-band 3.2 GHz)**. Real mineral crude oil dampens both short capillary and long gravity-capillary waves, whereas biogenic algal films only dampen high-frequency ripples. Computing the **Dual-Frequency Damping Index (DFDI)** eliminates false alarms at the physical layer.
- **MetOcean Environmental Sanity Mask**: Filters scenes against **ERA5 wind speed exclusion thresholds** (3.0 m/s to 12.0 m/s operational window) and **MODIS Aqua Chlorophyll-a** data to eliminate low-wind calm zones and algal blooms.

### Stage 2 — In-Browser Edge AI Deep Learning Segmentation

- **Zero-Cloud Edge Inference**: Runs convolutional segmentation directly inside the operator's browser memory using **ONNX Runtime WebAssembly (WASM)** with WebGL/WebGPU acceleration. Inference completes in **<150 ms** without sending sensitive radar granules across low-bandwidth shipboard satcom links.
- **SpillSegNet Architecture**: Deep convolutional U-Net with ResNet-50 backbone, enhanced with Atrous Spatial Pyramid Pooling (**ASPP**) and Convolutional Block Attention Modules (**CBAM**).
- **Bonn Agreement 5-Class Morphology**: Segments pixels into international standard classes:
  1. *Core Heavy Emulsion* (Bonn Code 4/5)
  2. *Iridescent / Metallic Sheen* (Bonn Code 1/2)
  3. *Tail Filaments*
  4. *Natural Look-Alikes / Biogenic Films*
  5. *Clean Navigable Seawater*
- **Explainable AI (XAI)**: Generates real-time **Occlusion Sensitivity Attention Heatmaps**, allowing naval operators to verify exactly which radar gradient anomalies triggered the neural network's detection.

### Stage 3 — 4D Lagrangian Hydrodynamic Drift Backtracking & Forecasting

- **Physics-Based Reverse Backtracking**: Couples with **OpenDrift / OpenOil** executing **4th-Order Runge-Kutta (RK4)** integration with 1,000 Monte Carlo simulated particles rewinding up to $T - 72\text{ hours}$.
- **MetOcean Forcing Vectors**:
  $$\vec{V}_{\text{drift}} = \vec{U}_{\text{current}} + \alpha \cdot \vec{U}_{\text{wind}} + \vec{U}'_{\text{turbulent}}$$
  Driven by **Copernicus Marine Service (CMEMS)** 3D ocean current velocity vectors and **ECMWF ERA5** 10m wind fields with a 3.5% Stokes drift windage factor.
- **Sovereign Peninsular Shoreline Clamping**: Enforces coastal land non-penetration geometry along the Indian peninsular boundary, ensuring particles do not propagate erroneously inland and providing exact time-to-shore impact estimates.
- **Spatiotemporal Probability Envelopes**: Computes standard GeoJSON isobar contours representing **50%**, **75%**, and **90% discharge probability zones**.

### Stage 4 — Dark Vessel Forensic Attribution & AIS Blackout Diagnostic

- **Spatiotemporal Trajectory Intersection**: Cross-correlates the reverse-backtracked $T-72\text{h}$ probability envelopes against historical **Automatic Identification System (AIS)** vessel tracks.
- **Explainable Multi-Factor Attribution Scoring**:
  $$S = w_{\text{dist}} \cdot S_{\text{dist}} + w_{\text{time}} \cdot S_{\text{time}} + w_{\text{gap}} \cdot S_{\text{gap}} + w_{\text{type}} \cdot S_{\text{type}}$$
  Every candidate vessel receives an explainable score ($0.00 \to 1.00$) balancing distance to isobar core, timestamp alignment, AIS silence gaps, and vessel deadweight tonnage / cargo type.
- **AIS Silence Gap & Pumping Detection**: Automatically flags transponder blackouts exceeding 60 minutes and detects speed reductions (e.g. 16 knots $\rightarrow$ 5 knots) characteristic of illegal ballast washing and bunker purging.
- **What-If Sensitivity Tuner**: Operators can dynamically adjust weight sliders in real time to test hypotheses and eliminate black-box investigative risk.

### Stage 5 — Cryptographic Evidence Locker & Court Dossier

- **SHA-256 Immutability**: Hashes all input rasters, environmental forcing files, AIS telemetry segments, and decision metrics into a tamper-evident cryptographic signature.
- **Statutory Admissibility**: Exports formatted briefing dossiers engineered for direct courtroom admission under:
  - **Section 356, Part XIA** of the **Indian Merchant Shipping Act, 1958** (Prevention and Containment of Pollution of the Sea by Oil).
  - **MARPOL 73/78 Annex I** (International Convention for the Prevention of Pollution from Ships).
  - **Section 65B of the Indian Evidence Act / BSA 2023** (Electronic records legal certification).

---

## 🖥️ Tactical C2 Command & Control Features

### Core Feature Matrix

| Feature | Operational Capability | Real-World Impact |
|:---|:---|:---|
| 🗺️ **ECDIS Hydrographic Engine** | Official OpenSeaMap seamarks (TSS channels, buoys, lighthouses, harbor approaches), ISRO Bhuvan WMTS, and sub-meter satellite layers. | Complies with international electronic navigation standards for naval command centers. |
| ⏱️ **4D Spatiotemporal Scrubber** | Persistent $T - 72\text{h} \to T_0$ timeline animating backward particle dispersion and vessel tracks simultaneously. | Pinpoints the exact minute and coordinate of the rogue discharge. |
| 🛰️ **Dual-Pane SAR Split Inspector** | Interactive before/after split slider comparing calibrated radar backscatter ($\sigma^0$) against AI segmentation masks. | Validates detection confidence visually against raw radar imagery without leaving the map. |
| 🧠 **SpillSegNet 5-Class Classifier** | In-browser ONNX WASM neural net isolating Core Emulsion, Sheen, Tail, Lookalike, and Water. | Eliminates false alarms from algae and calm water in under 150 ms. |
| 📊 **Live Spill Analytics Engine** | Computes slick area (ha), Bonn Agreement thickness codes, and estimated discharge volume ($m^3$). | Direct basis for Tier-1, Tier-2, and Tier-3 national disaster mobilization. |
| 🎛️ **What-If Sensitivity Tuner** | Dynamic sliders adjusting distance, timing, blackout, and vessel-type weights in real time. | Full investigative transparency; eliminates black-box legal vulnerability. |
| 📑 **Forensic Dossier & SHA-256 Locker**| One-click export of court-admissible dossiers with interactive hash verification modal. | Prevents evidence tampering and ensures admissible prosecution under Indian MSA 1958. |

---

### 8 Sovereign Pan-India Monitoring Basins

SpillSense ships pre-configured with operational presets across critical Indian maritime choke points:

1. **Mumbai High Offshore Platform Complex**: High-density extraction zone, underwater pipeline manifolds, offshore supply vessel tracks.
2. **Gulf of Kachchh / Vadinar Terminal**: Single point moorings (SPMs), heavy VLCC/ULCC crude tanker approaches.
3. **Palk Strait & Gulf of Mannar**: Shallow sensitive marine biosphere, international boundary line (IMBL) monitoring.
4. **Paradip Port Coastal Approach**: Major petroleum loading hub on Odisha coast, high-traffic anchorage corridor.
5. **Kochi / Arabian Sea Tanker Highway**: Cape Comorin convergence zone for transatlantic crude carriers.
6. **Ennore / Chennai Coastal Arc**: Industrial port cluster with dense maritime approach corridors.
7. **Kolkata / Haldia Hooghly Estuary**: Riverine-marine delta interface requiring complex tidal current simulation.
8. **Malacca Strait Western Gateway (Great Nicobar)**: Highest-density tanker choke point in the Indo-Pacific.

---

### ECDIS Hydrographic Engine & Interactive Tools

- **Sub-Meter Geospatial Resolution**: Seamlessly toggle between ISRO Bhuvan satellite imagery, OpenSeaMap nautical seamarks, and ArcGIS World Imagery.
- **NOS-DCP Cleanup Simulation**: Interactive containment calculator allocating boom lengths, skimmer recovery rates, and dispersant payloads based on slick viscosity.
- **Dynamic AIS Search Directory**: Search across vessels by MMSI, IMO, name, or call sign, or inspect coordinates directly (`18.74°N, 71.21°E`).

---

## 🏗️ System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          SPACEBORNE & METOCEAN INGEST         │
                  │  Sentinel-1 SAR • ISRO Bhoonidhi • CMEMS/ERA5 │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND CORE SERVICES                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │   SAR Ingestion &    │  │ 4D Lagrangian Drift  │  │ Spatiotemporal AIS Hub  │  │
│  │   Tiled Preprocessor │  │ Engine (OpenDrift)   │  │ (DuckDB Spatial Engine) │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └────────────┬────────────┘  │
└─────────────┼─────────────────────────┼───────────────────────────┼───────────────┘
              │                         │                           │
              ▼                         ▼                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                        TACTICAL C2 MARITIME DASHBOARD                             │
│       React 19 • TypeScript • Vite • Leaflet ECDIS Engine • ONNX Runtime WASM     │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │ In-Browser Edge AI   │  │ 4D Interactive Time  │  │ Multi-Factor What-If    │  │
│  │ Segmentation (WASM)  │  │ Scrubber (T-72h→T0)  │  │ Attribution Tuner       │  │
│  └──────────────────────┘  └──────────────────────┘  └─────────────────────────┘  │
└───────────────────────────────────────┬───────────────────────────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────────┐
                  │           CRYPTOGRAPHIC EVIDENCE LOCKER       │
                  │  SHA-256 Court Dossier • Section 356 MSA 1958 │
                  └───────────────────────────────────────────────┘
```

Full user and system flowcharts are documented in [`Files/AppFlow.md`](./Files/AppFlow.md).

---

## 🛠️ Tech Stack

| Layer | Component | Description |
|:---|:---|:---|
| **Satellite Remote Sensing** | Copernicus Sentinel-1 | C-band SAR IW GRD all-weather radar scenes (10m resolution) |
| | ISRO EOS-04 / RISAT | Dual-polarization C-band synthetic aperture radar |
| | NASA-ISRO NISAR | Dual-frequency L-band (1.26 GHz) & S-band (3.2 GHz) DFDI validation |
| **MetOcean Physics** | Copernicus CMEMS | 3D Global Ocean hydrodynamic velocity vectors (currents) |
| | ECMWF ERA5 | 10-meter surface wind vectors & atmospheric boundary forcing |
| | MODIS Aqua | Ocean color & Chlorophyll-a biogenic look-alike exclusion |
| **Edge AI & Computer Vision** | SpillSegNet | U-Net deep convolutional architecture with ResNet-50 backbone |
| | ASPP & CBAM | Multi-scale contextual pooling & dual channel/spatial attention |
| | ONNX Runtime Web | WebAssembly edge inference with zero-cloud data leakage (<150 ms) |
| **Hydrodynamic Simulation** | OpenDrift / OpenOil | 4th-Order Runge-Kutta Lagrangian particle trajectory solver |
| | Sovereign Boundary Clamping | Peninsular land non-penetration geometry & coastal stranding math |
| **Vessel Tracking & AIS** | DuckDB Spatial Engine | In-memory geospatial SQL query engine for high-throughput AIS tracks |
| | Spatiotemporal Intersector | Correlates vessel historical waypoints against 50/75/90% isobars |
| **Backend & APIs** | Python 3.11+ / FastAPI | High-concurrency RESTful microservices & real-time WebSocket streams |
| | Pytest & Docker | Containerized full-stack deployment with automated CI regression |
| **Frontend Tactical C2** | React 19 / TypeScript 5 | Modern reactive single-page command dashboard |
| | Leaflet & OpenSeaMap | Maritime hydrographic chart engine with nautical seamark tiles |
| | Tailwind CSS | High-contrast military C2 dark/light mode interface tokens |
| **Evidence & Integrity** | SHA-256 Hashing | Cryptographic tamper-evidence for input data and dossiers |
| | PDF Generation Engine | Automated forensic reporting formatted to Section 356 MSA 1958 |

---

## 📁 Repository Structure

```
.
├── backend/                        # FastAPI RESTful Microservices & Orchestration
│   ├── app/
│   │   ├── api/v1/endpoints.py     # Main REST routes (detect, drift, attribute, evidence)
│   │   ├── services/
│   │   │   ├── ais_duckdb_service.py # High-speed spatial AIS correlation engine
│   │   │   ├── sar_service.py      # Radar preprocessing & tile calibration
│   │   │   ├── drift_service.py    # OpenDrift RK4 Lagrangian simulation driver
│   │   │   └── pdf_evidence_generator.py # SHA-256 forensic dossier compiler
│   │   └── main.py                 # Application factory & WebSocket router
│   ├── requirements.txt            # Python dependencies (FastAPI, DuckDB, NumPy, etc.)
│   └── Dockerfile                  # Production container manifest
│
├── frontend/                       # Tactical C2 Maritime Dashboard (React 19 + TypeScript)
│   ├── src/
│   │   ├── components/
│   │   │   ├── MapPanel.tsx        # Leaflet ECDIS map with nautical overlays
│   │   │   ├── Topbar.tsx          # Tactical telemetry capsule & search directory
│   │   │   ├── Sidebar.tsx         # Primary command navigation drawer
│   │   │   ├── OpeningScreen.tsx   # 16:9 3D Globe launch sequence with zoom transition
│   │   │   └── views/
│   │   │       ├── DashboardView.tsx   # Executive situational awareness overview
│   │   │       ├── DetectionView.tsx   # SAR Detection Lab & ROI crop studio
│   │   │       ├── DriftView.tsx       # 4D reverse & forward trajectory workspace
│   │   │       ├── AttributionView.tsx # Candidate vessel scoring & What-If tuner
│   │   │       └── EvidenceView.tsx    # Forensic audit trail & cryptographic dossier
│   │   ├── services/
│   │   │   └── sarClassifier.ts    # In-browser ONNX Runtime WASM edge inference
│   │   └── index.css               # ECDIS hydrographic design tokens & styling
│   ├── package.json                # Frontend dependencies (React 19, Vite, Tailwind, Leaflet)
│   └── vite.config.ts              # Vite bundler configuration
│
├── ml/                             # AI / Deep Learning Training & Evaluation Pipeline
│   ├── models/                     # SpillSegNet U-Net PyTorch architectures
│   ├── weights/                    # Exported ONNX WASM model checkpoints
│   └── scripts/                    # DANN domain adaptation, CDF normalizers & bench tests
│
├── pipeline/                       # End-to-end orchestration chaining Stages 1–4
├── infra/                          # Deployment configurations (Docker Compose, Render)
├── docs/                           # Technical blueprints, ML improvement plans & docs
├── Files/                          # PRD, Tech Stack, Database Schema, and Flow Specifications
│   ├── prd.md                      # Complete Product Requirements Document
│   ├── tech_stack.md               # Verified data source list & priority rankings
│   ├── schema.md                   # Complete PostgreSQL/PostGIS database architecture
│   └── AppFlow.md                  # Comprehensive user & system flows with diagrams
├── index.html                      # Standalone Tactical C2 Maritime Dashboard (Zero-install entry point)
├── spill_sense_dashboard.html      # Fully self-contained single-file operational dashboard
├── start_localhost.ps1             # One-click dual-server launcher (Backend + Frontend)
├── run_frontend.ps1                # Vite frontend runner script
└── PROTOTYPE_PRESENTATION_SCRIPT.txt # Comprehensive 8-minute hackathon jury pitch script
```

---

## 🚀 Quick Start & Local Deployment

### Prerequisites

- **Node.js**: `v18.0.0` or higher (`v20+` recommended)
- **Python**: `3.10` or higher (`3.11` recommended)
- **Git**: Installed and configured on your system

---

### Option 1: One-Click Instant Local Launch (PowerShell)

On Windows, launch both the FastAPI backend and Vite frontend with a single command:

```powershell
# Clone the repository
git clone https://github.com/krshhh6/BUG-STALKERS-SPILL-SENSE.git
cd BUG-STALKERS-SPILL-SENSE

# Launch both servers in parallel
.\start_localhost.ps1
```

- **Frontend Tactical Dashboard**: [`http://localhost:5173`](http://localhost:5173)
- **Backend REST API**: [`http://localhost:8000`](http://localhost:8000)
- **Interactive Swagger API Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

### Option 2: Standalone Static Dashboard (Zero Dependencies)

To run the lightweight standalone command dashboard without installing Python packages:

```bash
# Start any static web server in the repository root
python -m http.server 8080

# Open in browser
# Navigate to: http://localhost:8080/index.html
```

---

### Option 3: Full-Stack Manual Installation

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend Setup

```bash
cd frontend

# Install Node modules
npm install

# Start development server
npm run dev -- --host --port 5173
```

---

## 📡 REST & WebSocket API Reference

The FastAPI backend exposes fully documented endpoints accessible via Swagger UI at `/docs`:

| Method | Endpoint | Description | Key Parameters |
|:---|:---|:---|:---|
| `GET` | `/api/v1/health` | Service health status and component checks | None |
| `POST` | `/api/v1/detect` | Runs deep learning SAR slick segmentation | `raster_file`, `band`, `threshold` |
| `POST` | `/api/v1/drift/simulate` | Triggers 4D Lagrangian reverse/forward simulation | `lat`, `lng`, `spill_time`, `hours` |
| `GET` | `/api/v1/drift/trajectory/{id}` | Fetches computed GeoJSON probability polygons | `incident_id` |
| `POST` | `/api/v1/attribute/vessels` | Scores candidate vessels within origin envelope | `incident_id`, `weights`, `search_radius` |
| `GET` | `/api/v1/evidence/dossier/{id}`| Generates SHA-256 cryptographically hashed PDF | `incident_id` |
| `GET` | `/api/v1/bhoonidhi/catalog` | Queries ISRO Bhoonidhi STAC catalog for SAR tiles | `bbox`, `start_date`, `end_date` |
| `WS` | `/ws/simulation` | WebSocket stream for live particle dispersion updates | `session_token` |

---

## 🎯 Operational Impact & Maritime Value

- **Who uses it:** Indian Coast Guard, Indian Navy Maritime Domain Awareness Cell, Directorate General of Shipping (DG Shipping), and State Pollution Control Boards.
- **What it replaces:** Manual visual radar interpretation, guesswork regarding spill origins, and unstructured retroactive AIS searches across thousands of vessel tracks.
- **What it delivers:** A single pane of glass taking an alert from *"unidentified dark anomaly on water"* to *"here is the top suspect vessel, here is the hydrodynamic physics math behind the conclusion, and here is a tamper-proof evidence dossier for legal prosecution."*

---

## ⚖️ Regulatory & Legal Compliance

SpillSense is engineered to comply directly with international and sovereign Indian maritime law:

- **Merchant Shipping Act, 1958 (Part XIA)**: Formatted for direct invocation by the Indian Coast Guard and DG Shipping under **Section 356** for maritime pollution containment, clean-up direction, and civil liability cost recovery.
- **MARPOL 73/78 (Annex I)**: Adheres to Regulations for the Prevention of Pollution by Oil, applying the Bonn Agreement Oil Appearance Code (BAOAC) for thickness and volume estimation.
- **UNCLOS Part XII (Articles 194, 211, 220)**: Establishes enforcement jurisdiction within India's 200 NM Exclusive Economic Zone against foreign-flagged dark vessels.
- **Section 65B of Indian Evidence Act / BSA 2023**: Generates cryptographically validated audit trails ensuring electronic satellite evidence is fully admissible in Indian courts of law.
- **WGS 84 (EPSG:4326)**: Standardized spatial projection across all ingested and exported vector layers.

---

## 🔬 Honest Limitations & Defensible Engineering

In accordance with responsible engineering practices:

1. **Probabilistic Decision Support**: SpillSense provides explainable, mathematically defensible evidence to accelerate investigations. It serves as an intelligence decision-support tool, not an autonomous judicial verdict; human maritime officer review is mandatory before punitive boarding actions.
2. **Satellite Revisit Latency**: While Sentinel-1 and NISAR provide unmatched all-weather radar fidelity, constellation revisit intervals range from 1 to 6 days. Supplementing with commercial SAR (e.g. Capella/ICEYE) or airborne Coast Guard Dornier patrol radar bridges revisit gaps.
3. **AIS Transponder Blackouts**: Attribution scoring relies on vessels that transmit AIS signals or exhibit identifiable transponder dropouts. Completely covert "ghost vessels" with zero historical AIS footprint are flagged as unidentified radar contacts.

---

## 🗺️ Engineering Roadmap

- [x] Full-Stack C2 Dashboard with real-time Leaflet ECDIS hydrographic engine
- [x] In-browser ONNX WASM deep learning segmentation with occlusion attention maps
- [x] Backward & Forward 4D Lagrangian hydrodynamic particle trajectory simulation
- [x] Explainable multi-factor AIS vessel attribution scoring with What-If tuner
- [x] Cryptographic SHA-256 evidence locker & automated court dossier generation
- [ ] Direct live STAC polling from ISRO Bhoonidhi and Copernicus CDSE data hubs
- [ ] Concurrent multi-basin monitoring across all 8 Indian maritime sectors
- [ ] Mobile-responsive field operations view for offshore interceptor patrol vessels
- [ ] Automated continuous retraining pipeline with expanded Indian EEZ radar dataset

---

## ❓ FAQ — Anticipated Judge Questions

**Q: Why not just use the satellite detection point as the spill location?**  
**A:** Ocean currents and wind continuously transport surface oil slicks after discharge. By the time a satellite acquisition occurs, the slick may have drifted 20 to 60 nautical miles from the actual discharge coordinates. The 4D Lagrangian back-tracking engine (Stage 3) is essential to rewind the drift physics and locate the true release site.

**Q: How is this different from a simple AIS proximity search?**  
**A:** Simple proximity searches only look for vessels near the current slick and completely ignore drift physics and vessels that went dark. Our engine rewinds ocean currents to the historic time of discharge, scores AIS silence gaps (blackouts), and analyzes engine throttle drops typical of illicit bilge-washing manifolds.

**Q: Is the attribution score an unexplainable "black box"?**  
**A:** No. The What-If Sensitivity Tuner exposes every mathematical weight ($w_{\text{dist}}$, $w_{\text{time}}$, $w_{\text{gap}}$, $w_{\text{type}}$) in real time. Maritime investigators can inspect each sub-score individually and adjust weights based on operational intelligence.

**Q: Can this evidence be admitted in an Indian court of law?**  
**A:** Yes. The evidence dossier is timestamped, SHA-256 cryptographically hashed for tamper-evidence, and formatted specifically under Section 356 of the Indian Merchant Shipping Act 1958 and Section 65B of the Indian Evidence Act / BSA 2023.

**Q: What happens if the SAR detection is a false positive (e.g., algal bloom or calm water)?**  
**A:** Stage 1 applies multi-physics verification: NASA-ISRO NISAR dual-frequency L+S Bragg damping ratios, ERA5 wind exclusion filtering (3–12 m/s), and MODIS Chlorophyll-a checks to physically eliminate biogenic look-alikes before pipeline execution.

---

## 👥 Team BUG STALKERS

Built with dedication for **Smart India Hackathon 2026** (Problem Statement **SIH26143**):

| Name | Role | Responsibilities | GitHub |
|:---|:---|:---|:---:|
| **Nitin Pandey** | **Remote Sensing & Geospatial AI Engineer** | Production SAR Image Intelligence, GeoTIFF Multiband Ingestion, Physics-Guided Capillary Wave Damping Gating, Forward Lagrangian Future Drift & Landfall Impact Modeling, Live MetOcean CMEMS/ERA5 API Integration | [@nitinpandey-dev](https://github.com/nitinpandey-dev) |
| **Shubh Jaiswal** | **Backend Core & DevOps Engineer** | Cryptographic Evidence Locker & Forensic Dossier Engine (BSA 2023 / Section 65B Compliance, SHA-256 Verifier), DuckDB Spatial Vector Engine, Docker Compose Containerization & GitHub Actions CI/CD Pipeline | [@shubhjaiswal551](https://github.com/shubhjaiswal551) |
| **Adityash Srivastava** | **Lead AI/ML Research Engineer** | Deep Learning Architecture (SpillSegNet U-Net, ASPP & CBAM Attention), DANN Domain Adaptation Training Pipeline, Active Learning Feedback API & Hard-Negative Mining, Domain Validation & Non-SAR Image Rejection Filter | [@adityashsrivastava](https://github.com/adityashsrivastava) |
| **Krishna Kant** | **Technical Lead & Systems Architect** | End-to-End System Architecture, Full-Stack C2 Pipeline, GIS Hydrographic Map Engine & AIS Kinematics Tracking, Multi-Sensor Satcom Integration (ISRO Bhuvan WMTS, NISAR, ArcGIS), Mobile View Optimization & Production Deployment | [@krshhh6](https://github.com/krshhh6) |
| **Tarun Agnihotri** | **Team Leader (Coordination & Documentation)** | Team coordination, repository documentation, and project submission management. | [@tarunagnihotri534](https://github.com/tarunagnihotri534) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed for **Smart India Hackathon 2026** by **Team BUG STALKERS**.  
Protected under sovereign maritime data handling standards.

<div align="center">

*SpillSense — Transforming Maritime Surveillance from Reactive Scramble to Sovereign Precision.*

</div>
