# 🛢️ SpillSense — Maritime Intelligence C2 Platform

<div align="center">

[![SIH 2026](https://img.shields.io/badge/SIH-2026-blue.svg?style=for-the-badge&logo=target&logoColor=white)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-SIH26143-orange.svg?style=for-the-badge)](https://www.sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Disaster%20Management%20%2F%20MDA-red.svg?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](#license)

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

[Live Web Demo](https://sih-26143-oil-spill.vercel.app) • [Key Features](#-tactical-c2-command--control-features) • [4-Stage Pipeline](#-how-it-works--the-4-stage-pipeline) • [System Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Legal Compliance](#-regulatory--legal-compliance)

</div>

---

## 📌 Table of Contents

- [Executive Summary](#-executive-summary)
- [The National Maritime Crisis](#-the-national-maritime-crisis)
- [How It Works — The 4-Stage Pipeline](#-how-it-works--the-4-stage-pipeline)
  - [Stage 1: Multi-Sensor Spaceborne Fusion & Bragg Damping](#stage-1--multi-sensor-spaceborne-radar-fusion--bragg-damping)
  - [Stage 2: In-Browser Edge AI Deep Learning Segmentation](#stage-2--in-browser-edge-ai-deep-learning-segmentation)
  - [Stage 3: 4D Lagrangian Hydrodynamic Drift Backtracking & Forecasting](#stage-3--4d-lagrangian-hydrodynamic-drift-backtracking--forecasting)
  - [Stage 4: Dark Vessel Forensic Attribution & AIS Blackout Diagnostic](#stage-4--dark-vessel-forensic-attribution--ais-blackout-diagnostic)
  - [Stage 5: Cryptographic Evidence Locker & Court Dossier](#stage-5--cryptographic-evidence-locker--court-dossier)
- [Tactical C2 Command & Control Features](#-tactical-c2-command--control-features)
  - [Pan-India Sovereign Basins](#8-sovereign-pan-india-monitoring-basins)
  - [ECDIS Hydrographic Engine & 4D Scrubber](#ecdis-hydrographic-engine--interactive-tools)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Quick Start & Local Deployment](#-quick-start--local-deployment)
  - [Option 1: One-Click Instant Local Launch](#option-1-one-click-instant-local-launch-powershell)
  - [Option 2: Standalone Static Dashboard](#option-2-standalone-static-dashboard-zero-dependencies)
  - [Option 3: Full-Stack Manual Installation](#option-3-full-stack-manual-installation)
- [REST & WebSocket API Reference](#-rest--websocket-api-reference)
- [Regulatory & Legal Compliance](#-regulatory--legal-compliance)
- [Honest Limitations & Defensible Engineering](#-honest-limitations--defensible-engineering)
- [Team BUG STALKERS](#-team-bug-stalkers)
- [License](#-license)

---

## 🌊 Executive Summary

India commands an **Exclusive Economic Zone (EEZ) exceeding 2.37 million km²**, traversed by over **70% of the world's seaborne crude petroleum trade**. Commercial tankers voyaging from the Persian Gulf across the Arabian Sea, around Cape Comorin, and through the Bay of Bengal to the Malacca Strait routinely pass through fragile coastal ecosystems and fisheries.

When illicit bilge dumping or accidental spills occur, traditional maritime monitoring struggles with:
1. **Cloud cover and nocturnal blindness** of optical satellites.
2. **False alarms** caused by natural look-alikes (algal blooms, low-wind sea slicks, biogenic films).
3. **The Static Location Fallacy**: ocean currents and winds transport surface slicks rapidly; the coordinates of detection are **never the original point of discharge**.
4. **Dark Vessel Evasion**: rogue vessels intentionally power off AIS (Automatic Identification System) transponders during illicit tank-washing.
5. **Lack of Court Admissibility**: unstructured satellite screenshots fail to meet legal thresholds for international maritime prosecution under **UNCLOS** and the **Indian Merchant Shipping Act, 1958**.

**SpillSense** is an autonomous tactical Command and Control (C2) maritime decision-support platform designed to solve all five bottlenecks in a unified, mathematically defensible system.

> **Our Solution in One Sentence:**
> Point a satellite at the ocean $\rightarrow$ SpillSense tells maritime commanders **what it is, where it came from, who did it, and generates a tamper-proof cryptographic dossier ready for international prosecution.**

---

## 🚨 The National Maritime Crisis

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  THE 4 OPERATIONAL BOTTLENECKS                              │
├───────────────────────────────┬─────────────────────────────────────────────────────────────┤
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
  - **UNCLOS Articles 194, 211, and 220** (Enforcement by coastal states against foreign-flagged polluters).

---

## 🖥️ Tactical C2 Command & Control Features

### 8 Sovereign Pan-India Monitoring Basins

SpillSense comes pre-configured with operational surveillance zones covering India's entire strategic coastline:

| Basin ID | Region / Corridor | Strategic Importance | Coordinates |
|:---:|:---|:---|:---:|
| **INC-001** | Gulf of Kutch / Vadinar SPM | Primary Indian crude import fairway & mangrove sanctuary | $22.48^\circ\text{N}, 69.52^\circ\text{E}$ |
| **INC-002** | Bombay High Offshore Fairway | Major offshore drilling infrastructure & supply corridor | $19.42^\circ\text{N}, 71.33^\circ\text{E}$ |
| **INC-003** | Goa Deepwater Shipping Corridor | Western Ghats iron-ore and container shipping route | $15.35^\circ\text{N}, 73.40^\circ\text{E}$ |
| **INC-004** | Cochin Port SPM Anchorage | Critical southern VLCC tanker traffic separation scheme | $09.95^\circ\text{N}, 76.10^\circ\text{E}$ |
| **INC-005** | Chennai Ennore Industrial Coast | High-density petroleum refinery & industrial port corridor | $13.25^\circ\text{N}, 80.35^\circ\text{E}$ |
| **INC-006** | Paradip Port Offshore Anchorage | Eastern energy hub & Mahanadi delta ecological sanctuary | $20.20^\circ\text{N}, 86.75^\circ\text{E}$ |
| **INC-007** | Strategic Six-Degree Channel | Gateway to Malacca Strait, highest tanker traffic globally | $06.00^\circ\text{N}, 93.50^\circ\text{E}$ |
| **INC-008** | Lakshadweep Kavaratti Sea | Coral atolls & Arabian Sea international transit lane | $10.55^\circ\text{N}, 72.60^\circ\text{E}$ |

### ECDIS Hydrographic Engine & Interactive Tools

- **OpenSeaMap & Hydrographic Seamarks**: Displays navigation buoys, lighthouses, TSS (Traffic Separation Scheme) corridors, harbor pilot approaches, and 200 NM sovereign EEZ boundaries.
- **GEBCO Ocean Bathymetry Relief**: Visualizes deep undersea trenches and continental shelves governing subsurface drift vectors.
- **4D Time-Scrubber ($T-72\text{h} \rightarrow T_0 \rightarrow T+24\text{h}$)**: Animate particle back-dispersion and vessel movements synchronously across time.
- **Dual-Pane SAR Split Inspector**: Side-by-side swipe tool comparing raw calibrated radar backscatter against the deep-learning mask.
- **Real-Time MetOcean HUD**: Live display of current speed (knots), wind direction, significant wave height ($H_s$), and Douglas Sea State.

---

## 🏛️ System Architecture

```
                                  MULTI-SENSOR DATA INGESTION
                   ┌────────────────────────────────────────────────────────┐
                   │  Copernicus Sentinel-1 SAR  │  ISRO EOS-04 C-Band SAR  │
                   │  NASA-ISRO NISAR L+S Dual   │  CMEMS Ocean Current 3D  │
                   │  ERA5 Wind / Stokes Drift   │  Live AIS Terrestrial+Sat│
                   └────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                              FASTAPI MODULAR BACKEND ENGINE
       ┌──────────────────────────────────────────────────────────────────────────────┐
       │                                                                              │
       │   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   │
       │   │  SAR Preprocessor  │   │  Bragg Wave Damping│   │ MetOcean Validator │   │
       │   │  Radiometric / Lee │──▶│  NISAR DFDI L+S    │──▶│ 3-12 m/s Wind Mask │   │
       │   └────────────────────┘   └────────────────────┘   └────────────────────┘   │
       │                                                               │              │
       │                                                               ▼              │
       │   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   │
       │   │ Cryptographic Vault│   │ Dark Vessel Engine │   │ OpenDrift Physics  │   │
       │   │ SHA-256 PDF Dossier│◀──│ AIS Blackout Sieve │◀──│ Lagrangian RK4     │   │
       │   └────────────────────┘   └────────────────────┘   └────────────────────┘   │
       │                                                                              │
       └──────────────────────────────────────────────────────────────────────────────┘
                         │                                         │
                         │ REST / WebSockets                       │ SQL / Spatial
                         ▼                                         ▼
            TACTICAL C2 OPERATIONAL FRONTEND           POSTGRESQL + POSTGIS STORAGE
       ┌──────────────────────────────────────┐   ┌───────────────────────────────────┐
       │ • Vite + React 19 + TypeScript       │   │ • Spatial AIS Trajectory PostGIS  │
       │ • In-Browser ONNX WASM Inference     │   │ • GeoJSON Drift Dispersion Cones  │
       │ • Leaflet Nautical / ECDIS Charts    │   │ • SHA-256 Evidence Manifest Store │
       │ • 4D Spatiotemporal Time Scrubber    │   │ • Incident Telemetry & Audit Logs │
       └──────────────────────────────────────┘   └───────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Component | Technology | Rationale & Purpose |
|:---|:---|:---|:---|
| **Spaceborne Radar** | Earth Observation | Copernicus Sentinel-1, ISRO EOS-04, NISAR | All-weather, day/night radar penetration through monsoons |
| **MetOcean Physics** | Environmental Data | CMEMS Marine Service, ECMWF ERA5 | High-resolution 3D current velocities and Stokes windage |
| **Edge AI Engine** | Slick Segmentation | ONNX Runtime Web (WASM), WebGPU | Sub-150ms zero-latency inference inside operator's browser |
| **Deep Learning** | Model Architecture | SpillSegNet (U-Net + ResNet-50 + ASPP + CBAM) | Multi-scale feature extraction & Bonn classification |
| **Hydrodynamics** | Trajectory Simulation | OpenDrift / OpenOil, Runge-Kutta 4th Order | Reversible particle dispersion with coastal land clamping |
| **Vessel Tracking** | AIS Telemetry | Spatiotemporal AIS parser, GeoPandas | Track reconstruction, transponder gap analysis & scoring |
| **Backend API** | Application Server | FastAPI, Python 3.11+, Pydantic v2, Uvicorn | Async performance, strict data validation, OpenAPI docs |
| **Database** | Geospatial Data Store | PostgreSQL 16 + PostGIS 3.4, SQLAlchemy | Standardized spatial queries on EPSG:4326 geometries |
| **Frontend UI** | Tactical Dashboard | React 19, TypeScript, Vite, Tailwind CSS | High-frequency rendering, dark operational C2 design system |
| **Mapping Engine** | GIS Visualizer | Leaflet, OpenSeaMap, GEBCO Bathymetry | Hydrographic charts, TSS channels, and time-scrubbing layers |
| **Evidence & Security**| Legal Chain of Custody | SHA-256 Hashing, ReportLab PDF, PyCryptodome| Non-repudiable legal dossiers compliant with Indian MSA |

---

## 📂 Repository Structure

```
.
├── backend/                        # FastAPI REST API & Core Intelligence Services
│   ├── app/
│   │   ├── api/v1/endpoints.py     # Main REST routes (detection, drift, attribution, evidence)
│   │   ├── core/                   # Security, settings, and configuration
│   │   ├── models/                 # SQLAlchemy & Pydantic domain models
│   │   ├── services/               # Satellite, OpenDrift, AIS, and forensic logic
│   │   └── main.py                 # FastAPI application factory & lifespan handler
│   ├── requirements.txt            # Python production dependencies
│   └── storage/                    # Calibrated rasters, GeoJSON slices & PDF exports
│
├── frontend/                       # Tactical C2 Maritime Intelligence Web Application
│   ├── src/
│   │   ├── components/             # Reusable UI controls, HUD cards & map overlays
│   │   │   ├── views/              # Core screens (DriftView, SARLab, AttributionView, Evidence)
│   │   │   └── ui/                 # ECDIS design tokens, buttons, dialogs, sliders
│   │   ├── services/               # API clients, marine weather service, AIS parsers
│   │   ├── App.tsx                 # Root application controller & tab router
│   │   └── main.tsx                # React DOM entry point
│   ├── public/                     # Static nautical assets, sample SAR tiles & WASM models
│   ├── package.json                # Frontend dependencies (React 19, Vite, Tailwind, Leaflet)
│   └── vite.config.ts              # Vite bundler configuration
│
├── ml/                             # AI / Deep Learning Training & Evaluation Pipeline
│   ├── models/                     # SpillSegNet U-Net PyTorch architectures
│   ├── weights/                    # Exported ONNX WASM model checkpoints
│   └── scripts/                    # DANN domain adaptation, CDF normalizers & bench tests
│
├── pipeline/                       # End-to-end orchestration chaining Stages 1–4
├── infra/                          # Deployment configurations (Vercel, Render, Docker)
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
├── PROTOTYPE_PRESENTATION_SCRIPT.txt# Comprehensive 8-minute hackathon jury pitch script
├── render.yaml                     # Render backend deployment manifest
└── vercel.json                     # Vercel frontend deployment manifest
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

## ⚖️ Regulatory & Legal Compliance

SpillSense is engineered to comply directly with international and sovereign Indian maritime law:

- **Merchant Shipping Act, 1958 (Part XIA)**: Formatted for direct invocation by the Indian Coast Guard and Directorate General of Shipping (DG Shipping) under Section 356 for maritime pollution containment and containment recovery costs.
- **MARPOL 73/78 (Annex I)**: Adheres to Regulations for the Prevention of Pollution by Oil, applying the Bonn Agreement Oil Appearance Code (BAOAC) for thickness and volume estimation.
- **UNCLOS Part XII (Articles 194, 211, 220)**: Establishes jurisdiction for enforcement within India's 200 NM Exclusive Economic Zone against foreign-flagged dark vessels.
- **WGS 84 (EPSG:4326)**: Standardized spatial projection across all ingested and exported vector layers.

---

## 🔬 Honest Limitations & Defensible Engineering

In accordance with responsible engineering practices:

1. **Probabilistic Decision Support**: SpillSense provides explainable, mathematically defensible evidence to accelerate investigations. It serves as an intelligence decision-support tool, not an autonomous judicial verdict; human maritime officer review is mandatory before punitive boarding actions.
2. **Satellite Revisit Times**: While Sentinel-1 and NISAR provide unmatched all-weather radar fidelity, constellation revisit intervals range from 1 to 6 days. Supplementing with commercial SAR (e.g. Capella/ICEYE) or airborne Coast Guard Dornier patrol radar bridges revisit gaps.
3. **AIS Broadcast Vulnerability**: Attribution scoring relies on vessels that transmit AIS signals or exhibit identifiable transponder dropouts. Completely covert "ghost vessels" with zero historical AIS footprint are flagged as unidentified radar contacts.

---

## 👥 Team BUG STALKERS

Built with dedication for **Smart India Hackathon 2026** (Problem Statement **SIH26143**):

| Name | Role | Responsibilities | GitHub |
|:---|:---|:---|:---:|
| **Krishna Kant** | **Team Lead** | Systems Architecture, Satellite STAC Ingestion & Backend Core | [@krshhh6](https://github.com/krshhh6) |
| **Tarun Agnihotri** | **Core Developer** | Full-Stack C2 Dashboard, Detection Pipeline & Geospatial UI | [@tarunagnihotri534](https://github.com/tarunagnihotri534) |
| **Adityash Srivastava** | **ML Engineer** | SpillSegNet Architecture, DANN Domain Adaptation & Training | [@adityashsrivastava](https://github.com/adityashsrivastava) |
| **Nitin Pandey** | **AI/ML Validation** | Model Quantization, ONNX WASM Edge Optimization & Benchmarks | [@nitinpandey-dev](https://github.com/nitinpandey-dev) |
| **Shubh Jaiswal** | **Backend & Pipeline** | Hydrodynamic Drift Orchestration & Evidence Cryptography | [@shubhjaiswal551](https://github.com/shubhjaiswal551) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed for **Smart India Hackathon 2026** by **Team BUG STALKERS**.
Protected under sovereign maritime data handling standards.

<div align="center">

*SpillSense — Transforming Maritime Surveillance from Reactive Scramble to Sovereign Precision.*

</div>
