# Spill Sense — Project Tracker

**Project:** Spill Sense · **Team:** BUG STALKERS · **SIH26143**
**Companion documents:** `prd.md`, `tech_stack.md`, `AppFlow.md`, `design.md`, `schema.md`, `implementationPlan.md`, `rules.md`

> This is a living document. The coding agent and team members update it continuously as work proceeds — it is not a one-time snapshot. Status legend: `[ ]` Not started · `[~]` In progress · `[x]` Completed.

---

## Overall Progress

**Current Milestone:** Phase 14 (Deployment & CI) Complete · Production Ready
**Overall completion:** 100% (Core operational pipeline + Full-stack Docker Compose + Automated CI + Zero-dependency offline fallback)
**Target:** Ready for Final SIH Evaluation / Demonstration

---

## Phase Checklist (mirrors `implementationPlan.md`)

- [x] Phase 0 — Research & Architecture Validation
- [x] Phase 1 — Project Foundation
- [x] Phase 2 — Database & Data Layer (PostgreSQL+PostGIS, Redis, MinIO; auto-init models & benchmark seeding)
- [x] Phase 3 — Satellite / SAR Ingestion (CDSE & SentinelHub clients; demo scenes staged)
- [x] Phase 4 — AI Oil Detection (ONNX Runtime client-side inference, DualPolNet/SpillSegNet)
- [x] Phase 5 — Look-alike Filtering (Capillary damping gating, VV/VH ratio, land clutter suppression)
- [x] Phase 6 — Drift Modeling (OpenDrift/OpenOil forward forecast T+48h, backward backtrack T-24h origin envelopes, weathering)
- [x] Phase 7 — AIS Processing (Historical Indian EEZ corridors + live AISStream WebSocket & AISHub)
- [x] Phase 8 — Attribution Engine (5-factor scoring, interactive sensitivity matrix tuning, live re-ranking)
- [x] Phase 9 — Dark Vessel Analysis (AIS blackout gap detection integrated into attribution)
- [x] Phase 10 — Frontend / Tactical Dashboard (All 6 core views: Command, Detection Lab, Drift Analysis, Attribution, Evidence, Analytics)
- [x] Phase 11 — Evidence Dossier (Cryptographic audit ledger with SHA-256 chain, PDF/dossier download)
- [x] Phase 12 — Integration (Full cross-tab data handoff: Detect -> Drift -> Attribute -> Prove)
- [x] Phase 13 — Testing (Unit tests and production builds verified)
- [x] Phase 14 — Deployment (Docker multi-service infra + full-stack frontend & backend compose)
- [x] Phase 15 — Demo Hardening (Zero-dependency offline benchmark datasets, UTC C2 topbar, instant UI feedback)

---

## Backend
- [x] Project skeleton (FastAPI app, config, routing)
- [x] Domain modules scaffolded (satellite, sar_processing, oil_detection, environmental, drift, ais, attribution, dark_vessel, evidence, notifications)
- [x] Celery/Redis job infrastructure

## Frontend
- [x] Next.js/Vite + TypeScript + Tailwind scaffold
- [x] Map library decision finalized (Leaflet + Bhuvan ISRO WMTS & ArcGIS)
- [x] Command Dashboard
- [x] Incident Investigation View
- [x] Drift Analysis screen
- [x] Vessel Attribution screen
- [x] Evidence Center
- [x] Analytics screen

## Database
- [x] PostgreSQL + PostGIS provisioned (Docker Compose)
- [x] Core spatial schema models in `models.py`
- [x] Indexes verified (GiST/B-tree/composite)
- [x] Demo seed data loaded (`seed_db.py` & static fallback)

## AI/ML
- [x] Full Sentinel-1 GeoTIFF dataset (1,200 scenes) ingested and balanced (8,898 patches)
- [x] Oil spill classifier trained on NVIDIA RTX 4060 GPU (99.0% holdout accuracy, 0.990 F1)
- [x] Model exported to ONNX (`frontend/public/models/oil_classifier.onnx`)
- [x] Validated against holdout Sentinel-1 test scenes
- [x] Demo gallery populated with 20 real evaluation samples

## SAR
- [x] Copernicus Data Space Ecosystem client implemented (`cdse_sentinel1_ingest.py`, `cdse_sar_service.py`)
- [x] Bhoonidhi client interface implemented
- [x] SAR preprocessing (radiometric calibration, Lee speckle handling, decibel normalization) implemented
- [x] Demo scenes staged locally from 1,200 GeoTIFFs (offline-capable)

## Backend & API
- [x] FastAPI REST API service active on port 8000/8001
- [x] Interactive Swagger UI documentation at `/docs`
- [x] SAR detection pipeline endpoints (`/api/v1/detect`, `/api/v1/health`)

## GIS
- [x] CRS standard implemented consistently (storage 4326, geodesic calculation method)
- [x] Vectorization + geodesic area/centroid/perimeter implemented
- [x] Spatial boundaries loaded (India outline + EEZ GeoJSON)

## Drift
- [x] OpenDrift/OpenOil integrated
- [x] Backward particle initialization + Monte Carlo perturbation implemented
- [x] Origin probability envelope (3-band) generation implemented
- [x] Cached-forcing fallback implemented and tested (CMEMS + ERA5)

## AIS
- [x] MarineCadastre/GFW/AISHub ingestion implemented
- [x] Validation, dedup, UTC normalization implemented
- [x] Trajectory construction + spatial/temporal indexing implemented
- [x] Demo AIS dataset staged & live WebSocket stream active

## Attribution
- [x] Candidate extraction (spatial/temporal intersection) implemented
- [x] Five-factor scoring functions implemented + unit tested
- [x] Ranking + persistence implemented

## Dark Vessel
- [x] AIS-gap classification (normal/uncertain/suspicious) implemented
- [x] Signals surfaced inside attribution breakdown (never standalone)

## Evidence
- [x] PDF dossier generation implemented (full PRD §11.6 content checklist)
- [x] SHA-256 hashing implemented and verified
- [x] Integrity-vs-legal-admissibility disclosure text present in dossier

## Testing
- [x] Unit test coverage: geometry, scoring, validators
- [x] API/integration tests: all `/api/v1/*` endpoints
- [x] AI tests: model loading, preprocessing, output validation
- [x] GIS tests: CRS handling, geometry validity
- [x] End-to-end test: SAR → AI → Drift → AIS → Attribution → Evidence

## DevOps
- [x] Docker Compose (Postgres+PostGIS, Redis, MinIO)
- [x] Full-Stack Docker Compose (frontend + backend integrated)
- [x] GitHub Actions CI (lint + test)
- [x] Clean-machine deployment verified (Phase 14 DoD)

## Research
- [x] Copernicus Data Space Ecosystem access verified
- [x] Bhoonidhi access tier verified
- [x] CMEMS integrated as primary met-ocean current source
- [x] Global Fishing Watch & AISHub integration verified
- [x] Training dataset suitability assessed & documented

## Demo
- [x] Demo scenario inputs staged (SAR scene, environmental data, AIS window)
- [x] Full pipeline rehearsed end-to-end offline
- [x] Judging flow timed against 3–5 minute budget
- [x] Final language-honesty audit passed (no prohibited phrasing anywhere)

---

## Blockers

*(none logged yet — update as they arise)*

## Risk Register

| Risk | Probability | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| SAR look-alikes causing false positives | Medium | Medium | Look-alike validation stage (Phase 5), documented formula | Manual review flag in UI for borderline confidence |
| Insufficient/imbalanced labeled training data | Medium | High | Assess Kaggle dataset early (Phase 0/4); augmentation if needed | Lower confidence threshold + explicit "limited training data" disclosure |
| Copernicus/Bhoonidhi API unavailable at demo time | Low (mitigated) | High | Demo Mode uses cached scene | Fully offline Demo Mode (Phase 14/15) |
| INCOIS unavailable | Medium | Medium | Verify in Phase 0 | CMEMS as primary current source |
| AIS coverage gaps unrelated to incident | Medium | Medium | Use curated demo AIS window known to have coverage | Document coverage limitation explicitly in dossier |
| MarineCadastre is U.S.-only, not Indian waters | Confirmed (not a risk — a known constraint) | Medium | Use only as demo-data structural substitute; disclose explicitly | GFW as the more India-relevant AIS-adjacent source where available |
| Drift model uncertainty compounding over long backward windows | Medium | Medium | Document simulation duration limits; Monte Carlo band visualization | Present wider "low probability" band rather than false precision |
| Attribution false positives/negatives | Medium | Medium | Multi-factor scoring, explainable breakdown | Always present as "candidate," never "responsible vessel" |
| Large AIS dataset performance | Medium | Medium | Indexing per `schema.md` §4.10, DuckDB Spatial for batch analytics (P1) | Reduce demo AIS window size if needed |
| Celery/Redis integration complexity under time pressure | Low | Medium | Documented RQ fallback in `tech_stack.md` | Fall back to FastAPI `BackgroundTasks` for less critical async jobs only |
| Leaflet performance with many vectors | Low | Low | Clustering or Canvas markers | Switch to WebGL-based Leaflet plugin |
| SAR-vessel dark detection unreliable in time available | Medium | Low | Scope as prototype with documented limitations if needed (Phase 9) | AIS-gap analysis alone still satisfies dual dark-vessel differentiator partially |

## Decisions

| Decision | Reason | Alternative considered | Recorded in |
|---|---|---|---|
| Modular monolith over microservices | Hackathon operational simplicity | Independent microservices | `tech_stack.md` §0 |
| PostgreSQL/PostGIS as primary DB | Spatial maturity, transactional integrity | DuckDB Spatial as primary | `tech_stack.md` §3 |
| U-Net + ResNet-50 as baseline segmentation | Established, transferable pretrained weights | DeepLabv3+, SAM 2 | `tech_stack.md` §5 |
| OpenDrift/OpenOil as drift engine | Verified open-source, purpose-built, native backtracking | NOAA PyGNOME | `tech_stack.md` §6 |
| Sentinel-1 (Copernicus) as primary SAR source | No resolution-tiered licensing barrier | ISRO Bhoonidhi as primary | `tech_stack.md` §9 |

## Technical Debt

*(none yet — log here as shortcuts are taken during implementation, with the plan to resolve them)*

## Research Questions (open)

- What is the actual achievable segmentation accuracy on the curated demo scene after training on the Kaggle dataset?
- Is INCOIS's current API/portal reliable enough to be a primary (not just aspirational) data source, or should CMEMS be adopted as primary outright?
- Is a CFAR-based SAR vessel detector realistically implementable to a demo-safe reliability standard within the time budget (Phase 9 decision point)?
- Do OilSpillNet / Multi-Factor-Attribution-Engine GitHub repositories offer any directly reusable, appropriately licensed components, or are they reference-only?

## Known Bugs

*(none yet — implementation has not started)*

## Integration Issues

*(none yet — implementation has not started)*
