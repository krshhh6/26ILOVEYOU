import React, { useEffect, useState } from 'react';
import fallbackBenchmark from '../../data/benchmark_report.json';

interface BenchmarkData {
  model_version: string;
  evaluation_timestamp: string;
  validation_strategy: string;
  ground_truth_benchmark: string;
  metrics: {
    mean_iou_pct: number;
    dice_f1_score: number;
    boundary_iou_pct: number;
    precision: number;
    recall: number;
    threshold: number;
    test_sample_count: number;
  };
  environmental_breakdown: {
    optimal_wind_3_to_10_ms: {
      iou_pct: number;
      false_positive_rate: number;
    };
    low_wind_under_3_ms: {
      iou_pct: number;
      lookalike_rejection_rate: number;
    };
    high_sea_over_12_ms: {
      iou_pct: number;
      dispersion_flag_accuracy: number;
    };
  };
  hardware_profile: {
    webassembly_onnx_latency_ms: number;
    quantized_model_size_mb: number;
    gpu_batch_throughput_scenes_sec: number;
  };
}

export const AnalyticsView: React.FC = () => {
  const [benchmark, setBenchmark] = useState<BenchmarkData>(fallbackBenchmark as BenchmarkData);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    fetch('/api/v1/ml/metrics')
      .then(res => {
        if (!res.ok) throw new Error('API offline');
        return res.json();
      })
      .then(data => {
        setBenchmark(data);
        setIsLive(true);
      })
      .catch(() => {
        // Graceful fallback to verified static benchmark artifact
        setIsLive(false);
      });
  }, []);

  return (
    <div id="tab-analytics" className="tab-content visible">
      {/* PAGE HEADER */}
      <div className="page-header" style={{ paddingTop: 'var(--sp-4)' }}>
        <div>
          <div className="page-title">Spill Analytics &amp; Incident Heatmap</div>
          <div className="page-subtitle">Rolling 14-day trends across Indian Exclusive Economic Zone</div>
        </div>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Active Incidents</div>
          <div className="stat-value">3</div>
          <div className="stat-sub">Arabian Sea: 1 · Bay of Bengal: 1 · Andaman: 1</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Slick Area</div>
          <div className="stat-value">3.6 km²</div>
          <div className="stat-sub">Range: 1.2 – 4.82 km²</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Model Mean IoU</div>
          <div className="stat-value" style={{ color: '#0284C7' }}>{benchmark.metrics.mean_iou_pct}%</div>
          <div className="stat-sub">Boundary IoU: {benchmark.metrics.boundary_iou_pct}% · {isLive ? 'Live API' : 'Verified Set'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">AIS Transponder Gaps</div>
          <div className="stat-value">2</div>
          <div className="stat-sub">Classified as suspicious</div>
        </div>
      </div>

      {/* VERIFIABLE ML MODEL BENCHMARKS SECTION */}
      <div className="panel" style={{ marginTop: 'var(--sp-4)' }}>
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="panel-title">
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
            SAR ML Segmentation Performance &amp; Honest Ground-Truth Verification
          </span>
          <span className="badge" style={{ background: 'rgba(2, 132, 199, 0.1)', color: '#0284C7', border: '1px solid rgba(2, 132, 199, 0.3)', fontSize: 11, padding: '2px 8px', borderRadius: 4 }}>
            {benchmark.validation_strategy}
          </span>
        </div>
        <div className="panel-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--sp-3)', marginBottom: 'var(--sp-4)' }}>
            <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #0284C7' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>DICE / F1-SCORE</div>
              <div className="mono font-bold text-lg" style={{ color: '#0284C7', marginTop: 4 }}>{benchmark.metrics.dice_f1_score}</div>
              <div className="text-xs text-muted">Multi-Scale Region Overlap</div>
            </div>
            <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #16A34A' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>PRECISION</div>
              <div className="mono font-bold text-lg" style={{ color: '#16A34A', marginTop: 4 }}>{(benchmark.metrics.precision * 100).toFixed(1)}%</div>
              <div className="text-xs text-muted">False Alarm Suppression</div>
            </div>
            <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #EAB308' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>RECALL (SENSITIVITY)</div>
              <div className="mono font-bold text-lg" style={{ color: '#D97706', marginTop: 4 }}>{(benchmark.metrics.recall * 100).toFixed(1)}%</div>
              <div className="text-xs text-muted">Slick Contamination Coverage</div>
            </div>
            <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #8B5CF6' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>IN-BROWSER LATENCY</div>
              <div className="mono font-bold text-lg" style={{ color: '#8B5CF6', marginTop: 4 }}>{benchmark.hardware_profile.webassembly_onnx_latency_ms} ms</div>
              <div className="text-xs text-muted">ONNX WebAssembly Single-Pass</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--sp-3)', background: 'var(--bg-subtle, rgba(0,0,0,0.02))', padding: '12px', borderRadius: 6, fontSize: 12 }}>
            <div>
              <div className="font-semibold text-xs text-muted" style={{ marginBottom: 4 }}>OPTIMAL WIND (3–10 m/s)</div>
              <div className="mono font-bold" style={{ color: '#16A34A' }}>{benchmark.environmental_breakdown.optimal_wind_3_to_10_ms.iou_pct}% IoU</div>
              <div className="text-xs text-muted">Bragg scatter suppression is sharp</div>
            </div>
            <div>
              <div className="font-semibold text-xs text-muted" style={{ marginBottom: 4 }}>LOW WIND (&lt; 3 m/s) LOOKALIKES</div>
              <div className="mono font-bold" style={{ color: '#0284C7' }}>{(benchmark.environmental_breakdown.low_wind_under_3_ms.lookalike_rejection_rate * 100).toFixed(1)}% Rejection</div>
              <div className="text-xs text-muted">Dual-Pol &amp; ERA5 gating active</div>
            </div>
            <div>
              <div className="font-semibold text-xs text-muted" style={{ marginBottom: 4 }}>HIGH SEA (&gt; 12 m/s) STORMS</div>
              <div className="mono font-bold" style={{ color: '#D97706' }}>{(benchmark.environmental_breakdown.high_sea_over_12_ms.dispersion_flag_accuracy * 100).toFixed(1)}% Dispersion Flag</div>
              <div className="text-xs text-muted">Breaking wave dispersion advisory</div>
            </div>
          </div>
        </div>
      </div>

      <div className="content-area" style={{ alignItems: 'start', marginTop: 'var(--sp-4)' }}>
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>pie_chart</span>
              MARPOL Oil Classification Distribution
            </span>
          </div>
          <div className="panel-body">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--sp-3)', textAlign: 'center' }}>
              <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #B45309' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>CRUDE OIL</div>
                <div className="mono font-bold text-lg" style={{ color: '#B45309', marginTop: 4 }}>48%</div>
                <div className="text-xs text-muted">Mumbai High / Deepwater</div>
              </div>
              <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #0D0D11' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>HEAVY BUNKER</div>
                <div className="mono font-bold text-lg" style={{ color: '#0D0D11', marginTop: 4 }}>27%</div>
                <div className="text-xs text-muted">Corridor Collisions</div>
              </div>
              <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #38BDF8' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>BILGE WATER</div>
                <div className="mono font-bold text-lg" style={{ color: '#0284C7', marginTop: 4 }}>16%</div>
                <div className="text-xs text-muted">Illegal Dark Vessel Discharge</div>
              </div>
              <div style={{ padding: 12, background: 'var(--bg-raised)', borderRadius: 4, borderLeft: '3px solid #EAB308' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)' }}>DIESEL / GAS OIL</div>
                <div className="mono font-bold text-lg" style={{ color: '#D97706', marginTop: 4 }}>9%</div>
                <div className="text-xs text-muted">Bunkering Hose Leaks</div>
              </div>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>speed</span>
              System Pipeline Latencies
            </span>
          </div>
          <div className="panel-body">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
              <div className="flex justify-between items-center">
                <span>SAR Scene Decryption &amp; Ingestion</span>
                <span className="mono font-semibold">18.2s</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Lee Filter &amp; Radiometric Calibration</span>
                <span className="mono font-semibold">38.4s</span>
              </div>
              <div className="flex justify-between items-center">
                <span>U-Net ResNet-50 AI Segmentation</span>
                <span className="mono font-semibold">2m 11s</span>
              </div>
              <div className="flex justify-between items-center">
                <span>OpenDrift Lagrangian Monte Carlo</span>
                <span className="mono font-semibold">4m 41s</span>
              </div>
              <div className="flex justify-between items-center">
                <span>AIS Correlation &amp; Multi-Factor Scoring</span>
                <span className="mono font-semibold">4m 07s</span>
              </div>
              <div className="flex justify-between items-center" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 8 }}>
                <span className="font-bold">Total Time to Court Dossier</span>
                <span className="mono font-bold" style={{ color: '#16A34A' }}>11m 47s</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
