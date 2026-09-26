import React, { useState } from 'react';
import type { AttributionWeights } from '../../types/dashboard';

export const AttributionView: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [evidence, setEvidence] = useState<any>(null);

  const runPhysicsEngine = async () => {
    setLoading(true);
    try {
      const payload = {
        detection: {
          timestamp: new Date().toISOString(),
          sensor_id: "S1A_IW_GRDH",
          bounding_box: {
            min_lon: 71.218, min_lat: 18.743, max_lon: 71.220, max_lat: 18.745
          },
          confidence_score: 0.94,
          estimated_length_m: 220,
          estimated_beam_m: 32,
          heading_deg: 145
        },
        ais_history: [
          {
            mmsi: "419001234",
            timestamp: new Date(Date.now() - 4 * 3600000).toISOString(),
            lon: 71.180, lat: 18.710,
            sog_knots: 4.1, cog_deg: 140,
            vessel_length_m: 215, vessel_beam_m: 30,
            vessel_type: "Crude Oil Tanker"
          }
        ]
      };

      const res = await fetch("http://localhost:8000/api/v1/evidence/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setEvidence(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div id="tab-attribution" className="tab-content visible">
      {/* PAGE HEADER */}
      <div className="page-header" style={{ paddingTop: 'var(--sp-4)' }}>
        <div>
          <div className="flex items-center gap-3">
            <div className="page-title">Vessel Attribution &amp; Sensitivity Tuner</div>
            <span className="id-tag">EXPLAINABLE ML</span>
          </div>
          <div className="page-subtitle">
            Spatiotemporal intersection between AIS trajectories and backward drift origin envelope
          </div>
        </div>
      </div>

      <div className="content-area" style={{ alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}>
          {/* WHAT-IF ATTRIBUTION WEIGHT TUNER */}
          <div className="weight-tuner">
            <div className="tuner-header">
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>science</span>
                Dynamic Physics Engine Correlation
              </span>
              <button
                className="btn btn-primary"
                onClick={runPhysicsEngine}
                disabled={loading}
                style={{ padding: '4px 12px', fontSize: 11 }}
              >
                {loading ? "Calculating Drift..." : "Run Physics Engine"}
              </button>
            </div>

            {evidence && (
              <div style={{ padding: '12px', background: '#0a0a0a', border: '1px solid #333', borderRadius: '4px', marginTop: '12px', fontSize: '11px', fontFamily: 'monospace' }}>
                <div style={{ color: '#0f0', marginBottom: '8px' }}>&gt; Backend Response Received:</div>
                <div style={{ color: '#aaa' }}>Classification: <span style={{ color: '#fff' }}>{evidence.classification}</span></div>
                <div style={{ color: '#aaa' }}>Drift Vector (dx/dy): <span style={{ color: '#fff' }}>{evidence.drift_applied?.delta_x_m?.toFixed(2)}m, {evidence.drift_applied?.delta_y_m?.toFixed(2)}m</span></div>
                <div style={{ color: '#aaa' }}>Spatial Score: <span style={{ color: '#fff' }}>{evidence.confidence?.spatial_score?.toFixed(2)}</span></div>
                <div style={{ color: '#aaa' }}>Dimension Match: <span style={{ color: '#fff' }}>{evidence.confidence?.dimension_score?.toFixed(2)}</span></div>
                <div style={{ color: '#aaa' }}>Heading Match: <span style={{ color: '#fff' }}>{evidence.confidence?.heading_score?.toFixed(2)}</span></div>
              </div>
            )}
          </div>

          {/* RANKED CANDIDATES */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>format_list_numbered</span>
                Ranked Candidate Vessels
              </span>
              <span className="text-xs text-muted">Spatiotemporal Search Window: T - 72h to 0h</span>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <div className="vessel-row bb">
                <div className="v-rank r1">1</div>
                <div className="vessel-info">
                  <div className="v-name" style={{ color: 'var(--vessel-color)' }}>{evidence?.correlated_ais?.vessel_type ? 'CRUDE ATLAS' : 'WAITING FOR PHYSICS ENGINE...'}</div>
                  <div className="v-mmsi">MMSI: 419001234 · IMO: 9412345 · Flag: India · Crude Oil Tanker</div>
                  <div className="v-meta">
                    CPA: {evidence?.drift_applied?.delta_x_m ? `${(evidence.drift_applied.delta_x_m / 1852).toFixed(2)} nm` : '--- nm'} from envelope centroid · Speed: 4.1 kn
                  </div>
                </div>
                <div className="score-col">
                  <div className="score-val sc-h">{evidence?.confidence?.overall_confidence?.toFixed(2) || '0.00'}</div>
                  <div className="score-lbl">Attribution Score</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}>
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>gavel</span>
                Legal &amp; Evidentiary Disclaimer
              </span>
            </div>
            <div className="panel-body">
              <div className="prob-note">
                <strong>Important Legal Notice:</strong> Spill Sense attribution scores reflect statistical correlation and
                hydrodynamic consistency. Under MARPOL 73/78 Annex I and Section 356 of the Indian Merchant Shipping Act
                1958, physical oil sampling and maritime inspection by Indian Coast Guard / Mercantile Marine Department
                (MMD) officers remain mandatory for statutory enforcement.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
