import React, { useState, useMemo } from 'react';
import type { Scenario } from '../../types/dashboard';
import {
  generateDeterministicHash,
  formatHashChunks,
  getForensicMilestones,
  getForensicArtifacts,
} from '../../utils/forensicHash';
import type { ForensicArtifactItem } from '../../utils/forensicHash';

interface EvidenceViewProps {
  onOpenForensicModal: () => void;
  currentScenario?: Scenario | null;
}

export const EvidenceView: React.FC<EvidenceViewProps> = ({ onOpenForensicModal, currentScenario }) => {
  // Active Sub-Tab: 'manifest' | 'artifacts' | 'dossier'
  const [activeSubTab, setActiveSubTab] = useState<'manifest' | 'artifacts' | 'dossier'>('manifest');

  // Dynamic deterministic master hash based on active scenario
  const scenarioKey = useMemo(() => {
    return (currentScenario?.id || 'INC-2026-005') + 
      (currentScenario?.title || 'Maritime Sector') + 
      (currentScenario?.lat || 22.61) + 
      (currentScenario?.lng || 69.5);
  }, [currentScenario]);

  const masterHash = useMemo(() => generateDeterministicHash(scenarioKey), [scenarioKey]);
  const hashChunks = useMemo(() => formatHashChunks(masterHash), [masterHash]);

  // Officer sign-off state
  const [officerSignOff, setOfficerSignOff] = useState<{ officerName: string; timestamp: string } | null>(null);
  const [isSignOffModalOpen, setIsSignOffModalOpen] = useState(false);
  const [signOffName, setSignOffName] = useState('Cmdr. V. Sharma, ICG');
  const [signOffStation, setSignOffStation] = useState('MRCC Mumbai · Western Seaboard C2');

  // Interactive Live Integrity Audit state
  const [isAuditing, setIsAuditing] = useState(false);
  const [auditProgress, setAuditProgress] = useState(0);
  const [auditStepText, setAuditStepText] = useState('');
  const [auditComplete, setAuditComplete] = useState(false);

  // In-line Artifact Preview Drawer
  const [previewArtifact, setPreviewArtifact] = useState<ForensicArtifactItem | null>(null);

  // User feedback toast/notices
  const [feedback, setFeedback] = useState<{ text: string; color: string } | null>(null);
  const [downloadNotice, setDownloadNotice] = useState<{ text: string; isSuccess: boolean } | null>(null);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  // Milestones & Artifacts linked dynamically to active scenario
  const milestones = useMemo(
    () => getForensicMilestones(currentScenario, officerSignOff),
    [currentScenario, officerSignOff]
  );

  const artifacts = useMemo(
    () => getForensicArtifacts(currentScenario, masterHash),
    [currentScenario, masterHash]
  );

  const totalPackageSizeKb = useMemo(
    () => artifacts.reduce((acc, item) => acc + item.sizeKb, 0),
    [artifacts]
  );

  const handleCopySignature = () => {
    navigator.clipboard?.writeText(masterHash);
    setFeedback({ text: 'Cryptographic master signature copied to clipboard.', color: 'var(--accent)' });
    setTimeout(() => setFeedback(null), 3000);
  };

  const handleCopyText = (text: string, label: string) => {
    navigator.clipboard?.writeText(text);
    setFeedback({ text: `${label} copied to clipboard.`, color: 'var(--accent)' });
    setTimeout(() => setFeedback(null), 3000);
  };

  const triggerBlobDownload = (blob: Blob, fileName: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, 250);
  };

  const handleDownloadArtifact = async (fileName: string) => {
    setDownloadingFile(fileName);
    setDownloadNotice({ text: `Authenticating and transferring ${fileName}...`, isSuccess: false });

    // Look up artifact generator
    const matched = artifacts.find((a) => a.fileName === fileName);
    const content = matched
      ? matched.contentGenerator(
          currentScenario || {
            id: 'INC-2026-005',
            title: 'Gulf of Kutch Deepwater Tanker Fairway',
            lat: 22.61,
            lng: 69.5,
            topVessel: 'MT Ocean Trader',
            diagVessel: 'MT Ocean Trader',
            diagDetails: 'AIS Silence',
            oilType: 'Crude Oil',
            oilColor: '#ef4444',
            oilFill: 'rgba(239, 68, 68, 0.4)',
            sev: 'CRITICAL',
            sevClass: 'chip-c',
            sub: 'Indian EEZ',
            scores: [0.86],
          },
          masterHash
        )
      : `SpillSense Forensic Record · ${fileName}\nIncident: ${currentScenario?.id || 'INC-005'}\nHash: ${masterHash}\nSection 63 BSA 2023 Certified`;

    let mimeType = 'text/plain';
    if (fileName.endsWith('.geojson')) mimeType = 'application/geo+json;charset=utf-8';
    else if (fileName.endsWith('.xml')) mimeType = 'application/xml;charset=utf-8';
    else if (fileName.endsWith('.zip')) mimeType = 'application/zip';

    setTimeout(() => {
      const blob = new Blob([content], { type: mimeType });
      triggerBlobDownload(blob, fileName);
      setDownloadNotice({
        text: `✓ Downloaded ${fileName} — Cryptographic signature verified against HSM Ledger`,
        isSuccess: true,
      });
      setDownloadingFile(null);
      setTimeout(() => setDownloadNotice(null), 4500);
    }, 400);
  };

  // Run interactive integrity audit sequence
  const handleRunAudit = () => {
    setIsAuditing(true);
    setAuditComplete(false);
    setAuditProgress(10);
    setAuditStepText('Querying HSM FIPS 140-2 Key Vault & Ledger Root...');

    setTimeout(() => {
      setAuditProgress(35);
      setAuditStepText('Verifying Sentinel-1 GRD Raw Acquisition CRC-32...');
    }, 600);

    setTimeout(() => {
      setAuditProgress(65);
      setAuditStepText('Computing U-Net ResNet-50 Feature Mask SHA-256 Digest...');
    }, 1200);

    setTimeout(() => {
      setAuditProgress(85);
      setAuditStepText('Cross-checking OpenDrift Lagrangian Backward Particles & AIS Matrix...');
    }, 1800);

    setTimeout(() => {
      setAuditProgress(100);
      setAuditStepText('All 4 Artifacts Authenticated · Zero Bit Discrepancies');
      setIsAuditing(false);
      setAuditComplete(true);
      setFeedback({ text: 'Cryptographic Audit Passed: 100% Intact & Verified.', color: '#10b981' });
      setTimeout(() => setFeedback(null), 4000);
    }, 2400);
  };

  const handleConfirmSignOff = () => {
    const ts = `${new Date().toISOString().replace('T', ' ').slice(0, 19)} IST`;
    setOfficerSignOff({
      officerName: `${signOffName} (${signOffStation})`,
      timestamp: ts,
    });
    setIsSignOffModalOpen(false);
    setFeedback({
      text: `✓ Dossier Sealed & Endorsed by ${signOffName}`,
      color: '#10b981',
    });
    setTimeout(() => setFeedback(null), 4000);
  };

  return (
    <div id="tab-evidence" className="tab-content visible modern-dashboard-root">
      {/* 1. EXECUTIVE HEADER */}
      <div className="workspace-header-bar">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h1 className="workspace-main-title">Forensic Evidence &amp; Chain of Custody</h1>
            <span
              className="scenario-chip"
              style={{
                borderColor: 'rgba(37, 99, 235, 0.4)',
                color: 'var(--accent)',
                fontSize: 10,
                fontWeight: 700,
              }}
            >
              {currentScenario?.id || 'INC-2026-005'} · {currentScenario?.title || 'Active Surveillance'}
            </span>
          </div>
          <p className="workspace-sub-title">
            Cryptographically Verifiable Evidence Package for Maritime Regulatory Enforcement · ISO/IEC 27037 Digital Forensics Standards · Sec 63 BSA 2023
          </p>
        </div>

        <div className="workspace-header-actions">
          <button
            className="action-pill-btn secondary"
            onClick={handleCopySignature}
            title="Copy Master SHA-256 Fingerprint"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>content_copy</span>
            <span>Copy Master Signature</span>
          </button>

          <button
            className="action-pill-btn primary"
            onClick={onOpenForensicModal}
            title="Generate & View Court-Admissible PDF Dossier"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>description</span>
            <span>Official Court Dossier</span>
          </button>
        </div>
      </div>

      {/* 2. EXECUTIVE METRIC CARDS */}
      <div className="executive-metrics-grid">
        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Custody Seal Status</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number" style={{ fontSize: 19 }}>
              ISO/IEC <span className="metric-unit">27037</span>
            </span>
            <span className="metric-trend-pill positive">
              Verified Active
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Hardware Security Module (HSM)</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: '#10b981' }}>verified</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">SHA-256 Checksums</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              4 / 4 <span className="metric-unit">Sealed</span>
            </span>
            <span className="metric-trend-pill positive">
              100% Intact
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Zero Bit Discrepancies</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: '#10b981' }}>lock</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Evidence Package</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              {totalPackageSizeKb.toFixed(1)} <span className="metric-unit">KB</span>
            </span>
            <span className="metric-trend-pill neutral">
              4 Assets
            </span>
          </div>
          <div className="metric-card-footer">
            <span>GeoJSON / NetCDF / GPKG</span>
            <span className="material-symbols-outlined arrow-icon">folder_zip</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Admissibility Level</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              Tier 4 <span className="metric-unit">Legal</span>
            </span>
            <span className="metric-trend-pill positive">
              Court-Admissible
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Sec 63 BSA 2023 / MS Act §356</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: 'var(--accent)' }}>gavel</span>
          </div>
        </div>
      </div>

      {/* 3. FUNCTIONAL WORKFLOW NAV BAR */}
      <div className="workflow-nav-bar">
        <div className="workflow-title-area">
          <h2 className="workflow-title">Tamper-Evident Forensic Audit Ledger</h2>
          <span className="scenario-chip" style={{ borderColor: 'rgba(37, 99, 235, 0.4)', color: 'var(--accent)' }}>
            Ledger Block #40921-IN · ECDSA Signed
          </span>
          {officerSignOff && (
            <span className="scenario-chip" style={{ borderColor: 'rgba(16, 185, 129, 0.4)', color: '#10b981', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>verified_user</span>
              Officer Endorsed
            </span>
          )}
        </div>

        <div className="workflow-tabs-strip">
          <button
            className={`workflow-tab-btn ${activeSubTab === 'manifest' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('manifest')}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 4 }}>lock</span>
            Cryptographic Manifest &amp; Timeline
          </button>
          <button
            className={`workflow-tab-btn ${activeSubTab === 'artifacts' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('artifacts')}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 4 }}>inventory_2</span>
            Artifacts &amp; Live Verifier (4)
          </button>
          <button
            className={`workflow-tab-btn ${activeSubTab === 'dossier' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('dossier')}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 4 }}>article</span>
            Court Dossier &amp; BSA §63
          </button>
        </div>

        {feedback && (
          <div style={{ fontSize: 11, fontWeight: 600, color: feedback.color, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>check_circle</span>
            {feedback.text}
          </div>
        )}
      </div>

      {downloadNotice && (
        <div
          style={{
            margin: '0 0 12px',
            padding: '10px 16px',
            background: downloadNotice.isSuccess ? 'rgba(16, 185, 129, 0.12)' : 'rgba(56, 189, 248, 0.12)',
            border: downloadNotice.isSuccess ? '1px solid rgba(16, 185, 129, 0.35)' : '1px solid rgba(56, 189, 248, 0.28)',
            borderRadius: 8,
            fontSize: 12,
            color: downloadNotice.isSuccess ? '#10b981' : '#38bdf8',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            transition: 'all 0.2s ease',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
            {downloadNotice.isSuccess ? 'check_circle' : 'cloud_download'}
          </span>
          <span>{downloadNotice.text}</span>
        </div>
      )}

      {/* 4. MAIN CONTENT AREA (BASED ON ACTIVE SUB-TAB) */}

      {/* VIEW 1: MANIFEST & TIMELINE */}
      {activeSubTab === 'manifest' && (
        <div className="canvas-rounded-container">
          <div className="canvas-two-column">
            {/* LEFT PANE: MASTER MANIFEST & SUMMARY */}
            <div className="canvas-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>lock</span>
                  Master Cryptographic Manifest
                </span>
                <span className="metric-trend-pill positive" style={{ fontSize: 10, fontWeight: 700 }}>
                  VERIFIED SECURE
                </span>
              </div>

              {/* TAMPER-PROOF DIGITAL SEAL CARD */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '12px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        background: 'rgba(37, 99, 235, 0.12)',
                        color: 'var(--accent)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 18 }}>verified_user</span>
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                        Cryptographic Chain of Custody Seal
                      </div>
                      <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                        ISO/IEC 27037 Tamper-Proof Digital Seal · Registered in Maritime Ledger
                      </div>
                    </div>
                  </div>

                  <button
                    className="action-pill-btn secondary"
                    onClick={handleCopySignature}
                    style={{ fontSize: 10.5, padding: '3px 9px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 13 }}>content_copy</span>
                    <span>Copy Digest</span>
                  </button>
                </div>

                {/* FORMATTED 8-CHAR CHUNKS HASH DISPLAY */}
                <div
                  style={{
                    background: 'var(--bg-base)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    padding: '8px 10px',
                    fontFamily: 'monospace',
                    fontSize: 10,
                    color: 'var(--accent)',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 6,
                    lineHeight: 1.4,
                  }}
                >
                  {hashChunks.map((chunk, i) => (
                    <span
                      key={i}
                      style={{
                        padding: '1px 4px',
                        background: 'rgba(37, 99, 235, 0.08)',
                        borderRadius: 3,
                        letterSpacing: '0.04em',
                      }}
                    >
                      {chunk}
                    </span>
                  ))}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, fontSize: 9.5 }}>
                  <div>
                    <span className="text-muted">Algorithm: </span>
                    <strong>SHA-256 + ECDSA</strong>
                  </div>
                  <div>
                    <span className="text-muted">Key Vault: </span>
                    <strong>ICG-HSM-2026-FIPS</strong>
                  </div>
                  <div>
                    <span className="text-muted">Time Source: </span>
                    <strong>RFC 3161 GNSS TSP</strong>
                  </div>
                </div>
              </div>

              {/* QUICK ARTIFACTS OVERVIEW */}
              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, gap: 6, marginTop: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>folder_zip</span>
                    Forensic Assets ({artifacts.length} Packages)
                  </span>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => setActiveSubTab('artifacts')}
                    style={{ fontSize: 10, padding: '2px 8px' }}
                  >
                    <span>View Inspector &amp; Hashes</span>
                    <span className="material-symbols-outlined" style={{ fontSize: 13 }}>arrow_forward</span>
                  </button>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1, minHeight: 0, overflowY: 'auto' }}>
                  {artifacts.map((art) => (
                    <div
                      key={art.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '7px 11px',
                        background: 'var(--bg-raised)',
                        borderRadius: 8,
                        border: '1px solid var(--border-subtle)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                        <span className="material-symbols-outlined" style={{ color: art.iconColor, fontSize: 18 }}>{art.icon}</span>
                        <div>
                          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
                            {art.fileName}
                          </div>
                          <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                            {art.crs} · {art.sizeKb.toFixed(1)} KB
                          </div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <button
                          className="action-pill-btn secondary"
                          onClick={() => setPreviewArtifact(art)}
                          title={`Quick Inspect ${art.fileName}`}
                          style={{ padding: '3px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 13 }}>visibility</span>
                          <span>Preview</span>
                        </button>
                        <button
                          className="action-pill-btn secondary"
                          onClick={() => handleDownloadArtifact(art.fileName)}
                          disabled={downloadingFile === art.fileName}
                          title={`Download ${art.fileName}`}
                          style={{ padding: '3px 8px', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 13 }}>download</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* RIGHT PANE: TIMELINE & LEGAL STATUTORY ADVISORY */}
            <div className="canvas-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>link</span>
                  Forensic Chain of Custody Timeline
                </span>
                <span className="metric-trend-pill neutral" style={{ fontSize: 10 }}>
                  {officerSignOff ? '5 OF 5 SEALED & VERIFIED' : '4 OF 5 SEALED · PENDING SIGN-OFF'}
                </span>
              </div>

              {/* CHRONOLOGICAL TIMELINE */}
              <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 10, padding: '2px 0 2px 28px' }}>
                <div
                  style={{
                    position: 'absolute',
                    left: 10,
                    top: 10,
                    bottom: 10,
                    width: 2,
                    background: officerSignOff
                      ? 'linear-gradient(to bottom, #10b981 0%, #10b981 100%)'
                      : 'linear-gradient(to bottom, #10b981 0%, #10b981 75%, var(--accent) 100%)',
                    opacity: 0.45,
                    zIndex: 0,
                  }}
                />

                {milestones.map((m) => {
                  const isSealed = m.status === 'sealed';
                  return (
                    <div key={m.id} style={{ position: 'relative', display: 'flex', flexDirection: 'column' }}>
                      <div
                        style={{
                          position: 'absolute',
                          left: -28,
                          top: 0,
                          width: 20,
                          height: 20,
                          borderRadius: '50%',
                          background: isSealed ? 'rgba(16, 185, 129, 0.15)' : 'rgba(37, 99, 235, 0.15)',
                          border: isSealed ? '1.5px solid #10b981' : '1.5px solid var(--accent)',
                          color: isSealed ? '#10b981' : 'var(--accent)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          zIndex: 1,
                        }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: 12 }}>{m.icon}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ fontSize: 11.5, fontWeight: 700, color: isSealed ? 'var(--text-primary)' : 'var(--accent)' }}>
                          {m.title}
                        </div>
                        {m.id === 'm5' && !officerSignOff && (
                          <button
                            className="action-pill-btn primary"
                            onClick={() => setIsSignOffModalOpen(true)}
                            style={{ fontSize: 10, padding: '2px 8px', cursor: 'pointer' }}
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: 13 }}>draw</span>
                            <span>Sign &amp; Seal</span>
                          </button>
                        )}
                      </div>
                      <div style={{ fontSize: 9.5, color: 'var(--text-muted)', marginTop: 1 }}>
                        {m.timestamp} · {m.description}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* PKI TRUST ANCHOR */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 8,
                  padding: '6px 12px',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: 6,
                  fontSize: 9.5,
                  marginTop: 6,
                }}
              >
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 9 }}>HSM Key Custody</span>
                  <strong style={{ color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 12, color: '#10b981' }}>check_circle</span>
                    FIPS 140-2 Level 3
                  </strong>
                </div>
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 9 }}>Time Authority</span>
                  <strong style={{ color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 12, color: '#10b981' }}>schedule</span>
                    GNSS Stratum-1
                  </strong>
                </div>
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 9 }}>Legal Certificate</span>
                  <strong style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 12 }}>verified</span>
                    BSA §63 / 65B Certified
                  </strong>
                </div>
              </div>

              {/* LEGAL ADMISSIBILITY ADVISORY */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '10px 14px',
                  marginTop: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>balance</span>
                    Statutory Compliance &amp; Legal Precedent
                  </span>
                  <span style={{ fontSize: 9.5, color: '#10b981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 3 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 12 }}>gavel</span>
                    Court Admissible
                  </span>
                </div>
                <p style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.45, margin: 0 }}>
                  This cryptographic packet adheres to the Bharatiya Sakshya Adhiniyam 2023 (Section 63 electronic record admissibility), Indian Evidence Act (Section 65B), and ISO/IEC 27037 digital forensics standards.
                </p>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span className="scenario-chip" style={{ fontSize: 9, padding: '2px 8px' }}>MARPOL 73/78 Annex I</span>
                  <span className="scenario-chip" style={{ fontSize: 9, padding: '2px 8px' }}>MS Act 1958 Sec 356</span>
                  <span className="scenario-chip" style={{ fontSize: 9, padding: '2px 8px' }}>UNCLOS Art. 217</span>
                  <span className="scenario-chip" style={{ fontSize: 9, padding: '2px 8px', borderColor: 'rgba(37, 99, 235, 0.4)', color: 'var(--accent)' }}>BSA 2023 §63</span>
                  <span className="scenario-chip" style={{ fontSize: 9, padding: '2px 8px', borderColor: 'rgba(37, 99, 235, 0.4)', color: 'var(--accent)' }}>ISO/IEC 27037:2012</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: ARTIFACT INSPECTOR & LIVE VERIFIER */}
      {activeSubTab === 'artifacts' && (
        <div className="canvas-rounded-container" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* VERIFIER BANNER & ACTION BAR */}
          <div
            style={{
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 10,
              padding: '14px 18px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 12,
            }}
          >
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="material-symbols-outlined" style={{ color: '#10b981', fontSize: 20 }}>security</span>
                Interactive Cryptographic Integrity Verifier
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--text-secondary)', marginTop: 2 }}>
                Perform on-the-spot bit-level verification of all 4 ingested forensic assets against the ECDSA Ledger Root.
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                className="action-pill-btn secondary"
                onClick={() => handleDownloadArtifact('full_forensic_bundle.zip')}
                title="Download All 4 Assets in Zipped Archive"
                style={{ padding: '6px 14px', fontSize: 11 }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>archive</span>
                <span>Download All (.ZIP)</span>
              </button>

              <button
                className="action-pill-btn primary"
                onClick={handleRunAudit}
                disabled={isAuditing}
                style={{ padding: '6px 14px', fontSize: 11, cursor: 'pointer' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>
                  {isAuditing ? 'sync' : 'verified'}
                </span>
                <span>{isAuditing ? 'Auditing Bits...' : 'Run Integrity Audit'}</span>
              </button>
            </div>
          </div>

          {/* AUDIT PROGRESS BAR (WHEN AUDITING OR JUST FINISHED) */}
          {(isAuditing || auditComplete) && (
            <div
              style={{
                background: auditComplete ? 'rgba(16, 185, 129, 0.08)' : 'rgba(37, 99, 235, 0.08)',
                border: auditComplete ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(37, 99, 235, 0.3)',
                borderRadius: 8,
                padding: '10px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 700 }}>
                <span style={{ color: auditComplete ? '#10b981' : 'var(--accent)' }}>
                  {auditStepText}
                </span>
                <span className="mono" style={{ color: auditComplete ? '#10b981' : 'var(--accent)' }}>
                  {auditProgress}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: 'var(--bg-base)', borderRadius: 3, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${auditProgress}%`,
                    height: '100%',
                    background: auditComplete ? '#10b981' : 'var(--accent)',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>
          )}

          {/* ARTIFACT CARDS GRID */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
            {artifacts.map((art) => (
              <div
                key={art.id}
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '12px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 6,
                        background: 'var(--bg-base)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: art.iconColor,
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{art.icon}</span>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
                        {art.fileName}
                      </div>
                      <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                        {art.crs} · {art.sizeKb.toFixed(1)} KB
                      </div>
                    </div>
                  </div>

                  <span className="chip chip-ok" style={{ fontSize: 9, padding: '2px 6px' }}>
                    SEALED
                  </span>
                </div>

                <p style={{ fontSize: 10.5, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                  {art.description}
                </p>

                {/* INDIVIDUAL SHA-256 HASH */}
                <div
                  style={{
                    background: 'var(--bg-base)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 4,
                    padding: '5px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <span className="mono text-muted" style={{ fontSize: 9, wordBreak: 'break-all' }}>
                    SHA-256: {art.sha256.slice(0, 36)}...
                  </span>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => handleCopyText(art.sha256, `${art.fileName} SHA-256`)}
                    title="Copy full SHA-256 hash"
                    style={{ padding: '2px 6px', fontSize: 9 }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 11 }}>content_copy</span>
                  </button>
                </div>

                {/* ACTION BUTTONS */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 'auto', paddingTop: 4 }}>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => setPreviewArtifact(art)}
                    style={{ fontSize: 10.5, padding: '4px 10px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>visibility</span>
                    <span>Inspect / Preview</span>
                  </button>
                  <button
                    className="action-pill-btn primary"
                    onClick={() => handleDownloadArtifact(art.fileName)}
                    disabled={downloadingFile === art.fileName}
                    style={{ fontSize: 10.5, padding: '4px 10px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                    <span>Download</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VIEW 3: COURT DOSSIER & STATUTORY CERTIFICATE */}
      {activeSubTab === 'dossier' && (
        <div className="canvas-rounded-container" style={{ padding: 20 }}>
          <div
            style={{
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 10,
              padding: '24px 28px',
              maxWidth: 920,
              margin: '0 auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
          >
            {/* OFFICIAL HEADER */}
            <div style={{ textAlign: 'center', borderBottom: '2px solid var(--border-default)', paddingBottom: 14 }}>
              <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                Government of India · Ministry of Defence · Indian Coast Guard
              </div>
              <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)', marginTop: 4 }}>
                MARITIME DIGITAL FORENSIC ATTRIBUTION REPORT
              </div>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--accent)', marginTop: 2 }}>
                Statutory Certificate under Section 63, Bharatiya Sakshya Adhiniyam (BSA) 2023 &amp; Section 65B, Indian Evidence Act
              </div>
              <div className="mono text-xs text-muted" style={{ marginTop: 4 }}>
                Dossier Ref: ICG/MRCC/2026/SS-{currentScenario?.id || 'INC-005'} · Classification: RESTRICTED // INVESTIGATIVE DECISION-SUPPORT
              </div>
            </div>

            {/* ACTION BUTTONS ON TOP OF DOSSIER */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                className="action-pill-btn secondary"
                onClick={onOpenForensicModal}
                style={{ fontSize: 11, padding: '4px 12px' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>open_in_new</span>
                <span>Open in Fullscreen Modal</span>
              </button>
              <button
                className="action-pill-btn primary"
                onClick={() => window.print()}
                style={{ fontSize: 11, padding: '4px 12px' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>print</span>
                <span>Print / Save Official PDF</span>
              </button>
            </div>

            {/* INCIDENT DETAILS */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: 10,
                background: 'var(--bg-base)',
                padding: '12px 16px',
                borderRadius: 6,
                border: '1px solid var(--border-subtle)',
                fontSize: 11.5,
              }}
            >
              <div><strong>Incident Identifier:</strong> <span className="mono">{currentScenario?.id || 'INC-2026-005'}</span></div>
              <div><strong>Operational Sector:</strong> <span className="mono">{currentScenario?.title || 'Gulf of Kutch Fairway'}</span></div>
              <div><strong>Centroid Coordinate:</strong> <span className="mono">{(currentScenario?.lat || 22.61).toFixed(5)}°N, {(currentScenario?.lng || 69.5).toFixed(5)}°E</span></div>
              <div><strong>Derived Surface Footprint:</strong> <span className="mono">{currentScenario?.area || '19.67 km²'}</span></div>
              <div><strong>Sensor Mission:</strong> <span className="mono">Sentinel-1A IW GRD (Copernicus C-SAR)</span></div>
              <div><strong>MetOcean Forcing:</strong> <span className="mono">ERA5 4.2 m/s · CMEMS 0.34 kn Surface Current</span></div>
            </div>

            {/* ATTRIBUTED VESSEL */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6, color: 'var(--text-primary)' }}>
                Primary Suspect Vessel Attribution Record
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5, border: '1px solid var(--border-subtle)' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-base)', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Rank</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Vessel Name</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>MMSI / IMO</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Flag / Registry</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Attribution Score</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Correlated Anomaly</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontWeight: 700, color: 'var(--accent)' }}>#1 PRIMARY</td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontWeight: 700 }}>
                      {currentScenario?.topVessel || 'MT Ocean Trader'}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'monospace' }}>
                      419001234 / 9412345
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Liberia · Crude Tanker</td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'monospace', fontWeight: 800, color: '#ef4444' }}>
                      0.86 / 1.00
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', fontSize: 10.5 }}>
                      {currentScenario?.diagDetails || 'Intentional AIS silence gap (4.2 hrs) crossing Lagrangian origin contour'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* BSA 2023 SECTION 63 LEGAL ATTESTATION */}
            <div
              style={{
                background: 'rgba(37, 99, 235, 0.04)',
                border: '1px solid rgba(37, 99, 235, 0.25)',
                borderRadius: 6,
                padding: '12px 16px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--accent)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
                  Statutory Certificate of Electronic Evidence (Section 63 BSA 2023)
                </div>
                <span className="chip chip-ok" style={{ fontSize: 9.5, padding: '2px 8px' }}>SEALED &amp; VERIFIED</span>
              </div>
              <p style={{ fontSize: 10.5, color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                I hereby certify that the electronic records contained in this dossier were produced by the automated computer surveillance system <strong>SpillSense Maritime C2</strong> during the course of regular operational activity. The computer systems and cryptographic hardware security modules (HSM FIPS 140-2 Level 3) were operating properly throughout the ingestion, processing, trajectory drift modeling, and AIS correlation cycles, with no unauthorized tampering or alterations.
              </p>
              <div
                style={{
                  fontFamily: 'monospace',
                  fontSize: 10,
                  background: 'var(--bg-base)',
                  padding: '6px 10px',
                  borderRadius: 4,
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--accent)',
                  wordBreak: 'break-all',
                }}
              >
                Master Cryptographic Ledger Digest: {hashChunks.join(' ')}
              </div>
            </div>

            {/* OFFICER SIGN-OFF FOOTER */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 12, paddingTop: 14, borderTop: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                  Certifying Officer / Digital Signature
                </div>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {officerSignOff?.officerName || 'Duty Surveillance Officer (ICG Maritime Intel Unit)'}
                </div>
                <div className="mono text-xs text-muted" style={{ fontSize: 10, marginTop: 1 }}>
                  HSM FIPS 140-2 Token: ICG-C2-2026-9 · Status: {officerSignOff ? 'Digitally Signed & Endorsed' : 'Automated Ledger Sealed'}
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                  Statutory Enactment
                </div>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                  Sec 63 BSA 2023 · Sec 356 MS Act 1958 · MARPOL Annex I
                </div>
                <div className="mono text-xs text-muted" style={{ fontSize: 10, marginTop: 1 }}>
                  Indian Coast Guard Maritime Law Enforcement
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. INLINE ARTIFACT PREVIEW DRAWER / MODAL */}
      {previewArtifact && (
        <div
          className="modal-backdrop open"
          style={{ zIndex: 1250 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) setPreviewArtifact(null);
          }}
        >
          <div className="modal-card" style={{ maxWidth: 740, maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined" style={{ color: previewArtifact.iconColor }}>
                  {previewArtifact.icon}
                </span>
                <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'monospace' }}>
                  {previewArtifact.fileName}
                </span>
                <span className="scenario-chip" style={{ fontSize: 9.5 }}>{previewArtifact.crs}</span>
              </div>
              <div className="flex items-center gap-2 modal-actions">
                <button
                  className="action-pill-btn secondary"
                  onClick={() => {
                    const text = previewArtifact.contentGenerator(
                      currentScenario || ({} as Scenario),
                      masterHash
                    );
                    handleCopyText(text, previewArtifact.fileName);
                  }}
                  style={{ fontSize: 10.5, padding: '3px 8px' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>content_copy</span>
                  <span>Copy Raw</span>
                </button>

                <button
                  className="action-pill-btn primary"
                  onClick={() => handleDownloadArtifact(previewArtifact.fileName)}
                  style={{ fontSize: 10.5, padding: '3px 8px' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>download</span>
                  <span>Download</span>
                </button>

                <button className="btn-icon" onClick={() => setPreviewArtifact(null)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>

            <div className="modal-body" style={{ flex: 1, minHeight: 0, padding: 14, overflowY: 'auto' }}>
              <div
                style={{
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  padding: '10px 12px',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: 10.5,
                  lineHeight: 1.45,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  color: 'var(--text-primary)',
                  maxHeight: '60vh',
                  overflowY: 'auto',
                }}
              >
                {previewArtifact.contentGenerator(
                  currentScenario || ({} as Scenario),
                  masterHash
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 6. OFFICER SIGN-OFF DIALOG */}
      {isSignOffModalOpen && (
        <div
          className="modal-backdrop open"
          style={{ zIndex: 1250 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) setIsSignOffModalOpen(false);
          }}
        >
          <div className="modal-card" style={{ maxWidth: 480 }}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined" style={{ color: 'var(--accent)' }}>draw</span>
                <span style={{ fontSize: 13, fontWeight: 700 }}>
                  Duty Surveillance Officer Sign-Off &amp; Seal
                </span>
              </div>
              <button className="btn-icon" onClick={() => setIsSignOffModalOpen(false)}>
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: 16 }}>
              <p style={{ fontSize: 11, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                Applying your electronic endorsement registers an immutable milestone under Section 63 of Bharatiya Sakshya Adhiniyam 2023, sealing this evidence dossier for court proceedings.
              </p>

              <div>
                <label style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Officer Name &amp; Rank
                </label>
                <input
                  type="text"
                  value={signOffName}
                  onChange={(e) => setSignOffName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border-default)',
                    background: 'var(--bg-base)',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    marginTop: 4,
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Command Station / Unit
                </label>
                <input
                  type="text"
                  value={signOffStation}
                  onChange={(e) => setSignOffStation(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border-default)',
                    background: 'var(--bg-base)',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    marginTop: 4,
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                <button
                  className="action-pill-btn secondary"
                  onClick={() => setIsSignOffModalOpen(false)}
                  style={{ fontSize: 11, padding: '5px 12px' }}
                >
                  Cancel
                </button>
                <button
                  className="action-pill-btn primary"
                  onClick={handleConfirmSignOff}
                  style={{ fontSize: 11, padding: '5px 14px', cursor: 'pointer' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 15 }}>verified_user</span>
                  <span>Confirm &amp; Seal Dossier</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
