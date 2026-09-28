import React, { useState, useMemo } from 'react';
import type { Scenario } from '../../types/dashboard';
import { calculateCleanupPlan, type CleanupPlan } from '../../utils/cleanupOptimizer';

interface CleanupModalProps {
  isOpen: boolean;
  onClose: () => void;
  scenario: Scenario | null;
}

export const CleanupModal: React.FC<CleanupModalProps> = ({ isOpen, onClose, scenario }) => {
  const [responseMode, setResponseMode] = useState<'balanced' | 'mechanical_only' | 'dispersant_heavy'>('balanced');
  
  const plan: CleanupPlan | null = useMemo(() => {
    if (!scenario) return null;
    return calculateCleanupPlan(scenario, responseMode);
  }, [scenario, responseMode]);

  if (!isOpen || !scenario || !plan) return null;

  const tierBadgeColor = plan.tier === 'Tier 3' ? '#EF4444' : plan.tier === 'Tier 2' ? '#F59E0B' : '#10B981';
  const tierBg = plan.tier === 'Tier 3' ? 'rgba(239, 68, 68, 0.15)' : plan.tier === 'Tier 2' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)';

  return (
    <div
      className="modal-backdrop open"
      id="cleanup-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{ zIndex: 1200 }}
    >
      <div className="modal-card print-dossier-card" style={{ maxWidth: 880, maxHeight: '92vh', overflowY: 'auto' }}>
        {/* HEADER */}
        <div className="modal-header hide-on-print">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ color: '#10B981', fontSize: 22 }}>
              cleaning_services
            </span>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                NOS-DCP Clean-Up Logistics &amp; Resource Allocation Plan
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Indian Coast Guard Operational Incident Action Plan (IAP) · {scenario.id}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 modal-actions">
            <button
              className="action-pill-btn primary"
              onClick={() => window.print()}
              style={{ padding: '4px 12px', fontSize: 11, cursor: 'pointer', background: '#10B981', borderColor: '#10B981' }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>print</span>
              Export Incident Action Plan
            </button>
            <button className="btn-icon" onClick={onClose} title="Close Modal">
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>

        <div className="modal-body" style={{ padding: '18px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* ICG OFFICIAL TITLE */}
          <div style={{ textAlign: 'center', borderBottom: '2px solid var(--border-default)', paddingBottom: 12 }}>
            <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              National Oil Spill Disaster Contingency Plan (NOS-DCP) · Maritime Response Directorate
            </div>
            <div style={{ fontSize: 17, fontWeight: 800, color: 'var(--text-primary)', marginTop: 4 }}>
              POLLUTION RESPONSE LOGISTICS &amp; FINANCIAL EXPOSURE ASSESSMENT
            </div>
            <div style={{ fontSize: 11, color: 'var(--accent)', marginTop: 2, fontWeight: 600 }}>
              {scenario.title} · Coordinates: {scenario.lat.toFixed(4)}°N, {scenario.lng.toFixed(4)}°E
            </div>
          </div>

          {/* RESPONSE STRATEGY TABS */}
          <div className="hide-on-print cleanup-strategy-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-raised)', padding: '6px 10px', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)' }}>
              Response Strategy Doctrine:
            </span>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className={`base-btn ${responseMode === 'balanced' ? 'active' : ''}`}
                onClick={() => setResponseMode('balanced')}
                style={{
                  fontSize: 11,
                  padding: '4px 10px',
                  borderRadius: 4,
                  background: responseMode === 'balanced' ? '#10B981' : 'transparent',
                  color: responseMode === 'balanced' ? '#FFFFFF' : 'var(--text-muted)',
                  fontWeight: responseMode === 'balanced' ? 700 : 500,
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                Balanced (Mechanical + Aerial)
              </button>
              <button
                className={`base-btn ${responseMode === 'mechanical_only' ? 'active' : ''}`}
                onClick={() => setResponseMode('mechanical_only')}
                style={{
                  fontSize: 11,
                  padding: '4px 10px',
                  borderRadius: 4,
                  background: responseMode === 'mechanical_only' ? '#10B981' : 'transparent',
                  color: responseMode === 'mechanical_only' ? '#FFFFFF' : 'var(--text-muted)',
                  fontWeight: responseMode === 'mechanical_only' ? 700 : 500,
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                Mechanical Only (Eco-Sensitive)
              </button>
              <button
                className={`base-btn ${responseMode === 'dispersant_heavy' ? 'active' : ''}`}
                onClick={() => setResponseMode('dispersant_heavy')}
                style={{
                  fontSize: 11,
                  padding: '4px 10px',
                  borderRadius: 4,
                  background: responseMode === 'dispersant_heavy' ? '#10B981' : 'transparent',
                  color: responseMode === 'dispersant_heavy' ? '#FFFFFF' : 'var(--text-muted)',
                  fontWeight: responseMode === 'dispersant_heavy' ? 700 : 500,
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                Deep Sea Aerial Sorties
              </button>
            </div>
          </div>

          {/* TIER HERO BANNER */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px 18px',
              borderRadius: 8,
              background: tierBg,
              border: `1.5px solid ${tierBadgeColor}66`,
            }}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 800,
                    padding: '2px 8px',
                    borderRadius: 4,
                    background: tierBadgeColor,
                    color: '#FFFFFF',
                    letterSpacing: '0.05em',
                  }}
                >
                  {plan.tier.toUpperCase()} EMERGENCY
                </span>
                <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {plan.tierAuthority}
                </span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                {plan.tierDescription}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Est. Clean-up Window</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: tierBadgeColor }}>
                {plan.estimatedDays} Days
              </div>
            </div>
          </div>

          {/* TOP 4 KEY METRIC CARDS */}
          <div className="cleanup-metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            {/* Card 1: Estimated Spill Mass */}
            <div style={{ background: 'var(--bg-raised)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Spill Mass / Volume</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: '#38BDF8', marginTop: 2 }}>
                {plan.estimatedMassMT.toLocaleString()} <span style={{ fontSize: 11 }}>MT</span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {plan.estimatedVolumeM3.toLocaleString()} m³ ({plan.oilType})
              </div>
            </div>

            {/* Card 2: Booms Required */}
            <div style={{ background: 'var(--bg-raised)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Containment Booms</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: '#F59E0B', marginTop: 2 }}>
                {plan.boomLengthMeters.toLocaleString()} <span style={{ fontSize: 11 }}>meters</span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Offshore Curtain Boom
              </div>
            </div>

            {/* Card 3: Skimmer Recovery Capacity */}
            <div style={{ background: 'var(--bg-raised)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Skimmer Units</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: '#10B981', marginTop: 2 }}>
                {plan.skimmerUnits} <span style={{ fontSize: 11 }}>units</span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {plan.skimmerCapacityM3PerHr} m³/hr nominal recovery
              </div>
            </div>

            {/* Card 4: Estimated Operation Cost */}
            <div style={{ background: 'var(--bg-raised)', padding: '10px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Total Financial Exposure</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: '#EC4899', marginTop: 2 }}>
                ₹{plan.costBreakdownINR.totalCrores} <span style={{ fontSize: 11 }}>Cr</span>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                ~${(plan.costUSD / 1000000).toFixed(2)}M USD
              </div>
            </div>
          </div>

          {/* EQUIPMENT ALLOCATION MANIFEST */}
          <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 6, overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px', background: 'var(--bg-raised)', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>inventory_2</span>
              <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Tactical Equipment &amp; Asset Mobilization Manifest
              </span>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <tbody>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '8px 12px', color: 'var(--text-muted)', width: '30%' }}>Containment Barriers</td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>
                    <strong>{plan.boomLengthMeters}m</strong> · {plan.boomType} ({plan.boomFormation})
                  </td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>Mechanical Skimmers</td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>
                    <strong>{plan.skimmerUnits} Units</strong> · {plan.skimmerType} with dedicated hydraulic power packs
                  </td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>Temporary Floating Storage</td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>
                    <strong>{plan.temporaryStorageM3} m³</strong> · {plan.storageType}
                  </td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>Chemical Dispersant &amp; Air Sorties</td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>
                    {plan.dispersantLiters > 0 ? (
                      <>
                        <strong>{plan.dispersantLiters.toLocaleString()} Liters</strong> · {plan.dispersantType} (<strong>{plan.dornierSorties} Dornier 228</strong> aerial spray sorties)
                      </>
                    ) : (
                      <span style={{ color: '#F59E0B' }}>Restricted — Mechanical Recovery Only due to shoreline/environmental protocol</span>
                    )}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>Flotilla Task Force</td>
                  <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>
                    {plan.vesselsRequired.prvCount > 0 && <span><strong>{plan.vesselsRequired.prvCount}x</strong> Dedicated Pollution Response Vessel (ICGS Samudra Class) · </span>}
                    <span><strong>{plan.vesselsRequired.tugsCount}x</strong> Ocean Towing Tugs · </span>
                    <span><strong>{plan.vesselsRequired.fastResponseCraft}x</strong> Fast Interceptor Crafts (FICs)</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* FINANCIAL COST BREAKDOWN TABLE */}
          <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 6, overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px', background: 'var(--bg-raised)', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#EC4899' }}>payments</span>
                <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Statutory Clean-Up Cost Breakdown (NOS-DCP Tariff Schedule)
                </span>
              </div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Rates in ₹ Lakhs &amp; ₹ Crores
              </span>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: 'var(--bg-raised)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left' }}>
                  <th style={{ padding: '6px 12px' }}>Operational Cost Component</th>
                  <th style={{ padding: '6px 12px' }}>Rate Basis</th>
                  <th style={{ padding: '6px 12px', textAlign: 'right' }}>Amount (₹ Lakhs)</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Vessel Flotilla Charter &amp; Bunkers</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>PRV, Tugs, and FICs for {plan.estimatedDays} days</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.vesselCharterLakhs} L</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Aviation Sorties (Dornier 228)</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>{plan.dornierSorties} reconnaissance &amp; spray flight hours</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.aircraftSortiesLakhs} L</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Boom &amp; Skimmer Equipment Mobilization</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>{plan.boomLengthMeters}m booms + {plan.skimmerUnits} skimmer powerpacks</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.equipmentDeploymentLakhs} L</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Chemical Dispersant Consumable</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>{plan.dispersantLiters.toLocaleString()} L Type-III concentrate</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.dispersantChemicalLakhs} L</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Hazmat Personnel &amp; Shore Teams</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>Certified oil spill response crew &amp; safety ops</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.manpowerHazmatLakhs} L</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '6px 12px', color: 'var(--text-primary)' }}>Oily Waste Disposal &amp; Bioremediation</td>
                  <td style={{ padding: '6px 12px', color: 'var(--text-muted)' }}>Port oily sludge transfer &amp; ecological treatment</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', fontWeight: 600 }}>₹{plan.costBreakdownINR.wasteDisposalBioremediationLakhs} L</td>
                </tr>
                <tr style={{ background: 'var(--bg-raised)', fontWeight: 800, color: 'var(--text-primary)' }}>
                  <td style={{ padding: '8px 12px' }} colSpan={2}>TOTAL ESTIMATED COST RECOVERY CLAIM</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10B981', fontSize: 13 }}>
                    ₹{plan.costBreakdownINR.totalCrores} Crores
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* STATUTORY LIABILITY RECOVERY NOTICE */}
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 6,
              background: 'rgba(56, 189, 248, 0.08)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              fontSize: 10.5,
              lineHeight: 1.5,
            }}
          >
            <div style={{ color: 'var(--accent)', fontWeight: 700, marginBottom: 2 }}>
              ⚖️ Statutory Polluter-Pays Cost Recovery Notice
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>
              Under <strong>{plan.statutoryAct}</strong> and the International Convention on Civil Liability for Oil Pollution Damage (CLC 1992), this expenditure manifest is directly recoverable from the registered shipowner and their P&amp;I Club insurer.
            </div>
            <div style={{ color: 'var(--text-primary)', marginTop: 4, fontWeight: 600 }}>
              Primary Offender Target: <span style={{ color: '#F59E0B' }}>{scenario.topVessel}</span> ({plan.piClubLiability})
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
