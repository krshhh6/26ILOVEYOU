import type { Scenario } from '../types/dashboard';

export interface CleanupPlan {
  scenarioId: string;
  scenarioTitle: string;
  oilType: string;
  spillAreaKm2: number;
  estimatedVolumeM3: number;
  estimatedMassMT: number; // Metric Tonnes
  densityTonnesPerM3: number;
  
  // NOS-DCP Tier
  tier: 'Tier 1' | 'Tier 2' | 'Tier 3';
  tierAuthority: string;
  tierDescription: string;
  
  // Equipment Requirements
  boomLengthMeters: number;
  boomType: string;
  boomFormation: string;
  skimmerUnits: number;
  skimmerType: string;
  skimmerCapacityM3PerHr: number;
  temporaryStorageM3: number;
  storageType: string;
  
  // Aviation & Chemical Dispersant
  dispersantLiters: number;
  dispersantType: string;
  dornierSorties: number;
  vesselsRequired: {
    prvCount: number; // Pollution Response Vessels (e.g. ICGS Samudra Prahari)
    tugsCount: number;
    fastResponseCraft: number;
  };
  
  // Operational Timeline & Financials
  estimatedDays: number;
  costBreakdownINR: {
    vesselCharterLakhs: number;
    aircraftSortiesLakhs: number;
    equipmentDeploymentLakhs: number;
    dispersantChemicalLakhs: number;
    manpowerHazmatLakhs: number;
    wasteDisposalBioremediationLakhs: number;
    totalCrores: number;
  };
  costUSD: number;
  
  // Legal & Regulatory
  statutoryAct: string;
  piClubLiability: string;
}

/**
 * Calculates a defense-grade NOS-DCP Clean-Up Logistics and Cost Allocation Plan
 * calibrated to Indian Coast Guard pollution response doctrine.
 */
export function calculateCleanupPlan(scenario: Scenario, responseMode: 'balanced' | 'mechanical_only' | 'dispersant_heavy' = 'balanced'): CleanupPlan {
  // 1. Extract area in km²
  const areaMatch = scenario.area ? parseFloat(scenario.area.replace(/[^\d.]/g, '')) : 2.5;
  const areaKm2 = isNaN(areaMatch) || areaMatch <= 0 ? 2.5 : areaMatch;
  
  // 2. Oil characteristics & density (tonnes/m³)
  const oilLower = (scenario.oilType || '').toLowerCase();
  let density = 0.88; // default crude
  let avgThicknessMm = 0.05; // 50 microns average across sheen + core
  let dispersantEfficacy = 0.75;
  
  if (oilLower.includes('bunker') || oilLower.includes('heavy') || oilLower.includes('fuel')) {
    density = 0.96;
    avgThicknessMm = 0.12; // thicker heavy fuel oil
    dispersantEfficacy = 0.45; // HFO resists chemical dispersants
  } else if (oilLower.includes('crude')) {
    density = 0.87;
    avgThicknessMm = 0.08;
    dispersantEfficacy = 0.80; // fresh crude disperses well
  } else if (oilLower.includes('bilge') || oilLower.includes('light') || oilLower.includes('condensate')) {
    density = 0.83;
    avgThicknessMm = 0.02; // light sheen
    dispersantEfficacy = 0.90;
  }
  
  // Volume (m³) = Area (m²) * Thickness (m)
  // Area in m² = areaKm2 * 1,000,000
  // Thickness in m = avgThicknessMm / 1,000
  const estimatedVolumeM3 = Math.round(areaKm2 * 1000000 * (avgThicknessMm / 1000));
  const estimatedMassMT = Math.round(estimatedVolumeM3 * density);
  
  // 3. Indian NOS-DCP Disaster Tier Classification
  let tier: 'Tier 1' | 'Tier 2' | 'Tier 3' = 'Tier 1';
  let tierAuthority = 'Local Port Trust / Facility Terminal Operator (NOS-DCP Tier 1)';
  let tierDescription = 'Localized spill handled with port & terminal organic pollution response equipment.';
  
  if (estimatedMassMT > 10000) {
    tier = 'Tier 3';
    tierAuthority = 'National Executive Committee (DG Indian Coast Guard + NDMA + PMO)';
    tierDescription = 'National catastrophe requiring multi-ministry mobilization & international MARPOL assistance.';
  } else if (estimatedMassMT >= 700) {
    tier = 'Tier 2';
    tierAuthority = 'Indian Coast Guard Regional Commander (Western / Eastern Seaboard)';
    tierDescription = 'Regional maritime emergency coordinating Coast Guard, Navy, state pollution boards & ports.';
  }
  
  // 4. Equipment Sizing
  // Containment boom: need perimeter to encircle the active slick or maintain sweeping J-booms
  const majorAxisKm = Math.sqrt(areaKm2) * 1.5;
  const rawBoomMeters = Math.round(majorAxisKm * 1000 * 1.25);
  const boomLengthMeters = Math.min(Math.max(rawBoomMeters, 800), 5000);
  
  const boomType = tier === 'Tier 1' 
    ? 'Inshore Foam-Filled Curtain Boom (750mm total height / 450mm skirt)'
    : 'Heavy-Duty Offshore Inflatable Ocean Curtain Boom (1,200mm total height / 750mm skirt)';
    
  const boomFormation = areaKm2 > 3.0 ? 'Dual-Vessel Dynamic U-Sweep with Sweeping Pocket' : 'J-Formation Containment with Anchored Deflection Barriers';
  
  // Mechanical skimmers
  const skimmerUnits = Math.min(Math.max(Math.ceil(estimatedVolumeM3 / 150), 2), 8);
  const skimmerType = density > 0.90
    ? 'Oleophilic Grooved Disc / Brush High-Viscosity Skimmer'
    : 'Dynamic Weir / Self-Adjusting Suction Skimmer';
  const skimmerCapacityM3PerHr = skimmerUnits * 45; // 45 m³/hr per unit nominal
  
  // Temporary storage (Dracone barges / floating bladders)
  const temporaryStorageM3 = Math.round(estimatedVolumeM3 * 0.40);
  const storageType = 'Towable Inflatable Dracone Barge Containers (100m³ modules)';
  
  // 5. Aviation & Chemical Dispersant (CPCB / ICG Type-III)
  // Ratio 1:20 (1 liter dispersant per 20 liters of oil)
  let dispersantLiters = 0;
  let dornierSorties = 0;
  
  if (responseMode !== 'mechanical_only') {
    const treatedVolumeLiters = (estimatedVolumeM3 * 1000) * dispersantEfficacy * (responseMode === 'dispersant_heavy' ? 0.60 : 0.35);
    dispersantLiters = Math.round(treatedVolumeLiters / 20);
    // Dornier 228 can carry ~1,000 liters of dispersant per flight sortie with spray booms
    dornierSorties = Math.min(Math.max(Math.ceil(dispersantLiters / 1000), 1), 12);
  }
  
  // Vessels
  const prvCount = tier === 'Tier 1' ? 0 : Math.min(Math.ceil(estimatedMassMT / 1200), 2);
  const tugsCount = Math.max(skimmerUnits, 2);
  const fastResponseCraft = Math.min(Math.max(Math.ceil(boomLengthMeters / 600), 2), 6);
  
  // 6. Operational Duration (Days)
  // Assuming 10 effective working hours per day at sea
  const dailyRecoveryM3 = (skimmerCapacityM3PerHr * 10 * 0.50); // 50% recovery efficiency in open sea
  const daysMechanical = Math.ceil(estimatedVolumeM3 / Math.max(dailyRecoveryM3, 1));
  const estimatedDays = Math.min(Math.max(daysMechanical, 2), 28);
  
  // 7. Financial Cost Estimation (INR Lakhs / Crores)
  const vesselCharterPerDayLakhs = (prvCount * 8.5) + (tugsCount * 2.2) + (fastResponseCraft * 0.8);
  const vesselCharterLakhs = +(vesselCharterPerDayLakhs * estimatedDays).toFixed(1);
  const aircraftSortiesLakhs = +(dornierSorties * 4.5).toFixed(1); // ₹4.5 Lakhs per Dornier flight hour/sorties
  const equipmentDeploymentLakhs = +(boomLengthMeters * 0.015 + skimmerUnits * 5.0).toFixed(1);
  const dispersantChemicalLakhs = +((dispersantLiters / 1000) * 1.8).toFixed(1); // ₹1.8 Lakhs per 1000L concentrate
  const manpowerHazmatLakhs = +(estimatedDays * 3.5).toFixed(1);
  const wasteDisposalBioremediationLakhs = +(estimatedMassMT * 0.08).toFixed(1);
  
  const totalLakhs = vesselCharterLakhs + aircraftSortiesLakhs + equipmentDeploymentLakhs + dispersantChemicalLakhs + manpowerHazmatLakhs + wasteDisposalBioremediationLakhs;
  const totalCrores = +(totalLakhs / 100).toFixed(2);
  const costUSD = Math.round((totalLakhs * 100000) / 86.5); // Approx 1 USD = 86.5 INR
  
  return {
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    oilType: scenario.oilType || 'Crude Oil',
    spillAreaKm2: areaKm2,
    estimatedVolumeM3,
    estimatedMassMT,
    densityTonnesPerM3: density,
    tier,
    tierAuthority,
    tierDescription,
    boomLengthMeters,
    boomType,
    boomFormation,
    skimmerUnits,
    skimmerType,
    skimmerCapacityM3PerHr,
    temporaryStorageM3,
    storageType,
    dispersantLiters,
    dispersantType: 'CPCB / ICG Type-III Concentrated Water-Soluble Marine Dispersant',
    dornierSorties,
    vesselsRequired: {
      prvCount,
      tugsCount,
      fastResponseCraft,
    },
    estimatedDays,
    costBreakdownINR: {
      vesselCharterLakhs,
      aircraftSortiesLakhs,
      equipmentDeploymentLakhs,
      dispersantChemicalLakhs,
      manpowerHazmatLakhs,
      wasteDisposalBioremediationLakhs,
      totalCrores,
    },
    costUSD,
    statutoryAct: 'Section 352J, Indian Merchant Shipping Act 1958 & NOS-DCP 2024 Guidelines',
    piClubLiability: `Billed to registered insurer: ${scenario.topVessel ? scenario.topVessel + ' P&I Club / International Group of P&I' : 'Standard P&I Club Guarantee'}`,
  };
}
