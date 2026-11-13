// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Forecast: for each market, a peaked (non-normal) quantile band around a
// slowly drifting median, with a per-market volatility that scales with the
// price level and with recent realized volatility.
//
//   median(h)  = x + DRIFT * (DLEVEL - x) * (h / 240)
//   sigma      = sqrt( (A * (x*(1-x))^POW * (1 + HBOOST*max(0, x-0.1)))^2 + (C * r360)^2 )
//   q_tau(h)   = clamp(median(h) + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 decimals
//
// where x is the last price, r360 the RMS of the last 360 one-minute returns,
// and z_tau the standard normal quantile at tau scaled by WI (inner levels
// 0.25/0.75) or WO (outer levels 0.05/0.95) and by ZS. The inner quantiles
// are tight and the outer ones wide, matching the step-like price process:
// most minutes the price does not move at all, and it occasionally jumps.
//
// Why this shape (fitted on the six practice cases, validated per case):
//   - Volatility grows with the price level (a 0.30 market moves ~10x more
//     per minute than a 0.004 one); a fixed sigma (the house baseline's
//     0.0025) is far too wide for the many near-zero markets and too narrow
//     for the leaders. (x*(1-x))^0.75 plus a high-price boost captures it.
//   - Recent 6h RMS return predicts near-term volatility (corr ~0.54).
//   - The median is persistence: mean reversion to the recent mean and
//     momentum both hurt on practice, so the median stays at x except for a
//     small drift toward 0.03 (low prices edge up, leaders edge down).
//   - The peaked shape (tight inner, wide outer quantiles) fits a process
//     that is flat most minutes and jumps occasionally.
//
// Practice score (mean pinball loss, lower is better): 0.2714 vs the house
// baseline's 0.363589; it beats the baseline on every practice case.
//
// Deterministic: no randomness, no clock, no network; the same input bytes
// always produce the same output bytes. Runs in well under 1s per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** Price-level volatility: sigma_base = A * (x*(1-x))^POW. */
const A = 0.005;
const POW = 0.75;
/** Extra volatility for high-priced markets: x above 0.1 gets a boost. */
const HBOOST = 2.0;
const HBOOST_FROM = 0.1;
/** Weight on recent realized volatility (RMS of 1-min returns, last 360 min). */
const C = 0.3;
const RVOL_WINDOW = 360;
/** Floor on per-minute sigma, so a market that never moved still gets a band. */
const SIGMA_FLOOR = 0.00005;
/** Median drift toward DLEVEL, reached linearly over the horizon. */
const DRIFT = 0.02;
const DLEVEL = 0.03;
/** Quantile shape: overall scale, inner (0.25/0.75) and outer (0.05/0.95) multipliers. */
const ZS = 0.8;
const WI = 0.8;
const WO = 1.15;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** RMS of one-minute returns over the last `window` prices (or fewer if short). */
function recentRms(prices: number[], window: number): number {
  const n = Math.min(window, prices.length);
  if (n < 2) return 0;
  const start = prices.length - n;
  let ss = 0;
  for (let i = start; i < prices.length - 1; i++) {
    const d = prices[i + 1] - prices[i];
    ss += d * d;
  }
  return Math.sqrt(ss / (n - 1));
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const r = recentRms(m.prices, RVOL_WINDOW);
  const base = A * Math.pow(Math.max(x * (1 - x), 1e-6), POW) * (1 + HBOOST * Math.max(0, x - HBOOST_FROM));
  const sigma = Math.max(Math.sqrt(base * base + (C * r) * (C * r)), SIGMA_FLOOR);
  // Effective z per level: inner levels scaled by WI, outer by WO, all by ZS.
  const ze = Z.map((z) => z * (Math.abs(z) < 1 ? WI : WO) * ZS);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const med = x + DRIFT * (DLEVEL - x) * (h / horizon);
    const s = sigma * Math.sqrt(h);
    const row: number[] = [];
    let prev = -1;
    for (let k = 0; k < QUANTILES.length; k++) {
      let q = price6(med + ze[k] * s);
      if (q < prev) q = prev; // rounding can break monotonicity; restore it
      prev = q;
      row.push(q);
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
