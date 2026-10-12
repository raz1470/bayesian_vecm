// SVG element helper
function el(tag, attrs, text) {
  const ns = 'http://www.w3.org/2000/svg';
  const e = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

// ── Chart sizing ───────────────────────────────────────────────────────────
// Font sizes are in viewBox units. A 700-unit viewBox renders at about 615px
// in the prose column, so FS lands near 12px on desktop. Below MOBILE_BP each
// chart redraws in a 340-unit viewBox so FSN lands near 12px on a phone.
const MOBILE_BP = 600;
const isNarrow = () => window.matchMedia(`(max-width: ${MOBILE_BP - 1}px)`).matches;
const FS  = { axis: 13.5, note: 12.5 };
const FSN = { axis: 11.5, note: 11 };

const COLOR = {
  sales: '#2563eb', awareness: '#059669', consideration: '#7c3aed',
  brand: '#d97706', pr: '#db2777', sponsorship: '#0891b2',
  mmm: '#dc2626', truth: '#111827',
};

const CHARTS = [];
function chart(id, vbWide, wide, narrow) { CHARTS.push({ id, vbWide, wide, narrow }); }

// Redraw only when the breakpoint is crossed: a phone fires resize on scroll.
let lastNarrow = null;
function renderCharts() {
  const narrow = isNarrow();
  if (narrow === lastNarrow) return;
  lastNarrow = narrow;
  for (const c of CHARTS) {
    const svg = document.getElementById(c.id);
    if (!svg) continue;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (narrow && c.narrow) { c.narrow(svg); } else { svg.setAttribute('viewBox', c.vbWide); c.wide(svg); }
  }
}

// One shared frame: gridlines, ticks, axes and axis labels.
// spec: { hW, hN, x: [min, max], y: [min, max], xTicks, yTicks, xLabel, yLabel }
function frame(svg, narrow, spec) {
  const W = narrow ? 340 : 700;
  const H = narrow ? spec.hN : spec.hW;
  const fs = narrow ? FSN.axis : FS.axis;
  const m = narrow ? { top: 10, right: 16, bottom: 46, left: 50 } : { top: 12, right: 24, bottom: 48, left: 70 };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pw = W - m.left - m.right, ph = H - m.top - m.bottom;
  const sx = v => m.left + (v - spec.x[0]) / (spec.x[1] - spec.x[0]) * pw;
  const sy = v => m.top + ph - (v - spec.y[0]) / (spec.y[1] - spec.y[0]) * ph;
  const g = el('g', {});
  for (const [v, lab] of spec.yTicks) {
    const y = sy(v);
    g.appendChild(el('line', { x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: '#e5e7eb', 'stroke-width': 1 }));
    g.appendChild(el('text', { x: m.left - 6, y: y + 4, 'text-anchor': 'end', 'font-size': fs, fill: '#9ca3af' }, lab));
  }
  const xt = narrow && spec.xTicksN ? spec.xTicksN : spec.xTicks;
  for (const [v, lab] of xt) {
    const x = sx(v);
    g.appendChild(el('line', { x1: x, y1: m.top + ph, x2: x, y2: m.top + ph + 5, stroke: '#d1d5db', 'stroke-width': 1 }));
    g.appendChild(el('text', { x, y: m.top + ph + 18, 'text-anchor': 'middle', 'font-size': fs, fill: '#9ca3af' }, lab));
  }
  g.appendChild(el('line', { x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));
  g.appendChild(el('line', { x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: '#d1d5db', 'stroke-width': 1 }));
  g.appendChild(el('text', { x: m.left + pw / 2, y: m.top + ph + 36, 'text-anchor': 'middle', 'font-size': fs, fill: '#9ca3af' }, spec.xLabel));
  const yl = narrow ? 11 : 16;
  g.appendChild(el('text', { x: yl, y: m.top + ph / 2, 'text-anchor': 'middle', 'font-size': fs, fill: '#9ca3af', transform: `rotate(-90 ${yl} ${m.top + ph / 2})` }, spec.yLabel));
  svg.appendChild(g);
  return { g, sx, sy, m, pw, ph, fs, narrow };
}

const linePath = (xs, ys, sx, sy) => xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ');

function line(f, xs, ys, color, opts = {}) {
  f.g.appendChild(el('path', {
    d: linePath(xs, ys, f.sx, f.sy), fill: 'none', stroke: color,
    'stroke-width': opts.width || (f.narrow ? 1.6 : 2), 'stroke-linejoin': 'round',
    ...(opts.dash ? { 'stroke-dasharray': opts.dash } : {}), ...(opts.opacity ? { opacity: opts.opacity } : {}),
  }));
}

function bandArea(f, xs, lo, hi, color, opacity = 0.18) {
  const fwd = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${f.sx(x).toFixed(1)},${f.sy(hi[i]).toFixed(1)}`).join(' ');
  const back = xs.map((x, i) => [x, lo[i]]).reverse().map(([x, v]) => `L${f.sx(x).toFixed(1)},${f.sy(v).toFixed(1)}`).join(' ');
  f.g.appendChild(el('path', { d: `${fwd} ${back} Z`, fill: color, opacity }));
}

function hline(f, v, color = '#9ca3af', dash = '3,3') {
  const y = f.sy(v);
  f.g.appendChild(el('line', { x1: f.m.left, y1: y, x2: f.m.left + f.pw, y2: y, stroke: color, 'stroke-width': 1, 'stroke-dasharray': dash }));
}

const range = n => Array.from({ length: n }, (_, i) => i);

// ── 1 · The data: brand series indexed to 100 ────────────────────────────────
const yearTicks = (() => {
  const t = []; let prev = null;
  DATA.series.dates.forEach((d, i) => { const y = d.slice(0, 4); if (y !== prev) { t.push([i, y]); prev = y; } });
  return t;
})();

function drawSeries(svg, narrow) {
  const n = DATA.series.dates.length;
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [0, n - 1], y: [80, 220],
    xTicks: yearTicks, xTicksN: yearTicks.filter((_, i) => i % 2 === 0),
    yTicks: [100, 140, 180, 220].map(v => [v, String(v)]),
    xLabel: 'Week', yLabel: 'Index (first week = 100)',
  });
  hline(f, 100);
  const xs = range(n);
  for (const [key, color] of [['brand_consideration', COLOR.consideration], ['brand_awareness', COLOR.awareness], ['organic_sales', COLOR.sales]]) {
    const v = DATA.series.levels[key];
    line(f, xs, v.map(x => 100 * x / v[0]), color, { width: narrow ? 1.2 : 1.4, opacity: 0.9 });
  }
}

// ── 2 · The problem: adstock against error correction ───────────────────────
function drawIllus(svg, narrow) {
  const il = DATA.illustration;
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [0, 52], y: [0, 1.1],
    xTicks: [0, 10, 20, 30, 40, 52].map(v => [v, String(v)]),
    yTicks: [0, 0.25, 0.5, 0.75, 1].map(v => [v, v.toFixed(2)]),
    xLabel: 'Weeks after the burst', yLabel: 'Weekly sales response',
  });
  hline(f, 1, COLOR.sales, '2,3');
  line(f, il.weeks, il.mmm, COLOR.mmm, { dash: '6,4' });
  line(f, il.weeks, il.vecm, COLOR.sales);
  const x10 = f.sx(10);
  f.g.appendChild(el('line', { x1: x10, y1: f.m.top, x2: x10, y2: f.m.top + f.ph, stroke: '#d1d5db', 'stroke-width': 1, 'stroke-dasharray': '2,3' }));
  const fn = narrow ? FSN.note : FS.note;
  f.g.appendChild(el('text', { x: x10 + 6, y: f.sy(il.vecm[10]) + (narrow ? 16 : 18), 'font-size': fn, fill: COLOR.sales, 'font-weight': 700 }, `${Math.round(il.vecm[10] * 100)}% by week 10`));
  f.g.appendChild(el('text', { x: x10 + 6, y: f.sy(il.mmm[10]) - 8, 'font-size': fn, fill: COLOR.mmm, 'font-weight': 700 }, `${Math.round(il.mmm[10] * 100)}% by week 10`));
}

// ── What a VECM is: two unrelated random walks ───────────────────────────────
function drawSpurious(svg, narrow) {
  const sp = DATA.spurious, n = sp.walk_a.length;
  const all = sp.walk_a.concat(sp.walk_b);
  const lo = Math.floor(Math.min(...all) / 10) * 10, hi = Math.ceil(Math.max(...all) / 10) * 10;
  const ticks = []; for (let v = lo; v <= hi; v += 10) ticks.push([v, String(v)]);
  const f = frame(svg, narrow, {
    hW: 240, hN: 220, x: [0, n - 1], y: [lo, hi],
    xTicks: [0, 52, 104, 156, 208, 259].map(v => [v, String(v)]),
    xTicksN: [0, 104, 208].map(v => [v, String(v)]),
    yTicks: ticks, xLabel: 'Week', yLabel: 'Level',
  });
  hline(f, 0);
  line(f, range(n), sp.walk_a, COLOR.sales, { width: narrow ? 1.2 : 1.4 });
  line(f, range(n), sp.walk_b, COLOR.brand, { width: narrow ? 1.2 : 1.4 });
}

// ── Forest: estimate, interval and truth per row ─────────────────────────────
// rows: [{ label, mean, lo, hi, truth, color }]. On a phone each label sits on
// its own line above its bar, so long labels never squeeze the plot.
function drawForest(svg, narrow, rows, opts) {
  const W = narrow ? 340 : 700;
  const fs = narrow ? FSN.axis : FS.axis;
  const rowH = narrow ? 50 : 38;
  const m = narrow ? { top: 8, right: 16, bottom: 44, left: 16 } : { top: 10, right: 24, bottom: 46, left: opts.labelW || 220 };
  const H = m.top + m.bottom + rowH * rows.length;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pw = W - m.left - m.right, ph = rowH * rows.length;
  const sx = v => m.left + (v - opts.x[0]) / (opts.x[1] - opts.x[0]) * pw;
  const g = el('g', {});
  rows.forEach((_, i) => { if (i % 2 === 1) g.appendChild(el('rect', { x: m.left, y: m.top + rowH * i, width: pw, height: rowH, fill: '#f9fafb' })); });
  for (const v of opts.ticks) {
    const x = sx(v);
    g.appendChild(el('line', { x1: x, y1: m.top, x2: x, y2: m.top + ph, stroke: v === 0 ? '#9ca3af' : '#e5e7eb', 'stroke-width': 1 }));
    g.appendChild(el('text', { x, y: m.top + ph + 18, 'text-anchor': 'middle', 'font-size': fs, fill: '#9ca3af' }, opts.fmtTick(v)));
  }
  g.appendChild(el('text', { x: m.left + pw / 2, y: m.top + ph + 36, 'text-anchor': 'middle', 'font-size': fs, fill: '#9ca3af' }, opts.axisLabel));
  rows.forEach((r, i) => {
    const top = m.top + rowH * i;
    const cy = narrow ? top + rowH * 0.68 : top + rowH / 2;
    if (narrow) {
      g.appendChild(el('text', { x: m.left + 4, y: top + 16, 'font-size': fs, fill: '#111827', 'font-weight': 600 }, r.label));
    } else {
      g.appendChild(el('text', { x: 12, y: cy + 4.5, 'font-size': fs, fill: '#111827', 'font-weight': 600 }, r.label));
    }
    g.appendChild(el('line', { x1: sx(r.lo), y1: cy, x2: sx(r.hi), y2: cy, stroke: r.color, 'stroke-width': 7, 'stroke-linecap': 'round', opacity: 0.85 }));
    g.appendChild(el('circle', { cx: sx(r.mean), cy, r: 3.8, fill: '#fff', stroke: r.color, 'stroke-width': 1.8 }));
    if (r.truth !== undefined) {
      const tx = sx(r.truth), h = narrow ? 11 : 12;
      g.appendChild(el('line', { x1: tx, y1: cy - h, x2: tx, y2: cy + h, stroke: COLOR.truth, 'stroke-width': 1.6, 'stroke-dasharray': '3,2' }));
    }
  });
  g.appendChild(el('rect', { x: m.left, y: m.top, width: pw, height: ph, fill: 'none', stroke: '#e5e7eb', 'stroke-width': 1 }));
  svg.appendChild(g);
}

// ── 3 · The long run ─────────────────────────────────────────────────────────
function drawGaps(svg, narrow) {
  const n = DATA.series.dates.length;
  const f = frame(svg, narrow, {
    hW: 240, hN: 220, x: [0, n - 1], y: [-0.08, 0.08],
    xTicks: yearTicks, xTicksN: yearTicks.filter((_, i) => i % 2 === 0),
    yTicks: [-0.08, -0.04, 0, 0.04, 0.08].map(v => [v, v.toFixed(2)]),
    xLabel: 'Week', yLabel: 'Gap (log units)',
  });
  hline(f, 0, '#6b7280', '4,3');
  const xs = range(n);
  line(f, xs, DATA.relations.relation_2, COLOR.consideration, { width: narrow ? 1.1 : 1.3, opacity: 0.85 });
  line(f, xs, DATA.relations.relation_1, COLOR.sales, { width: narrow ? 1.1 : 1.3, opacity: 0.85 });
}

const NICE = { organic_sales: 'Sales', brand_awareness: 'Awareness', brand_consideration: 'Consideration' };
const VCOL = { organic_sales: COLOR.sales, brand_awareness: COLOR.awareness, brand_consideration: COLOR.consideration };
const REL = { 1: 'gap 1', 2: 'gap 2' };

function drawAlpha(svg, narrow) {
  const rows = DATA.alpha.map(a => ({
    label: `${NICE[a.variable]}, ${REL[a.relation]}`,
    mean: a.mean, lo: a.lo, hi: a.hi, truth: a.truth, color: VCOL[a.variable],
  }));
  drawForest(svg, narrow, rows, {
    x: [-0.35, 0.3], ticks: [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], labelW: 190,
    fmtTick: v => v.toFixed(1), axisLabel: 'Adjustment speed (α)',
  });
}

// ── 5 · The fit: hold-out forecast of sales ──────────────────────────────────
function drawHoldout(svg, narrow) {
  const ho = DATA.holdout, s = ho.series.organic_sales;
  const nh = ho.history_sales.length, n = nh + ho.weeks;
  const xsH = range(nh), xsF = range(ho.weeks).map(i => nh + i);
  const vals = ho.history_sales.concat(s.actual, s.lo, s.hi, s.naive);
  const lo = Math.floor(Math.min(...vals) / 200) * 200, hi = Math.ceil(Math.max(...vals) / 200) * 200;
  const ticks = []; for (let v = lo; v <= hi; v += 400) ticks.push([v, `£${(v / 1000).toFixed(1)}k`]);
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const lab = d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(2, 4)}`;
  const xt = [0, nh, nh + 13, nh + 26, nh + 39].map(i => [i, lab(ho.dates[i])]);
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [0, n - 1], y: [lo, hi],
    xTicks: xt, xTicksN: [xt[0], xt[1], xt[3]],
    yTicks: ticks, xLabel: 'Week', yLabel: 'Weekly organic sales',
  });
  const xc = f.sx(nh - 0.5);
  f.g.appendChild(el('line', { x1: xc, y1: f.m.top, x2: xc, y2: f.m.top + f.ph, stroke: '#9ca3af', 'stroke-width': 1, 'stroke-dasharray': '2,3' }));
  const fn = narrow ? FSN.note : FS.note;
  f.g.appendChild(el('text', { x: xc + 6, y: f.m.top + 14, 'font-size': fn, fill: '#6b7280', 'font-weight': 700 }, 'Forecast from here'));
  bandArea(f, xsF, s.lo, s.hi, COLOR.sales, 0.16);
  line(f, xsF, s.naive, COLOR.mmm, { dash: '6,4' });
  line(f, xsF, s.mean, COLOR.sales);
  line(f, xsH.concat(xsF), ho.history_sales.concat(s.actual), COLOR.truth, { width: narrow ? 1 : 1.2 });
}

// ── 4 · The awareness effect ─────────────────────────────────────────────────
function drawIrf(svg, narrow) {
  const ir = DATA.irf_awareness;
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [0, 52], y: [0, 1.4],
    xTicks: [0, 4, 13, 26, 39, 52].map(v => [v, String(v)]),
    xTicksN: [0, 13, 26, 39, 52].map(v => [v, String(v)]),
    yTicks: [0, 0.4, 0.8, 1.2].map(v => [v, `${v.toFixed(1)}%`]),
    xLabel: 'Weeks after the awareness surprise', yLabel: '% response',
  });
  for (const k of ['brand_consideration', 'brand_awareness', 'organic_sales']) bandArea(f, ir.horizon, ir[k].lo, ir[k].hi, VCOL[k], 0.13);
  for (const k of ['brand_consideration', 'brand_awareness', 'organic_sales']) line(f, ir.horizon, ir[k].mean, VCOL[k]);
}

function drawCum(svg, narrow) {
  const c = DATA.irf_cumulative;
  const pct = a => a.map(v => v * 100);
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [0, 52], y: [0, 70],
    xTicks: [0, 4, 13, 26, 39, 52].map(v => [v, String(v)]),
    xTicksN: [0, 13, 26, 39, 52].map(v => [v, String(v)]),
    yTicks: [0, 20, 40, 60].map(v => [v, `${v}%`]),
    xLabel: 'Weeks after the awareness surprise', yLabel: 'Sum of weekly uplifts',
  });
  bandArea(f, c.horizon, pct(c.vecm.lo), pct(c.vecm.hi), COLOR.sales, 0.16);
  line(f, c.horizon, pct(c.vecm.mean), COLOR.sales);
  line(f, c.horizon, pct(c.adstock_matched_week4), COLOR.mmm, { dash: '6,4' });
  const fn = narrow ? FSN.note : FS.note;
  const last = c.horizon.length - 1;
  f.g.appendChild(el('text', { x: f.sx(52) - 4, y: f.sy(c.vecm.mean[last] * 100) - 12, 'text-anchor': 'end', 'font-size': fn, fill: COLOR.sales, 'font-weight': 700 }, `${(c.vecm.mean[last] * 100).toFixed(1)}%`));
  f.g.appendChild(el('text', { x: f.sx(52) - 4, y: f.sy(c.adstock_matched_week4[last] * 100) - 8, 'text-anchor': 'end', 'font-size': fn, fill: COLOR.mmm, 'font-weight': 700 }, `${(c.adstock_matched_week4[last] * 100).toFixed(1)}%`));
}

// ── 5 · What drives what ─────────────────────────────────────────────────────
function drawChannels(svg, narrow) {
  const names = { brand_spend: ['Brand spend', COLOR.brand], pr_reach: ['PR reach', COLOR.pr], sponsorship_reach: ['Sponsorship reach', COLOR.sponsorship] };
  const rows = DATA.causality.exog
    .filter(r => r.effect === 'brand_awareness' && names[r.cause])
    .map(r => ({ label: names[r.cause][0], mean: r.mean * 100, lo: r.lower * 100, hi: r.upper * 100, truth: r.truth * 100, color: names[r.cause][1] }));
  drawForest(svg, narrow, rows, {
    x: [0, 2.5], ticks: [0, 0.5, 1, 1.5, 2, 2.5], labelW: 170,
    fmtTick: v => `${v}%`, axisLabel: '% change in awareness per unit',
  });
}

// ── 6 · A campaign ───────────────────────────────────────────────────────────
function drawCampaign(svg, narrow) {
  const c = DATA.campaign;
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [1, 52], y: [0, 30],
    xTicks: [1, 13, 26, 39, 52].map(v => [v, String(v)]),
    yTicks: [0, 10, 20, 30].map(v => [v, `${v}%`]),
    xLabel: 'Week', yLabel: '% above baseline',
  });
  f.g.insertBefore(el('rect', { x: f.sx(1), y: f.m.top, width: f.sx(c.uplift_weeks) - f.sx(1), height: f.ph, fill: COLOR.brand, opacity: 0.08 }), f.g.firstChild);
  const fn = narrow ? FSN.note : FS.note;
  f.g.appendChild(el('text', { x: f.sx(2), y: f.m.top + 16, 'font-size': fn, fill: COLOR.brand, 'font-weight': 700 }, 'Campaign'));
  bandArea(f, c.week, c.awareness_pct.lo, c.awareness_pct.hi, COLOR.awareness, 0.14);
  bandArea(f, c.week, c.sales_pct.lo, c.sales_pct.hi, COLOR.sales, 0.14);
  line(f, c.week, c.awareness_pct.mean, COLOR.awareness);
  line(f, c.week, c.sales_pct.mean, COLOR.sales);
}

function drawCurve(svg, narrow) {
  const rc = DATA.response_curve;
  const xs = rc.map(r => r.spend_sd);
  const f = frame(svg, narrow, {
    hW: 260, hN: 240, x: [-2, 2], y: [-30, 50],
    xTicks: [-2, -1, 0, 1, 2].map(v => [v, v === 0 ? 'Average' : `${v > 0 ? '+' : ''}${v} sd`]),
    yTicks: [-20, 0, 20, 40].map(v => [v, `${v}%`]),
    xLabel: 'Brand spend held for the year', yLabel: 'Sales uplift over the year',
  });
  hline(f, 0, '#6b7280', '4,3');
  bandArea(f, xs, rc.map(r => r.lo), rc.map(r => r.hi), COLOR.sales, 0.16);
  line(f, xs, rc.map(r => r.mean), COLOR.sales);
  for (const r of rc) f.g.appendChild(el('circle', { cx: f.sx(r.spend_sd), cy: f.sy(r.mean), r: narrow ? 2.4 : 3, fill: COLOR.sales }));
}

// ── 7 · Return per channel ───────────────────────────────────────────────────
function drawReturn(svg, narrow) {
  const rp = DATA.return_per_channel;
  const f = frame(svg, narrow, {
    hW: 280, hN: 260, x: [0, 260], y: [0, 15000],
    xTicks: [[0, '0'], [52, '1 year'], [156, '3 years'], [260, '5 years']],
    xTicksN: [[0, '0'], [52, '1y'], [156, '3y'], [260, '5y']],
    yTicks: [0, 5000, 10000, 15000].map(v => [v, v === 0 ? '£0' : `£${v / 1000}k`]),
    xLabel: 'Weeks after the one-week pulse', yLabel: 'Revenue per unit',
  });
  for (const w of [52, 156, 260]) {
    const x = f.sx(w);
    f.g.appendChild(el('line', { x1: x, y1: f.m.top, x2: x, y2: f.m.top + f.ph, stroke: '#d1d5db', 'stroke-width': 1, 'stroke-dasharray': '2,3' }));
  }
  const spec = [['brand_spend', COLOR.brand], ['pr_reach', COLOR.pr], ['sponsorship_reach', COLOR.sponsorship]];
  const xs = rp.week.map(w => w - 1);
  for (const [k, col] of spec) {
    const ch = rp.channels[k], s = ch.cost_gbp ? ch.cost_gbp : 1;   // back to £ per unit
    bandArea(f, xs, ch.path.lo.map(v => v * s), ch.path.hi.map(v => v * s), col, 0.12);
  }
  for (const [k, col] of spec) {
    const ch = rp.channels[k], s = ch.cost_gbp ? ch.cost_gbp : 1;
    line(f, xs, ch.path.mean.map(v => v * s), col);
    line(f, xs, ch.truth.map(v => v * s), COLOR.truth, { dash: '4,3', width: narrow ? 1 : 1.2, opacity: 0.8 });
  }
}

chart('chartSpurious', '0 0 700 240', s => drawSpurious(s, false), s => drawSpurious(s, true));
chart('chartHoldout', '0 0 700 260', s => drawHoldout(s, false), s => drawHoldout(s, true));
chart('chartSeries', '0 0 700 260', s => drawSeries(s, false), s => drawSeries(s, true));
chart('chartIllus', '0 0 700 260', s => drawIllus(s, false), s => drawIllus(s, true));
chart('chartGaps', '0 0 700 240', s => drawGaps(s, false), s => drawGaps(s, true));
chart('chartAlpha', '0 0 700 300', s => drawAlpha(s, false), s => drawAlpha(s, true));
chart('chartIrf', '0 0 700 260', s => drawIrf(s, false), s => drawIrf(s, true));
chart('chartCum', '0 0 700 260', s => drawCum(s, false), s => drawCum(s, true));
chart('chartChannels', '0 0 700 200', s => drawChannels(s, false), s => drawChannels(s, true));
chart('chartCampaign', '0 0 700 260', s => drawCampaign(s, false), s => drawCampaign(s, true));
chart('chartCurve', '0 0 700 260', s => drawCurve(s, false), s => drawCurve(s, true));
chart('chartReturn', '0 0 700 280', s => drawReturn(s, false), s => drawReturn(s, true));

renderCharts();
window.addEventListener('resize', renderCharts);
