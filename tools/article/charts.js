// Draws the article's charts as inline SVG from the specs in the DATA block.
// Each <div class="panels" id="NAME"> gets the chart NAME. Charts redraw on resize,
// so axis text stays at its real size on phones and desktops.
(function () {
  const DATA = JSON.parse(document.getElementById("DATA").textContent);
  const COL = {
    ink: "#111827", muted: "#6b7280", grey: "#d1d5db", light: "#e5e7eb",
    blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", purple: "#7c3aed",
  };
  const NS = "http://www.w3.org/2000/svg";
  const col = (c) => COL[c] || c || COL.ink;

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function fmtY(v, fmt) {
    if (fmt === "pct") return (v > 0 ? "+" : "") + Math.round(v) + "%";
    if (fmt === "pct_plain") return Math.round(v) + "%";
    return Math.abs(v) >= 10 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(1);
  }

  function fmtX(v, kind) {
    if (kind === "week") return "Week " + Math.round(v);
    if (kind === "gbp_week") return "£" + Math.round(v).toLocaleString("en-GB") + " a week";
    return String(Math.round(v * 100) / 100);
  }

  function extent(panel) {
    let lo = Infinity, hi = -Infinity;
    for (const L of panel.layers) {
      const vals = L.type === "band" || L.type === "interval" ? L.lo.concat(L.hi) : L.type === "line" || L.type === "bars" ? L.y : [];
      for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    return [lo, hi];
  }

  function legend(spec) {
    if (!spec.legend) return null;
    const box = document.createElement("div");
    box.className = "chart-legend";
    for (const it of spec.legend) {
      const s = document.createElement("span");
      const c = col(it.color);
      let sw;
      if (it.kind === "dash") sw = `<svg width="20" height="4"><line x1="0" y1="2" x2="20" y2="2" stroke="${c}" stroke-width="1.6" stroke-dasharray="4 3"/></svg>`;
      else if (it.kind === "bar") sw = `<svg width="12" height="12"><rect width="12" height="12" rx="2" fill="${c}"/></svg>`;
      else if (it.kind === "band") sw = `<svg width="16" height="10"><rect width="16" height="10" rx="2" fill="${c}" fill-opacity="0.22"/></svg>`;
      else sw = `<svg width="20" height="4"><line x1="0" y1="2" x2="20" y2="2" stroke="${c}" stroke-width="2.6" stroke-linecap="round"/></svg>`;
      s.innerHTML = sw + `<span>${it.label}</span>`;
      box.appendChild(s);
    }
    return box;
  }

  function drawPanel(holder, spec, panel, width, yRange, showYLabel) {
    const H = spec.height || 240;
    const hasTitle = !!panel.title;
    const top = (hasTitle ? 22 : 8) + (panel.subtitle ? 16 : 0);
    const yTicks = Array.isArray(panel.y.ticks) ? panel.y.ticks : niceTicks(yRange[0], yRange[1], 5);
    const hasYTicks = yTicks.length > 0;
    const yLabel = showYLabel ? panel.y.label : "";
    const m = { t: top, r: 10, b: 40, l: (hasYTicks ? 34 : 6) + (yLabel ? 18 : 0) };
    const svg = el("svg", { viewBox: `0 0 ${width} ${H}`, width: width, height: H, class: "chart", role: "img",
      "aria-label": panel.title || spec.title });
    holder.appendChild(svg);
    const iw = width - m.l - m.r, ih = H - m.t - m.b;
    const xs = (v) => m.l + ((v - panel.x.min) / (panel.x.max - panel.x.min)) * iw;
    const ys = (v) => m.t + ih - ((v - yRange[0]) / (yRange[1] - yRange[0])) * ih;

    if (hasTitle) el("text", { x: 0, y: 14, class: "p-title", fill: panel.titleColor ? col(panel.titleColor) : COL.ink }, svg, panel.title);
    if (panel.subtitle) el("text", { x: 0, y: 30, class: "p-sub" }, svg, panel.subtitle);

    for (const v of yTicks) {
      el("line", { x1: m.l, x2: m.l + iw, y1: ys(v), y2: ys(v), stroke: COL.light, "stroke-width": 1 }, svg);
      el("text", { x: m.l - 6, y: ys(v) + 4, class: "tick", "text-anchor": "end" }, svg, fmtY(v, panel.y.fmt));
    }
    if (yLabel) el("text", { class: "ax-label", transform: `translate(11 ${m.t + ih / 2}) rotate(-90)`, "text-anchor": "middle" }, svg, yLabel);
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: "#c9ced6", "stroke-width": 1 }, svg);
    for (const [v, lab] of panel.x.ticks || []) {
      el("line", { x1: xs(v), x2: xs(v), y1: m.t + ih, y2: m.t + ih + 4, stroke: "#c9ced6" }, svg);
      el("text", { x: xs(v), y: m.t + ih + 17, class: "tick", "text-anchor": v === panel.x.min ? "start" : "middle" }, svg, lab);
    }
    if (panel.x.label) el("text", { x: m.l + iw / 2, y: H - 4, class: "ax-label", "text-anchor": "middle" }, svg, panel.x.label);

    const clipId = "c" + Math.random().toString(36).slice(2, 9);
    const clip = el("clipPath", { id: clipId }, el("defs", {}, svg));
    el("rect", { x: m.l, y: m.t - 2, width: iw, height: ih + 2 }, clip);
    const g = el("g", { "clip-path": `url(#${clipId})` }, svg);
    const hoverLines = [];

    for (const L of panel.layers) {
      if (L.type === "hspan") el("rect", { x: m.l, width: iw, y: ys(L.hi), height: ys(L.lo) - ys(L.hi), fill: "#f1f2f4" }, g);
      if (L.type === "hline") el("line", { x1: m.l, x2: m.l + iw, y1: ys(L.y), y2: ys(L.y), stroke: COL.muted, "stroke-width": 1, "stroke-dasharray": L.dash ? "4 3" : undefined }, g);
      if (L.type === "vline") {
        el("line", { x1: xs(L.x), x2: xs(L.x), y1: m.t, y2: m.t + ih, stroke: COL.muted, "stroke-width": 1, "stroke-dasharray": "2 3" }, g);
        if (L.label) el("text", { x: xs(L.x) + 4, y: m.t + 10, class: "note-t" }, svg, L.label);
      }
      if (L.type === "bars") {
        const bw = L.barWidth ? Math.abs(xs(L.barWidth) - xs(0)) : Math.max(iw / L.x.length, 1);
        const base = L.base !== undefined ? L.base : 0;
        let d = "";
        const dx = L.offset || 0;
        L.x.forEach((x, i) => { if (L.y[i] > base) d += `M${xs(x + dx) - bw / 2},${ys(base)}V${ys(L.y[i])}h${bw}V${ys(base)}Z`; });
        el("path", { d, fill: col(L.color) }, g);
      }
      if (L.type === "band") {
        let d = L.x.map((x, i) => `${i ? "L" : "M"}${xs(x)},${ys(L.hi[i])}`).join("");
        d += L.x.map((x, i) => [x, i]).reverse().map(([x, i]) => `L${xs(x)},${ys(L.lo[i])}`).join("") + "Z";
        el("path", { d, fill: col(L.color), "fill-opacity": 0.16 }, g);
      }
      if (L.type === "line") {
        const d = L.x.map((x, i) => `${i ? "L" : "M"}${xs(x).toFixed(1)},${ys(L.y[i]).toFixed(1)}`).join("");
        el("path", { d, fill: "none", stroke: col(L.color), "stroke-width": L.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": L.dash ? "5 4" : undefined }, g);
        if (L.label) {
          const n = L.x.length - 1;
          el("text", { x: Math.min(xs(L.x[n]), m.l + iw) - 2, y: ys(L.y[n]) - 7, class: "line-lab", "text-anchor": "end", fill: col(L.color) }, svg, L.label);
        }
        if (L.name) hoverLines.push(L);
      }
      if (L.type === "interval") {
        // One vertical range per category, with a dot at the middle value and a native tooltip.
        const dx = L.offset || 0;
        L.x.forEach((x, i) => {
          const cx = xs(x + dx);
          const mark = el("g", {}, svg);
          el("line", { x1: cx, x2: cx, y1: ys(L.lo[i]), y2: ys(L.hi[i]), stroke: col(L.color), "stroke-width": 6, "stroke-opacity": 0.3, "stroke-linecap": "round" }, mark);
          el("circle", { cx, cy: ys(L.mid[i]), r: 4.5, fill: col(L.color) }, mark);
          if (L.tips) el("title", {}, mark, L.tips[i]);
        });
      }
      if (L.type === "text") el("text", { x: xs(L.x), y: ys(L.y), class: "note-t", fill: col(L.color), "text-anchor": L.anchor || "start", "font-weight": L.weight }, svg, L.text);
    }

    hoverLines.sort((p, q) => (p.dash ? 1 : 0) - (q.dash ? 1 : 0));
    if (hoverLines.length && panel.x.hover !== false) {
      const guide = el("line", { y1: m.t, y2: m.t + ih, stroke: COL.ink, "stroke-width": 1, opacity: 0, "pointer-events": "none" }, svg);
      const dots = hoverLines.map((L) => el("circle", { r: 3.5, fill: col(L.color), opacity: 0, "pointer-events": "none" }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      const tip = holder.closest(".chart-card").querySelector(".tip");
      const hide = () => { guide.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); tip.style.opacity = 0; };
      const move = (ev) => {
        const r = svg.getBoundingClientRect();
        const px = ((ev.clientX - r.left) / r.width) * width;
        const xv = panel.x.min + ((px - m.l) / iw) * (panel.x.max - panel.x.min);
        const X = hoverLines[0].x;
        let i = 0;
        for (let k = 1; k < X.length; k++) if (Math.abs(X[k] - xv) < Math.abs(X[i] - xv)) i = k;
        guide.setAttribute("x1", xs(X[i])); guide.setAttribute("x2", xs(X[i])); guide.setAttribute("opacity", 0.25);
        let rows = `<b>${fmtX(X[i], panel.x.hover)}</b>`;
        hoverLines.forEach((L, k) => {
          dots[k].setAttribute("cx", xs(L.x[i])); dots[k].setAttribute("cy", ys(L.y[i])); dots[k].setAttribute("opacity", 1);
          const band = L.band ? ` <span class="tip-m">(${fmtY(L.band[0][i], panel.y.fmt)} to ${fmtY(L.band[1][i], panel.y.fmt)})</span>` : "";
          rows += `<div><i style="background:${col(L.color)}"></i>${L.name}: ${L.y[i].toFixed(L.y[i] >= 10 || panel.y.fmt === "pct" ? 0 : 2)}${panel.y.fmt === "pct" ? "%" : ""}${band}</div>`;
        });
        tip.innerHTML = rows;
        const card = holder.closest(".chart-card").getBoundingClientRect();
        const left = ev.clientX - card.left, topPx = ev.clientY - card.top;
        tip.style.opacity = 1;
        tip.style.left = Math.min(Math.max(left + 14, 8), card.width - tip.offsetWidth - 8) + "px";
        tip.style.top = Math.max(topPx - tip.offsetHeight - 12, 4) + "px";
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", hide);
    }
  }

  function draw(id) {
    const spec = DATA[id];
    const box = document.getElementById(id);
    if (!spec || !box) return;
    box.innerHTML = "";
    const lg = legend(spec);
    if (lg) box.appendChild(lg);
    const grid = document.createElement("div");
    grid.className = "panel-grid";
    box.appendChild(grid);
    const W = box.clientWidth || 660;
    const n = spec.panels.length;
    const minW = spec.minPanel || 240;
    const cols = Math.max(1, Math.min(n, Math.floor((W + 16) / (minW + 16))));
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    const pw = Math.floor((W - 16 * (cols - 1)) / cols);
    let shared = null;
    if (spec.shareY) {
      const ex = spec.panels.map(extent);
      shared = [Math.min(...ex.map((e) => e[0])), Math.max(...ex.map((e) => e[1]))];
    }
    spec.panels.forEach((p, k) => {
      let [lo, hi] = shared || extent(p);
      const pad = 0.06 * (hi - lo || 1);
      lo = p.y.min !== undefined ? p.y.min : lo - pad;
      hi = p.y.max !== undefined ? p.y.max : hi + pad;
      const cell = document.createElement("div");
      grid.appendChild(cell);
      drawPanel(cell, spec, p, pw, [lo, hi], !spec.shareY || k % cols === 0);
    });
  }

  function drawAll() { for (const id in DATA) draw(id); }
  let last = 0;
  window.addEventListener("resize", () => {
    const w = window.innerWidth;
    if (Math.abs(w - last) < 30) return;
    last = w;
    drawAll();
  });
  last = window.innerWidth;
  drawAll();
})();
