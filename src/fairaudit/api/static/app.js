"use strict";

const $ = (selector) => document.querySelector(selector);
const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
const pct = (v, digits = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(digits)}%`);
const num = (v, digits = 3) => (v == null ? "n/a" : v.toFixed(digits));
const truncate = (text, max = 16) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

const RATE_SERIES = [
  { key: "selection_rate", label: "Selection rate", color: "var(--series-1)" },
  { key: "tpr", label: "True-positive rate", color: "var(--series-2)" },
  { key: "fpr", label: "False-positive rate", color: "var(--series-3)" },
];
const SAMPLE_MAPPING = {
  label: "label",
  prediction: "pred_logreg",
  score: "score_logreg",
  sensitive: ["gender", "age_50_70", "view_position"],
};

let currentFile = null;
let studyLoaded = false;
let lastReport = null;
let lastStudy = null;

// Charts are drawn at the container's pixel width so their text stays at 12px on every screen.
const chartWidth = (el) => Math.max(260, Math.min(900, el.clientWidth - 44));

/* ---------- tabs ---------- */

function selectTab(name) {
  for (const tab of ["audit", "study"]) {
    const active = tab === name;
    $(`#tab-${tab}`).setAttribute("aria-selected", String(active));
    $(`#panel-${tab}`).hidden = !active;
  }
  if (name === "study" && !studyLoaded) loadStudy();
}

/* ---------- CSV header & column mapping ---------- */

function parseHeader(text) {
  const line = text.split(/\r?\n/, 1)[0] ?? "";
  const cells = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cell += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { cells.push(cell); cell = ""; }
    else cell += ch;
  }
  cells.push(cell);
  return cells.filter((c) => c !== "");
}

const guess = (columns, pattern) => columns.find((c) => pattern.test(c)) ?? "";

function fillSelect(select, columns, value, optional) {
  const options = optional ? ['<option value="">(none)</option>'] : [];
  for (const c of columns) options.push(`<option value="${esc(c)}">${esc(c)}</option>`);
  select.innerHTML = options.join("");
  select.value = value;
}

function fillMapping(columns, preset) {
  const label = preset?.label ?? guess(columns, /^(label|y|y_true|target|outcome|truth)$/i) ?? columns[0];
  const prediction = preset?.prediction ?? guess(columns, /^(pred|prediction|y_pred)/i);
  const score = preset?.score ?? guess(columns, /^(score|proba|prob|probability|y_score)/i);
  fillSelect($("#label"), columns, label || columns[0], false);
  fillSelect($("#prediction"), columns, prediction, true);
  fillSelect($("#score"), columns, score, true);

  const used = new Set([label, prediction, score]);
  const sensitive = preset?.sensitive
    ?? columns.filter((c) => !used.has(c) && /(gender|sex|race|ethnic|age|group|region)/i.test(c));
  $("#sensitive").innerHTML = columns
    .map((c) => `<label><input type="checkbox" value="${esc(c)}"${sensitive.includes(c) ? " checked" : ""}> ${esc(c)}</label>`)
    .join("");
  $("#mapping").disabled = false;
}

async function loadFile(file, name, preset) {
  currentFile = file;
  $("#file-name").textContent = name;
  const columns = parseHeader(await file.slice(0, 65536).text());
  if (columns.length < 2) return showError("Could not read a header row from this file.");
  hideError();
  fillMapping(columns, preset);
}

async function useSample() {
  const response = await fetch("/case-study/sample.csv");
  if (!response.ok) return showError("The case-study sample is not available on this server.");
  const blob = await response.blob();
  const file = new File([blob], "chestxray_test.csv", { type: "text/csv" });
  selectTab("audit");
  await loadFile(file, "chestxray_test.csv (ChestX-ray14 test patients)", SAMPLE_MAPPING);
}

function showError(message) {
  const el = $("#form-error");
  el.textContent = message;
  el.hidden = false;
}

function hideError() {
  $("#form-error").hidden = true;
}

function errorMessage(body) {
  if (typeof body?.detail === "string") return body.detail;
  if (Array.isArray(body?.detail)) return body.detail.map((e) => `${(e.loc ?? []).join(".")}: ${e.msg}`).join("; ");
  return "The audit failed.";
}

async function runAudit(event) {
  event.preventDefault();
  hideError();
  if (!currentFile) return showError("Choose a CSV file first.");
  const sensitive = [...document.querySelectorAll("#sensitive input:checked")].map((i) => i.value);
  if (!sensitive.length) return showError("Select at least one sensitive attribute.");
  if (!$("#prediction").value && !$("#score").value) return showError("Select a prediction column, a score column, or both.");

  const form = new FormData();
  form.append("file", currentFile);
  form.append("label", $("#label").value);
  for (const field of ["prediction", "score"]) if ($(`#${field}`).value) form.append(field, $(`#${field}`).value);
  if ($("#positive-label").value.trim()) form.append("positive_label", $("#positive-label").value.trim());
  form.append("threshold", $("#threshold").value || "0.5");
  for (const column of sensitive) form.append("sensitive", column);

  const button = $("#run");
  button.disabled = true;
  button.textContent = "Auditing…";
  try {
    const response = await fetch("/audit/csv", { method: "POST", body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) return showError(errorMessage(body));
    lastReport = body;
    $("#audit-result").innerHTML = renderReport(body);
    $("#audit-result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(`Request failed: ${err.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Run audit";
  }
}

/* ---------- charts (inline SVG) ---------- */

function barPath(x0, y, width, height, color) {
  if (width <= 0) return "";
  const r = Math.min(4, width, height / 2); // rounded data end, square at the baseline
  return `<path d="M${x0},${y}h${width - r}a${r},${r} 0 0 1 ${r},${r}v${height - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(width - r)}z" style="fill:${color}"/>`;
}

function axis(x, top, bottom, ticks, format) {
  return ticks
    .map((t) => `<line class="${t === 0 ? "baseline" : "gridline"}" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${bottom}"/>`
      + `<text x="${x(t)}" y="${bottom + 16}" text-anchor="middle">${format(t)}</text>`)
    .join("");
}

function legend(series) {
  return `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("")}</div>`;
}

function groupedBars(groups, series, ariaLabel, width) {
  const labelWidth = Math.min(120, Math.round(width * 0.22)), rightPad = 20, top = 6;
  const barHeight = 10, gap = 2, groupGap = 14;
  const rowHeight = series.length * (barHeight + gap) - gap;
  const plotBottom = top + groups.length * (rowHeight + groupGap) - groupGap + 4;
  const height = plotBottom + 22;
  const x = (v) => labelWidth + v * (width - labelWidth - rightPad);
  const maxChars = Math.max(4, Math.floor((labelWidth - 10) / 7));
  let body = axis(x, 0, plotBottom, [0, 0.25, 0.5, 0.75, 1], (t) => `${t * 100}%`);
  groups.forEach((g, i) => {
    const y0 = top + i * (rowHeight + groupGap);
    body += `<text x="${labelWidth - 8}" y="${y0 + rowHeight / 2 + 4}" text-anchor="end">${esc(truncate(g.group, maxChars))}</text>`;
    series.forEach((s, j) => {
      const y = y0 + j * (barHeight + gap);
      const v = g[s.key];
      const tip = `${g.group} · ${s.label}: ${pct(v)}`;
      body += barPath(x(0), y, v == null ? 0 : v * (x(1) - x(0)), barHeight, s.color);
      body += `<rect class="hit" x="${labelWidth}" y="${y - 1}" width="${x(1) - labelWidth}" height="${barHeight + 2}" data-tip="${esc(tip)}"/>`;
    });
  });
  return `<div class="chart">${legend(series)}<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}">${body}</svg></div>`;
}

function ratioBars(rows, ruleAt, ariaLabel, width) {
  // Label above each bar, so long experiment names never need truncating.
  const left = 4, rightPad = 44, labelHeight = 16, barHeight = 14, gap = 12;
  const rowHeight = labelHeight + barHeight;
  const plotBottom = rows.length * (rowHeight + gap) - gap + 4;
  const height = plotBottom + 22;
  const x = (v) => left + v * (width - left - rightPad);
  let body = axis(x, 0, plotBottom, [0, 0.2, 0.4, 0.6, 0.8, 1], (t) => t.toFixed(1));
  body += `<line class="rule" x1="${x(ruleAt)}" x2="${x(ruleAt)}" y1="0" y2="${plotBottom}"/>`;
  rows.forEach((row, i) => {
    const y = i * (rowHeight + gap);
    body += `<text class="value" x="${left}" y="${y + 12}">${esc(row.label)}</text>`;
    body += barPath(x(0), y + labelHeight, (row.value ?? 0) * (x(1) - x(0)), barHeight, "var(--series-1)");
    body += `<text class="value" x="${x(row.value ?? 0) + 6}" y="${y + labelHeight + barHeight / 2 + 4}">${num(row.value, 2)}</text>`;
    body += `<rect class="hit" x="${left}" y="${y}" width="${x(1) - left}" height="${rowHeight}" data-tip="${esc(`${row.label}: ${num(row.value, 3)}`)}"/>`;
  });
  return `<div class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}">${body}</svg></div>`;
}

function bindTooltips() {
  const tooltip = $("#tooltip");
  document.addEventListener("mousemove", (event) => {
    const target = event.target.closest?.("[data-tip]");
    if (!target) { tooltip.hidden = true; return; }
    tooltip.textContent = target.getAttribute("data-tip");
    tooltip.hidden = false;
    const { innerWidth } = window;
    const left = Math.min(event.clientX + 12, innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  });
}

/* ---------- audit report ---------- */

const tile = (k, v) => `<div class="tile"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;

function table(headers, rows, highlight = () => false) {
  const head = headers.map((h) => `<th>${esc(h)}</th>`).join("");
  const body = rows
    .map((r, i) => `<tr${highlight(i) ? ' class="highlight"' : ""}>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function verdict(attr, policy) {
  if (attr.passes_four_fifths_rule == null) return '<span class="badge">DI n/a</span>';
  const ok = attr.passes_four_fifths_rule;
  return `<span class="badge ${ok ? "pass" : "fail"}">${ok ? "✓ Passes" : "✕ Fails"} four-fifths rule · DI ${num(attr.disparate_impact_ratio, 2)} (limit ${policy.min_disparate_impact})</span>`;
}

function renderAttribute(attr, policy) {
  const findings = attr.findings.length
    ? `<ul class="findings">${attr.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`
    : '<p class="hint">No metric breaches the policy.</p>';
  const summary = table(
    ["Metric", "Value"],
    [
      ["Disparate impact ratio (min / max selection rate)", num(attr.disparate_impact_ratio)],
      ["Demographic parity difference", num(attr.demographic_parity_difference)],
      ["Equal opportunity difference (TPR gap)", num(attr.equal_opportunity_difference)],
      ["False-positive-rate difference", num(attr.false_positive_rate_difference)],
      ["Equalized odds difference", num(attr.equalized_odds_difference)],
      ["Predictive parity difference (PPV gap)", num(attr.predictive_parity_difference)],
      ["Base-rate difference (in the labels)", num(attr.base_rate_difference)],
      ["Calibration gap (ECE max - min)", num(attr.calibration_gap)],
    ],
  );
  const groups = table(
    ["Group", "n", "Base rate", "Selection rate", "TPR", "FPR", "PPV", "Accuracy", "ECE"],
    attr.groups.map((g) => [
      g.small_sample ? `${g.group} (small)` : g.group, g.n, pct(g.base_rate), pct(g.selection_rate),
      pct(g.tpr), pct(g.fpr), pct(g.ppv), pct(g.accuracy), num(g.ece),
    ]),
  );
  return `<article class="card">
    <div class="attr-head"><h3>${esc(attr.attribute)}</h3>${verdict(attr, policy)}</div>
    ${findings}
    ${groupedBars(attr.groups, RATE_SERIES, `Selection rate, TPR and FPR by ${attr.attribute}`, chartWidth($("#audit-result")))}
    ${groups}
    <details><summary>All disparity metrics</summary>${summary}</details>
  </article>`;
}

function renderReport(report) {
  const o = report.overall;
  const tiles = [
    tile("Samples", report.n_samples.toLocaleString()),
    tile("Accuracy", num(o.accuracy)),
    tile("Balanced accuracy", num(o.balanced_accuracy)),
    o.roc_auc != null ? tile("ROC AUC", num(o.roc_auc)) : "",
    tile("Selection rate", pct(o.selection_rate)),
    tile("Base rate", pct(o.base_rate)),
  ].join("");
  const warnings = report.warnings.length
    ? `<div class="banner">${report.warnings.map(esc).join("<br>")}</div>`
    : "";
  const mode = report.threshold == null ? "binary predictions supplied" : `predictions = score ≥ ${report.threshold}`;
  return `<section class="card"><h2>Report</h2>
      <p class="hint">${esc(mode)} · ${report.findings.length} finding(s) · fairaudit ${esc(report.fairaudit_version)}</p>
      <div class="tiles">${tiles}</div></section>
    ${warnings}
    ${report.attributes.map((a) => renderAttribute(a, report.policy)).join("")}`;
}


/* ---------- case study ---------- */

async function loadStudy() {
  studyLoaded = true;
  const el = $("#study-result");
  try {
    const response = await fetch("/case-study");
    const body = await response.json();
    if (!response.ok) throw new Error(errorMessage(body));
    lastStudy = body;
    showStudy();
  } catch (err) {
    studyLoaded = false;
    el.innerHTML = `<p class="error">${esc(err.message)}</p>`;
  }
}

function showStudy() {
  $("#study-result").innerHTML = renderStudy(lastStudy);
  $("#study-sample")?.addEventListener("click", useSample);
}

function renderStudy(study) {
  const cfg = study.config;
  const target = cfg.target_attribute;
  const ds = study.dataset;
  const rows = {};
  for (const r of study.comparison) (rows[r.experiment] ??= {})[r.attribute] = r;
  const experiments = study.experiments;
  const passing = experiments.filter((e) => (rows[e.name][target].disparate_impact_ratio ?? 0) >= cfg.policy.min_disparate_impact);
  const best = passing.reduce((a, b) => ((rows[b.name][target].accuracy ?? 0) > (rows[a?.name]?.[target].accuracy ?? -1) ? b : a), null);

  const chart = ratioBars(
    experiments.map((e) => ({ label: e.title, value: rows[e.name][target].disparate_impact_ratio })),
    cfg.policy.min_disparate_impact,
    `Disparate impact ratio on ${target} by experiment`,
    chartWidth($("#study-result")),
  );
  const comparison = table(
    ["Experiment", "Stage", "Accuracy", "ROC AUC", "DI ratio", "DP diff.", "EO diff.", "EOdds diff."],
    experiments.map((e) => {
      const r = rows[e.name][target];
      return [e.title, e.stage, num(r.accuracy), num(r.roc_auc), num(r.disparate_impact_ratio, 2),
        num(r.demographic_parity_difference), num(r.equal_opportunity_difference), num(r.equalized_odds_difference)];
    }),
    (i) => best && experiments[i].name === best.name,
  );
  const matrix = table(
    ["Experiment", ...cfg.audit_attributes],
    experiments.map((e) => [e.title, ...cfg.audit_attributes.map((a) => num(rows[e.name][a].disparate_impact_ratio, 2))]),
  );
  const baseRates = table(
    ["Attribute", "Group", "Images", "Share", "Positive rate"],
    Object.entries(study.label_base_rates).flatMap(([a, groups]) => groups.map((g) => [a, g.group, g.n.toLocaleString(), pct(g.share), pct(g.base_rate)])),
  );
  const details = experiments.map((e) => {
    const attr = e.report.attributes.find((a) => a.attribute === target);
    const items = attr.findings.length ? attr.findings.map((f) => `<li>${esc(f)}</li>`).join("") : "<li>No metric breaches the policy.</li>";
    return `<details><summary>${esc(e.title)}</summary><p class="hint">${esc(e.description)}</p><ul class="findings">${items}</ul></details>`;
  }).join("");

  return `<div class="banner"><strong>Disclaimer.</strong> These models only see image metadata (age, sex, view position, image size, follow-up number), never the X-ray. This is a fairness case study, not a clinical tool.</div>
    <section class="card">
      <h2>Accuracy vs fairness on <code>${esc(target)}</code></h2>
      <p class="hint">${ds.n_rows.toLocaleString()} NIH ChestX-ray14 images of ${ds.n_patients.toLocaleString()} patients, split by patient (seed ${cfg.seed}); metrics on ${study.splits.test.n_rows.toLocaleString()} held-out test images. Mitigations target <code>${esc(target)}</code>; the vertical rule marks the four-fifths rule (${cfg.policy.min_disparate_impact}).</p>
      ${chart}
      ${comparison}
      ${best ? `<p class="hint">Bold: best accuracy among experiments passing the four-fifths rule (${esc(best.title)}). Accuracy differences of about 0.01 or less are within sampling noise.</p>` : ""}
      <div class="row"><button type="button" id="study-sample" class="secondary">Audit the test predictions yourself</button>
        <a href="/case-study/report">Markdown report</a><a href="/case-study/model-card">Model card</a><a href="/case-study/sample.csv">Test predictions (CSV)</a><a href="/case-study">JSON</a></div>
    </section>
    <section class="card"><h2>Disparate impact ratio on every audited attribute</h2>
      <p class="hint">Mitigating one attribute says nothing about the others.</p>${matrix}</section>
    <section class="card"><h2>Disparities already present in the labels</h2>${baseRates}</section>
    <section class="card"><h2>Findings per experiment (<code>${esc(target)}</code>)</h2>${details}</section>`;
}

/* ---------- wiring ---------- */

document.addEventListener("DOMContentLoaded", () => {
  $("#tab-audit").addEventListener("click", () => selectTab("audit"));
  $("#tab-study").addEventListener("click", () => selectTab("study"));
  $("#file").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) loadFile(file, file.name);
  });
  $("#use-sample").addEventListener("click", useSample);
  $("#audit-form").addEventListener("submit", runAudit);
  bindTooltips();
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (lastReport) $("#audit-result").innerHTML = renderReport(lastReport);
      if (lastStudy && !$("#panel-study").hidden) showStudy();
    }, 200);
  });
});
