// Diabetes Risk Check: chat page (Coffee-house menu design).
const $ = id => document.getElementById(id);
const log = $("log"), form = $("form"), input = $("input"), send = $("send");

const MODEL_NAMES = { mlp: "Neural net", naive_bayes: "Naive Bayes" };
const DATASET_NAMES = {
  nhanes: "US adults, fasting sugar (recommended)",
  pima: "Pima women, 2-hour sugar test",
  kaggle: "Kaggle set (comparison only)",
};
const DATASET_SHORT = { nhanes: "US adults (NHANES)", pima: "Pima women", kaggle: "Kaggle set" };
const FIELD_NAMES = {
  age: "Age", glucose: "Sugar (glucose)", bmi: "BMI", insulin: "Insulin",
  blood_pressure: "Blood pressure (lower)", pregnancies: "Pregnancies",
  skin_thickness: "Skin-fold (mm)", pedigree: "Family-history score",
};
const UNITS = { age: "yrs", glucose: "mg/dL", insulin: "µU/mL", blood_pressure: "mm Hg", skin_thickness: "mm" };
const RING = 2 * Math.PI * 38;
let config = null;

const store = (k, v) => { try { localStorage.setItem(k, v); } catch (_) {} };
const load = k => { try { return localStorage.getItem(k); } catch (_) { return null; } };

let sessionId = null;
try { sessionId = sessionStorage.getItem("sessionId"); } catch (_) {}
// Random id made once per browser, used only to count unique visitors at /stats.
let visitorId = load("visitorId");
if (!visitorId) {
  visitorId = crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now();
  store("visitorId", visitorId);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const fieldName = id => FIELD_NAMES[id] || id;
const pct = p => `${Math.round(p * 100)}%`;
function scrollDown() { window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }); }

function bubble(cls, text) {
  const b = el("div", `bubble ${cls}`);
  if (text !== undefined) b.appendChild(el("p", null, text));
  log.appendChild(b);
  scrollDown();
  return b;
}

function ring(p, high) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 92 92");
  svg.setAttribute("aria-hidden", "true");
  const track = document.createElementNS(NS, "circle");
  const arc = document.createElementNS(NS, "circle");
  [track, arc].forEach(c => { c.setAttribute("cx", 46); c.setAttribute("cy", 46); c.setAttribute("r", 38); c.setAttribute("fill", "none"); c.setAttribute("stroke-width", 10); });
  track.setAttribute("stroke", "#EFE3D0");
  arc.setAttribute("stroke", high ? "#9E4630" : "#4F5D33");
  arc.setAttribute("stroke-linecap", "round");
  arc.setAttribute("stroke-dasharray", `${Math.max(p * RING, 0.01)} ${RING}`);
  arc.setAttribute("transform", "rotate(-90 46 46)");
  svg.append(track, arc);
  return svg;
}

function menuItem(name, value, unit, emphasis) {
  const row = el("div", "item");
  row.appendChild(el("span", "name", name));
  row.appendChild(el("span", "dots"));
  const v = el("span", "val" + (emphasis ? " hi" : ""), String(value));
  if (unit) { v.appendChild(document.createTextNode(" ")); v.appendChild(el("small", null, unit)); }
  row.appendChild(v);
  return row;
}

function readingCard(r) {
  const high = r.risk_class === "high";
  const card = el("article", `reading ${high ? "high" : "low"}`);
  const top = el("div", "top");
  const cup = el("div", "cup");
  cup.appendChild(ring(r.probability, high));
  cup.appendChild(el("div", "pct", pct(r.probability)));
  top.appendChild(cup);
  const head = el("div");
  head.appendChild(el("div", "k", r.what_if ? "What if · estimated risk" : "Estimated diabetes risk"));
  head.appendChild(el("div", "v", high ? "Higher" : "Lower"));
  const rows = config && config.datasets.find(d => d.id === r.dataset);
  head.appendChild(el("p", "note", `${MODEL_NAMES[r.model_used] || r.model_used}, learned from ${rows ? rows.rows.toLocaleString() + " " : ""}${DATASET_SHORT[r.dataset] || r.dataset || ""}`));
  top.appendChild(head);
  card.appendChild(top);

  if (r.comparison) {
    const cmp = el("div", "compare-row");
    Object.entries(r.comparison.results).forEach(([name, res]) => {
      const box = el("div");
      box.appendChild(el("b", null, pct(res.probability)));
      box.appendChild(document.createTextNode(MODEL_NAMES[name] || name));
      cmp.appendChild(box);
    });
    card.appendChild(cmp);
    card.appendChild(el("p", "tip", r.comparison.agree ? "Both models agree." : "The two models disagree, so treat this result with extra care."));
  }

  const used = r.inputs_used || {};
  const menu = el("div", "menu");
  menu.appendChild(el("h3", "menu-h", r.what_if ? "Values in this what-if" : "Your values"));
  Object.entries(used).forEach(([k, v]) => {
    if (v === null || v === undefined) return;
    menu.appendChild(menuItem(fieldName(k), v, UNITS[k], k === "glucose" && v >= 126));
  });
  card.appendChild(menu);

  const explain = el("div", "explain");
  r.explanation.split(/\n+/).filter(Boolean).forEach(p => explain.appendChild(el("p", null, p)));
  card.appendChild(explain);

  if (r.more_values_could_help) {
    const h = r.more_values_could_help;
    card.appendChild(el("p", "tip", `Tip: adding your ${h.ask_for.map(f => fieldName(f).toLowerCase()).join(", ")} lets a more accurate model be used.`));
  }

  const det = el("details");
  det.appendChild(el("summary", null, "How this was worked out"));
  const dl = el("dl");
  const add = (k, v) => { if (v) { dl.appendChild(el("dt", null, k)); dl.appendChild(el("dd", null, v)); } };
  const ds = config && config.datasets.find(d => d.id === r.dataset);
  add("Learned from", ds ? `${ds.label}, ${ds.rows.toLocaleString()} people` : r.dataset);
  add("Model", `${MODEL_NAMES[r.model_used] || r.model_used} · ${r.model_version || ""}`);
  add("Inputs it uses", (r.features_used || []).map(fieldName).join(", "));
  const m = r.tier_test_metrics;
  if (m) add("Measured", `${Math.round(m.accuracy * 100)}% accuracy on unseen people · F1 ${(m.f1).toFixed(2)}${m.cv_f1_mean ? ` (CV ${m.cv_f1_mean.toFixed(2)})` : ""}`);
  if (r.insulin_imputed) add("Note", "Insulin wasn't given, so the training median was used.");
  det.appendChild(dl);
  card.appendChild(det);
  card.appendChild(el("p", "fine", r.disclaimer));
  return card;
}

function renderReply(r) {
  if (r.status === "result" && r.probability !== null) {
    log.appendChild(readingCard(r));
    scrollDown();
    return;
  }
  const b = bubble("bot" + (r.status === "error" ? " error" : ""));
  if (r.status === "need_info") r.missing_fields.forEach(f => b.appendChild(el("span", "asks", `Needs ${fieldName(f)}`)));
  if (r.status === "invalid_input") r.invalid_fields.forEach(f => b.appendChild(el("span", "asks bad", `Check ${fieldName(f)}`)));
  r.explanation.split(/\n+/).filter(Boolean).forEach(p => b.appendChild(el("p", null, p)));
}

function settings() {
  if (!config) return null;
  const model = $("optModel").querySelector('[aria-checked="true"]');
  return {
    dataset: $("optDataset").value || null,
    ml_model: model ? model.dataset.value : "mlp",
    llm: $("optLlm").value || null,
  };
}

async function sendMessage(text, values) {
  text = text.trim();
  if (!text) return;
  showView("chat");
  bubble("you", text);
  input.value = "";
  autosize();
  send.disabled = true;
  const typing = el("div", "bubble bot typing");
  typing.setAttribute("aria-label", "Thinking");
  typing.append(el("i"), el("i"), el("i"));
  log.appendChild(typing);
  scrollDown();
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text, visitor_id: visitorId, values: values || null, settings: settings() }),
    });
    const data = await resp.json().catch(() => ({}));
    typing.remove();
    if (!resp.ok) {
      bubble("bot error", typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.");
      return;
    }
    sessionId = data.session_id;
    try { sessionStorage.setItem("sessionId", sessionId); } catch (_) {}
    renderReply(data.reply);
  } catch (_) {
    typing.remove();
    bubble("bot error", "Couldn't reach the server. Check your connection and try again.");
  } finally {
    send.disabled = false;
    input.focus({ preventScroll: true });
  }
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

// ---- view tabs (phone) -------------------------------------------------
function showView(view) {
  document.querySelectorAll("[data-tab]").forEach(t => t.setAttribute("aria-selected", String(t.dataset.tab === view)));
  $("panel").classList.toggle("collapsed", view !== "form");
  document.querySelector(".page").classList.toggle("form-view", view === "form");
}
document.querySelectorAll("[data-tab]").forEach(t => t.addEventListener("click", () => { showView(t.dataset.tab); window.scrollTo({ top: 0 }); }));

// ---- form + settings ---------------------------------------------------
function buildFields() {
  const main = $("fields"), more = $("moreFieldsBox");
  main.replaceChildren(); more.replaceChildren();
  config.fields.forEach(f => {
    const label = el("label");
    const name = el("span", null, fieldName(f.id));
    if (f.required) { name.appendChild(document.createTextNode(" ")); name.appendChild(el("b", null, "*")); }
    label.appendChild(name);
    const inp = el("input");
    inp.type = "number";
    inp.name = f.id;
    inp.min = f.range[0];
    inp.max = f.range[1];
    inp.step = f.id === "pedigree" ? "0.01" : "any";
    inp.required = f.required;
    inp.inputMode = "decimal";
    inp.placeholder = f.id === "insulin" ? "optional" : `${f.range[0]}–${f.range[1]}`;
    label.appendChild(inp);
    (f.required || f.id === "insulin" ? main : more).appendChild(label);
  });
}

function showDatasetInfo() {
  const d = config.datasets.find(x => x.id === $("optDataset").value);
  if (!d) { $("datasetInfo").textContent = ""; return; }
  const best = d.tiers.map(t => t.test_metrics.mlp).filter(Boolean).sort((a, b) => b.accuracy - a.accuracy)[0];
  $("datasetInfo").textContent = `${d.rows.toLocaleString()} people` + (best ? ` · ${Math.round(best.accuracy * 100)}% accurate on unseen data` : "") + (d.id === "kaggle" ? " · likely synthetic data" : "");
}

function remember(id, key) {
  const saved = load(key);
  if (saved !== null && [...$(id).options].some(o => o.value === saved)) $(id).value = saved;
  $(id).addEventListener("change", () => { store(key, $(id).value); if (id === "optDataset") showDatasetInfo(); });
}

async function loadConfig() {
  try { config = await (await fetch("/config")).json(); } catch (_) { return; }
  buildFields();
  config.datasets.forEach(d => $("optDataset").appendChild(new Option(DATASET_NAMES[d.id] || d.label, d.id)));
  $("optDataset").value = config.default_dataset;
  $("optLlm").appendChild(new Option("Server default", ""));
  config.llms.forEach(l => $("optLlm").appendChild(new Option(l.provider === "rules" ? "Offline (no AI)" : l.label, l.id)));
  remember("optDataset", "set.dataset");
  remember("optLlm", "set.llm");
  const savedModel = load("set.model");
  if (savedModel) setModel(savedModel);
  showDatasetInfo();
}

function setModel(value) {
  $("optModel").querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.value === value)));
}
$("optModel").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  setModel(b.dataset.value);
  store("set.model", b.dataset.value);
});

$("valuesForm").addEventListener("submit", e => {
  e.preventDefault();
  const values = {}, parts = [];
  new FormData(e.target).forEach((v, k) => {
    if (v === "") return;
    values[k] = parseFloat(v);
    parts.push(`${fieldName(k)} ${v}`);
  });
  sendMessage(parts.join(", "), values);
});
$("clearValues").addEventListener("click", () => $("valuesForm").reset());

form.addEventListener("submit", e => { e.preventDefault(); sendMessage(input.value); });
input.addEventListener("input", autosize);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
document.querySelectorAll("#chips button").forEach(c => c.addEventListener("click", () => sendMessage(c.textContent)));
$("newChat").addEventListener("click", () => {
  sessionId = null;
  try { sessionStorage.removeItem("sessionId"); } catch (_) {}
  log.querySelectorAll(".bubble, .reading").forEach((m, i) => { if (i > 0) m.remove(); });
  showView("chat");
});
loadConfig();
