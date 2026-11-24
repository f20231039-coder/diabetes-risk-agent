const $ = id => document.getElementById(id);
let token = "";
try { token = sessionStorage.getItem("adminToken") || ""; } catch (_) {}
$("token").value = token;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function api(path, opts = {}) {
  const resp = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), "X-Admin-Token": token } });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${resp.status}`);
  return data;
}

function setStatus(text, isErr) {
  const s = $("uploadStatus");
  s.textContent = text;
  s.className = "status" + (isErr ? " err" : "");
}

const pct = x => (x === null || x === undefined ? "–" : (x * 100).toFixed(1) + "%");
const num = x => (x === null || x === undefined ? "–" : x.toFixed(3));

function renderVersions(data) {
  const box = $("versions");
  box.replaceChildren();
  const ds = $("dataset");
  if (!ds.options.length) {
    Object.entries(data.datasets).forEach(([id, d]) => ds.appendChild(new Option(`${id} – ${d.label}`, id)));
  }
  [...data.versions].reverse().forEach(v => {
    const card = el("div", "ver" + (v.active ? " active" : ""));
    const head = el("div", "ver-head");
    const title = el("div");
    title.appendChild(el("strong", null, `${v.version} · ${v.dataset}`));
    if (v.active) title.appendChild(el("span", "badge", "active"));
    title.appendChild(el("span", "small", `  ${v.rows} rows · ${v.created}`));
    head.appendChild(title);
    if (!v.active) {
      const btn = el("button", "ghost", "Activate (roll back to this)");
      btn.type = "button";
      btn.addEventListener("click", async () => {
        if (!confirm(`Make ${v.version} the live ${v.dataset} model?`)) return;
        try { await api(`/admin/models/${v.version}/activate`, { method: "POST" }); load(); }
        catch (e) { alert(e.message); }
      });
      head.appendChild(btn);
    }
    card.appendChild(head);

    const wrap = el("div", "table-wrap");
    const table = el("table");
    const hr = el("tr");
    ["Tier", "Model", "Accuracy", "F1 (test)", "F1 (5-fold CV)", "ROC-AUC"].forEach(h => hr.appendChild(el("th", null, h)));
    table.appendChild(hr);
    Object.entries(v.tiers).forEach(([tier, models]) => {
      Object.entries(models).forEach(([model, m]) => {
        const tr = el("tr");
        const cv = m.cv_f1_mean !== undefined ? `${num(m.cv_f1_mean)} ± ${num(m.cv_f1_std)}` : "–";
        [tier, model, pct(m.accuracy), num(m.f1), cv, num(m.roc_auc)].forEach(c => tr.appendChild(el("td", null, c)));
        table.appendChild(tr);
      });
    });
    wrap.appendChild(table);
    card.appendChild(wrap);
    if (v.promotion) {
      const p = v.promotion;
      const ci = p.f1_difference_95ci ? ` · F1 difference 95% CI [${p.f1_difference_95ci.join(", ")}]${p.difference_is_noise ? " (within noise)" : ""}` : "";
      card.appendChild(el("p", "decision",
        `Promotion: ${p.promoted ? "promoted" : "not promoted"} · candidate F1 ${num(p.candidate_standard_mlp_f1)} vs ${p.active_version || "none"} ${num(p.active_standard_mlp_f1_same_test_rows)}${ci}`));
    }
    box.appendChild(card);
  });
}

async function load() {
  if (!token) return;
  try { renderVersions(await api("/admin/models")); }
  catch (e) { $("versions").replaceChildren(el("p", "status err", e.message)); }
}

async function pollJob(id) {
  for (;;) {
    const job = await api(`/admin/jobs/${id}`);
    if (job.status === "running") {
      setStatus(`Retraining… ${Math.round(Date.now() / 1000 - job.started)} s`);
      await new Promise(r => setTimeout(r, 2000));
      continue;
    }
    if (job.status === "failed") return setStatus("Training failed: " + job.error, true);
    const r = job.result;
    setStatus(`Trained ${r.version}. ${r.promoted ? "Now live." : "Not promoted (F1 dropped); the previous version stays live."}`);
    return load();
  }
}

$("saveToken").addEventListener("click", () => {
  token = $("token").value.trim();
  try { sessionStorage.setItem("adminToken", token); } catch (_) {}
  load();
});
$("refresh").addEventListener("click", load);
$("upload").addEventListener("click", async () => {
  const f = $("file").files[0];
  if (!f) return setStatus("Choose a CSV file first.", true);
  $("upload").disabled = true;
  setStatus("Uploading…");
  try {
    const q = new URLSearchParams({ dataset: $("dataset").value, force: $("force").checked });
    const out = await api(`/admin/dataset?${q}`, { method: "POST", headers: { "Content-Type": "text/csv" }, body: f });
    const m = out.merge;
    setStatus(`Merged: ${m.rows_uploaded} uploaded, ${m.duplicates_skipped} duplicates skipped, ${m.rows_before} → ${m.rows_after} rows.`);
    if (out.job_id) await pollJob(out.job_id);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("upload").disabled = false;
  }
});
load();
