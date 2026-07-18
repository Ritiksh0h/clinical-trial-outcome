"""CTO-Predict serving app (v1) — FastAPI.

    uv run uvicorn cto.serving.app:app --reload

GET /         minimal HTML form (clickable demo)
GET /health   {status, models_loaded}
POST /predict raw trial fields -> calibrated completion probability
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from cto.serving.predict import available_phases, predict_completion
from cto.serving.schema import HealthResponse, PredictionResponse, TrialInput

app = FastAPI(
    title="CTO-Predict",
    version="1.0.0",
    description="Predicts clinical trial COMPLETION (vs termination/withdrawal) — not drug efficacy.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=available_phases())


@app.post("/predict", response_model=PredictionResponse)
def predict(trial: TrialInput) -> dict:
    try:
        return predict_completion(trial)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTO-Predict — trial completion</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
  h1{margin-bottom:.2rem} .sub{color:#555;margin-top:0}
  .banner{background:#fff8e1;border:1px solid #e0c56e;border-radius:8px;padding:.7rem .9rem;font-size:.9rem;margin:1rem 0}
  label{display:block;margin:.6rem 0 .15rem;font-weight:600;font-size:.9rem}
  input,select,textarea{width:100%;padding:.45rem;border:1px solid #ccc;border-radius:6px;font:inherit;box-sizing:border-box}
  .row{display:flex;gap:1rem}.row>div{flex:1}
  button{margin-top:1.1rem;padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2f6fed;color:#fff;font-weight:600;cursor:pointer}
  #out{margin-top:1.3rem;padding:1rem;border-radius:8px;display:none}
  .tier{font-size:1.5rem;font-weight:700;margin-bottom:.15rem} .rawp{font-size:.95rem;color:#333;margin-top:.6rem} .note{font-size:.82rem;color:#444;margin-top:.4rem}
</style>
</head>
<body>
<h1>CTO-Predict</h1>
<p class="sub">Phase-relative completion-likelihood tier for a clinical trial, from registration-time details.</p>
<div class="banner"><b>What this is:</b> it predicts whether a trial will <b>COMPLETE</b> vs be terminated/withdrawn —
<b>not</b> whether the drug works (not efficacy). The probability is calibrated.
<b>v1 caveat:</b> sponsor / therapeutic-area history features use population-median defaults (real lookup comes in v2).<br>
<b>How to read it:</b> the result is a <b>tier</b> ranking this trial against the model's evaluation set for its phase — a relative ranking, <b>not</b> an absolute real-world probability.</div>

<form id="f">
  <div class="row">
    <div><label>Phase</label><select name="phase"><option value="3">III</option><option value="2">II</option><option value="1">I</option></select></div>
    <div><label>Lead sponsor class</label><select name="sponsor_class">
      <option value="INDUSTRY">Industry</option><option value="NIH">NIH</option><option value="OTHER_GOV">Other gov</option>
      <option value="NETWORK">Network</option><option value="OTHER">Other</option></select></div>
  </div>
  <label>Eligibility criteria (free text)</label>
  <textarea name="eligibility_criteria" rows="4" placeholder="Inclusion Criteria: adults 18+ ... Exclusion Criteria: ..."></textarea>
  <div class="row">
    <div><label>Allocation</label><select name="allocation"><option value="RANDOMIZED">Randomized</option><option value="NON_RANDOMIZED">Non-randomized</option></select></div>
    <div><label>Masking</label><select name="masking"><option value="NONE">None</option><option value="SINGLE">Single</option><option value="DOUBLE">Double</option><option value="TRIPLE">Triple</option><option value="QUADRUPLE">Quadruple</option></select></div>
  </div>
  <div class="row">
    <div><label>Primary purpose</label><select name="primary_purpose"><option value="TREATMENT">Treatment</option><option value="PREVENTION">Prevention</option><option value="DIAGNOSTIC">Diagnostic</option></select></div>
    <div><label>Intervention types (comma-separated)</label><input name="intervention_types" placeholder="DRUG, BIOLOGICAL"></div>
  </div>
  <div class="row">
    <div><label>Enrollment (planned)</label><input type="number" name="enrollment" min="0" placeholder="200"></div>
    <div><label>Number of arms</label><input type="number" name="number_of_arms" min="1" placeholder="2"></div>
    <div><label>Registration year</label><input type="number" name="registration_year" min="1990" max="2100" placeholder="2021"></div>
  </div>
  <button type="submit">Predict completion probability</button>
</form>

<div id="out"></div>

<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target), body = {};
  for (const [k, v] of fd.entries()) { if (v === '') continue; body[k] = v; }
  body.phase = parseInt(body.phase);
  ['enrollment','number_of_arms','registration_year'].forEach(k => { if (k in body) body[k] = parseInt(body[k]); });
  if (body.intervention_types) body.intervention_types = body.intervention_types.split(',').map(s => s.trim()).filter(Boolean);
  const out = document.getElementById('out');
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { out.style.display='block'; out.style.background='#fdecea'; out.innerHTML = 'Error: ' + (j.detail || JSON.stringify(j)); return; }
    out.style.display='block'; out.style.background='#eef4ff';
    const raw = (j.raw_calibrated_probability * 100).toFixed(1);
    out.innerHTML =
      '<div class="tier">' + j.completion_tier + '</div>' +
      '<div>Phase ' + j.phase + ' — ' + j.tier_context + '.</div>' +
      '<div class="rawp">Raw calibrated probability: <b>' + raw + '%</b> (secondary)</div>' +
      '<div class="note">' + j.probability_caveat + '</div>';
  } catch (err) { out.style.display='block'; out.style.background='#fdecea'; out.textContent = 'Request failed: ' + err; }
});
</script>
</body></html>"""
