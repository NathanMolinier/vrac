"""Keyboard-driven review of the QC previews made by ``qc_prepare.sh``.

Opens a small local web page (Python stdlib only — works over an SSH tunnel too)
showing ``<qc_dir>/metrics/previews/<stem>.png`` (ground truth | prediction) one
sample at a time. For each sample choose:

    P / 1   replace the BIDS ground truth by the PREDICTION
    G / 2   keep the GROUND TRUTH
    D / 3   DROP the sample from the discslabel JSON(s) (file stays in BIDS)
    U       clear the decision
    C       add / edit a comment
    ← / →   previous / next sample        N   next undecided sample

Each decision is written immediately to ``<qc_dir>/qc_decisions.json`` (resume any
time). Nothing is changed in BIDS or in the JSONs until you run ``qc_apply.py``.

Usage:
    python qc_review.py <qc_dir> [--port 8765] [--order f1|name] [--no-browser]

Remote data, two options:
  * run it on the server, then ``ssh -L 8765:localhost:8765 <server>`` and open
    http://localhost:8765 locally;
  * or review offline: ``qc_bundle.py`` on the server, copy + extract the bundle
    locally and run ``qc_review.py <bundle_dir>`` (no NIfTI needed — prediction
    availability comes from ``review_manifest.json``). Send the resulting
    ``qc_decisions.json`` back and use ``qc_apply.py --decisions``.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

DECISIONS = ("prediction", "keep", "drop")


def image_f1(p: dict, tolerance_mm: float) -> float:
    """Same as ``discslabel.evaluate._image_f1``."""
    denom = p["n_gt"] + p["n_pred"]
    if denom == 0:
        return 1.0
    hits = sum(1 for m in p["matched"] if m["distance_mm"] <= tolerance_mm)
    return 2.0 * hits / denom


def load_samples(qc_dir: str, tolerance_mm: float, order: str) -> list:
    with open(os.path.join(qc_dir, "qc_index.json")) as f:
        index = json.load(f)
    metrics_path = os.path.join(qc_dir, "metrics", "metrics.json")
    per_image = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            per_image = {p["image"]: p for p in json.load(f)["per_image"]}

    # Offline bundle: predictions/ is not copied, the manifest says which exist.
    manifest = {}
    manifest_path = os.path.join(qc_dir, "review_manifest.json")
    pred_dir = os.path.join(qc_dir, "predictions")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f).get("predictions", {})
    elif not os.path.isdir(pred_dir):
        print(f"[review] WARN: neither {pred_dir} nor {manifest_path} — no prediction can be chosen")

    samples = []
    for stem, s in index["samples"].items():
        if s.get("missing"):
            continue
        p = per_image.get(stem)
        far = []
        if p:
            far = [f"{m['label']} ({m['distance_mm']:.0f}mm)" for m in p["matched"]
                   if m["distance_mm"] > tolerance_mm]
        samples.append({
            "stem": stem,
            "image": s["image"],
            "label": s["label"],
            "uses": [f"{os.path.basename(u['json'])}:{u['split']}" for u in s["uses"]],
            "has_pred": stem in manifest or os.path.isfile(os.path.join(pred_dir, f"{stem}_discs.nii.gz")),
            "pred_sha256": manifest.get(stem),
            "has_preview": os.path.isfile(os.path.join(qc_dir, "metrics", "previews", f"{stem}.png")),
            "f1": None if p is None else round(image_f1(p, tolerance_mm), 3),
            "n_gt": None if p is None else p["n_gt"],
            "n_pred": None if p is None else p["n_pred"],
            "missed": [] if p is None else p["false_negative_labels"],
            "extra": [] if p is None else p["false_positive_labels"],
            "far": far,
        })
    if order == "f1":
        samples.sort(key=lambda s: (s["f1"] if s["f1"] is not None else -1.0, s["stem"]))
    else:
        samples.sort(key=lambda s: s["stem"])
    return samples


class Store:
    """``qc_decisions.json``: ``{stem: {"decision": ..., "comment": ..., "time": ...}}``."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if os.path.isfile(path):
            with open(path) as f:
                self.data = json.load(f)

    def update(self, stem: str, decision=None, comment=None, clear=False, pred_sha256=None) -> dict:
        with self.lock:
            rec = dict(self.data.get(stem, {}))
            if clear:
                rec.pop("decision", None)
            if decision is not None:
                rec["decision"] = decision
            if "decision" in rec or clear:
                # Prediction reviewed: remember which file, qc_apply checks it is unchanged.
                if rec.get("decision") == "prediction" and pred_sha256:
                    rec["pred_sha256"] = pred_sha256
                elif rec.get("decision") != "prediction":
                    rec.pop("pred_sha256", None)
            if comment is not None:
                rec["comment"] = comment
                if not comment:
                    rec.pop("comment")
            rec["time"] = datetime.now().isoformat(timespec="seconds")
            if "decision" in rec or "comment" in rec:
                self.data[stem] = rec
            else:
                self.data.pop(stem, None)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
            return self.data.get(stem, {})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Discs QC</title>
<style>
  :root { --bg:#111; --panel:#1c1c1c; --fg:#e8e8e8; --dim:#999; --line:#333;
          --pred:#3b82f6; --keep:#22c55e; --drop:#ef4444; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system, system-ui, sans-serif;
         display:flex; height:100vh; }
  #main { flex:1; display:flex; align-items:center; justify-content:center; min-width:0; padding:8px; }
  #main img { max-width:100%; max-height:100%; object-fit:contain; }
  #nopreview { color:var(--dim); font-size:18px; }
  #side { width:340px; background:var(--panel); border-left:1px solid var(--line); padding:14px;
          overflow-y:auto; display:flex; flex-direction:column; gap:12px; }
  h1 { font-size:13px; margin:0; word-break:break-all; }
  .muted { color:var(--dim); font-size:12px; word-break:break-all; }
  .row { display:flex; gap:6px; }
  button { flex:1; padding:10px 4px; border:2px solid var(--line); border-radius:6px; background:#262626;
           color:var(--fg); font-weight:600; cursor:pointer; }
  button:disabled { opacity:.35; cursor:not-allowed; }
  button.prediction.on { background:var(--pred); border-color:var(--pred); }
  button.keep.on { background:var(--keep); border-color:var(--keep); color:#000; }
  button.drop.on { background:var(--drop); border-color:var(--drop); }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-weight:600; font-size:12px; }
  .b-prediction { background:var(--pred); } .b-keep { background:var(--keep); color:#000; }
  .b-drop { background:var(--drop); } .b-none { background:#444; }
  #bar { height:6px; background:#333; border-radius:3px; overflow:hidden; }
  #bar > div { height:100%; background:var(--keep); }
  table { border-collapse:collapse; width:100%; font-size:12px; }
  td { padding:2px 4px; vertical-align:top; } td:first-child { color:var(--dim); white-space:nowrap; }
  select, input { background:#262626; color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:4px; }
  kbd { background:#333; border-radius:3px; padding:0 4px; font-size:11px; }
  #comment { font-style:italic; color:#fbbf24; word-break:break-word; }
</style></head>
<body>
<div id="main"><img id="img" alt=""><div id="nopreview" hidden>no preview (prediction missing?)</div></div>
<div id="side">
  <div class="row" style="align-items:center">
    <span id="pos" style="flex:1"></span>
    <select id="filter">
      <option value="all">all</option><option value="undecided">undecided</option>
      <option value="errors">F1 &lt; 1</option><option value="prediction">→ prediction</option>
      <option value="keep">→ keep GT</option><option value="drop">→ drop</option>
    </select>
  </div>
  <div id="bar"><div></div></div>
  <div class="muted" id="counts"></div>
  <h1 id="stem"></h1>
  <div><span id="decision" class="badge"></span></div>
  <div class="row">
    <button class="prediction" data-d="prediction">Prediction <kbd>P</kbd></button>
    <button class="keep" data-d="keep">Keep GT <kbd>G</kbd></button>
    <button class="drop" data-d="drop">Drop <kbd>D</kbd></button>
  </div>
  <div id="comment"></div>
  <table id="info"></table>
  <div class="muted">
    <kbd>P</kbd>/<kbd>1</kbd> prediction · <kbd>G</kbd>/<kbd>2</kbd> keep GT · <kbd>D</kbd>/<kbd>3</kbd> drop<br>
    <kbd>←</kbd>/<kbd>→</kbd> prev/next · <kbd>N</kbd> next undecided · <kbd>U</kbd> clear · <kbd>C</kbd> comment<br>
    Decisions are saved instantly; nothing is applied until <code>qc_apply.py</code>.
  </div>
</div>
<script>
let samples = [], decisions = {}, view = [], pos = 0;
const $ = id => document.getElementById(id);
const isErr = s => s.f1 === null || s.f1 < 1;

function rebuild(keepStem) {
  const f = $('filter').value;
  view = samples.filter(s => {
    const d = (decisions[s.stem] || {}).decision;
    if (f === 'undecided') return !d || s.stem === keepStem;
    if (f === 'errors') return isErr(s);
    if (['prediction','keep','drop'].includes(f)) return d === f || s.stem === keepStem;
    return true;
  });
  const i = view.findIndex(s => s.stem === keepStem);
  pos = i >= 0 ? i : Math.min(pos, Math.max(0, view.length - 1));
  render();
}

function render() {
  const n = samples.length, done = samples.filter(s => (decisions[s.stem]||{}).decision).length;
  const c = {prediction:0, keep:0, drop:0};
  Object.values(decisions).forEach(r => { if (r.decision) c[r.decision]++; });
  $('bar').firstElementChild.style.width = (100 * done / Math.max(1, n)) + '%';
  $('counts').textContent = `${done}/${n} decided · prediction ${c.prediction} · keep ${c.keep} · drop ${c.drop}`;
  if (!view.length) { $('pos').textContent = '0 / 0'; $('stem').textContent = 'nothing to show';
    $('img').hidden = true; $('info').innerHTML = ''; return; }
  const s = view[pos], rec = decisions[s.stem] || {}, d = rec.decision;
  $('pos').textContent = `${pos + 1} / ${view.length}`;
  $('stem').textContent = s.stem;
  const b = $('decision'); b.className = 'badge b-' + (d || 'none');
  b.textContent = {prediction:'→ replace GT by prediction', keep:'→ keep ground truth', drop:'→ drop from JSON'}[d] || 'undecided';
  document.querySelectorAll('button[data-d]').forEach(x => {
    x.classList.toggle('on', x.dataset.d === d);
    if (x.dataset.d === 'prediction') x.disabled = !s.has_pred;
  });
  $('comment').textContent = rec.comment ? '“' + rec.comment + '”' : '';
  $('img').hidden = !s.has_preview; $('nopreview').hidden = s.has_preview;
  if (s.has_preview) $('img').src = 'preview/' + encodeURIComponent(s.stem) + '.png';
  const rows = [
    ['F1', s.f1 === null ? '—' : s.f1.toFixed(2)],
    ['discs', `GT ${s.n_gt ?? '—'} · pred ${s.n_pred ?? '—'}`],
    ['missed', s.missed.join(', ') || '—'], ['extra', s.extra.join(', ') || '—'],
    ['too far', s.far.join(', ') || '—'], ['used in', s.uses.join('<br>')],
    ['image', s.image], ['label', s.label],
  ];
  $('info').innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  [1, 2].forEach(k => { const t = view[pos + k];   // prefetch
    if (t && t.has_preview) new Image().src = 'preview/' + encodeURIComponent(t.stem) + '.png'; });
}

async function post(body) {
  const r = await fetch('api/decide', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (!r.ok) { alert('save failed: ' + await r.text()); return null; }
  return r.json();
}

async function decide(d) {
  const s = view[pos]; if (!s) return;
  if (d === 'prediction' && !s.has_pred) return;
  const rec = await post({stem: s.stem, decision: d}); if (!rec) return;
  decisions[s.stem] = rec;
  if (pos < view.length - 1) pos++;
  rebuild(view[pos].stem);
}
async function clearDecision() {
  const s = view[pos]; if (!s) return;
  const rec = await post({stem: s.stem, clear: true}); if (!rec) return;
  if (Object.keys(rec).length) decisions[s.stem] = rec; else delete decisions[s.stem];
  rebuild(s.stem);
}
async function comment() {
  const s = view[pos]; if (!s) return;
  const txt = prompt('Comment for ' + s.stem, (decisions[s.stem] || {}).comment || '');
  if (txt === null) return;
  const rec = await post({stem: s.stem, comment: txt}); if (!rec) return;
  if (Object.keys(rec).length) decisions[s.stem] = rec; else delete decisions[s.stem];
  render();
}
function nextUndecided() {
  for (let k = 1; k <= view.length; k++) {
    const i = (pos + k) % view.length;
    if (!(decisions[view[i].stem] || {}).decision) { pos = i; render(); return; }
  }
}

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === 'p' || k === '1') decide('prediction');
  else if (k === 'g' || k === 'k' || k === '2') decide('keep');
  else if (k === 'd' || k === 'x' || k === '3') decide('drop');
  else if (k === 'u') clearDecision();
  else if (k === 'c') comment();
  else if (k === 'n') nextUndecided();
  else if (k === 'arrowright' || k === 'l') { if (pos < view.length - 1) { pos++; render(); } }
  else if (k === 'arrowleft' || k === 'h') { if (pos > 0) { pos--; render(); } }
  else return;
  e.preventDefault();
});
document.querySelectorAll('button[data-d]').forEach(b => b.onclick = () => decide(b.dataset.d));
$('filter').onchange = () => { const s = view[pos]; rebuild(s && s.stem); $('filter').blur(); };

fetch('api/state').then(r => r.json()).then(st => {
  samples = st.samples; decisions = st.decisions;
  rebuild(null);
  const i = view.findIndex(s => !(decisions[s.stem] || {}).decision);   // resume
  if (i >= 0) { pos = i; render(); }
});
</script>
</body></html>
"""


def make_handler(qc_dir: str, samples: list, store: Store):
    previews = os.path.join(qc_dir, "metrics", "previews")
    stems = {s["stem"]: s for s in samples}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype != "image/png" else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json({"samples": samples, "decisions": store.data})
            elif path.startswith("/preview/") and path.endswith(".png"):
                stem = path[len("/preview/"):-len(".png")]
                if stem not in stems:  # also blocks path traversal
                    return self._send(404, b"unknown stem", "text/plain")
                fp = os.path.join(previews, stem + ".png")
                if not os.path.isfile(fp):
                    return self._send(404, b"no preview", "text/plain")
                with open(fp, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/decide":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                stem = body["stem"]
                decision = body.get("decision")
                if stem not in stems or (decision is not None and decision not in DECISIONS):
                    raise ValueError("bad stem or decision")
            except Exception as exc:  # noqa: BLE001
                return self._send(400, str(exc).encode(), "text/plain")
            rec = store.update(stem, decision=decision, comment=body.get("comment"),
                               clear=bool(body.get("clear")), pred_sha256=stems[stem].get("pred_sha256"))
            self._json(rec)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qc_dir", help="Folder produced by qc_prepare.sh.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--order", choices=("f1", "name"), default="f1",
                        help="Sample order: worst F1 first (default) or by name.")
    parser.add_argument("--tolerance-mm", type=float, default=None,
                        help="Distance counted as an error (default: config val_tolerance_mm, else 10).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()

    qc_dir = os.path.abspath(args.qc_dir)
    tol = args.tolerance_mm
    if tol is None:
        tol = 10.0
        cfg_path = os.path.join(qc_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                tol = float(json.load(f).get("val_tolerance_mm", 10.0))

    samples = load_samples(qc_dir, tol, args.order)
    store = Store(os.path.join(qc_dir, "qc_decisions.json"))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(qc_dir, samples, store))
    url = f"http://localhost:{args.port}/"
    n_done = sum(1 for s in samples if store.data.get(s["stem"], {}).get("decision"))
    print(f"[review] {len(samples)} samples ({n_done} already decided) — {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"\n[review] decisions saved in {store.path}")


if __name__ == "__main__":
    main()
