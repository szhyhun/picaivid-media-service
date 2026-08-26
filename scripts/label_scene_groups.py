#!/usr/bin/env python
"""Generate a self-contained HTML labeling sheet for scene-graph ground truth.

Usage:
    python scripts/label_scene_groups.py <images_folder> [--out <dir>] [--slug <listing-slug>]

Reads every image in the folder, orders photos by the numeric filename prefix
(matching job_photos.position, 0-based), embeds medium-size thumbnails, and
writes one HTML file. The maintainer opens it in a browser, groups photos into
room instances, marks duplicates / preferred cinematic pairs / must-not-group
negatives and open-plan connections, then exports ground_truth JSON matching
tests/fixtures/scene_graph schema (Scene-Graph V2 plan, Stage 0).

The page needs no server and autosaves progress to localStorage.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_LONG_SIDE = 640
THUMB_QUALITY = 70


def collect_photos(folder: Path) -> list[dict]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise SystemExit(f"No images found in {folder}")

    def prefix(path: Path) -> tuple[int, str]:
        match = re.match(r"^(\d+)", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)

    files.sort(key=prefix)
    photos = []
    for index, path in enumerate(files):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=THUMB_QUALITY)
        photos.append({
            "position": index,
            "filename": path.name,
            "data": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        })
        print(f"  {index:3d}  {path.name}", file=sys.stderr)
    return photos


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Room labeling — __SLUG__</title>
<style>
  :root { --bg:#f5f4f1; --panel:#ffffff; --ink:#26241f; --muted:#8a857a; --line:#e3e0d8; --accent:#3d6b4f; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:20; background:var(--panel); border-bottom:1px solid var(--line);
           padding:10px 16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0 12px 0 0; }
  .modes { display:flex; gap:6px; }
  .modes button { border:1px solid var(--line); background:var(--bg); border-radius:6px; padding:6px 10px; cursor:pointer; font-size:13px; }
  .modes button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  #status { color:var(--muted); font-size:12px; margin-left:auto; }
  main { display:grid; grid-template-columns: 1fr 320px; gap:0; }
  #grid { padding:14px; display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:10px; align-content:start; }
  .tile { position:relative; border-radius:8px; overflow:hidden; background:#000; cursor:pointer;
          outline:3px solid transparent; outline-offset:-3px; }
  .tile img { width:100%; height:130px; object-fit:cover; display:block; }
  .tile .pos { position:absolute; top:5px; left:5px; background:rgba(0,0,0,.65); color:#fff; font-size:11px;
               padding:1px 6px; border-radius:4px; font-variant-numeric:tabular-nums; }
  .tile .zoom { position:absolute; top:5px; right:5px; background:rgba(0,0,0,.55); color:#fff; border:none;
                border-radius:4px; cursor:zoom-in; font-size:12px; padding:1px 6px; }
  .tile .badge { position:absolute; left:0; right:0; bottom:0; font-size:11px; color:#fff; padding:2px 6px;
                 background:rgba(0,0,0,.45); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tile.pairflash { outline-color:#e0a400 !important; }
  aside { border-left:1px solid var(--line); background:var(--panel); padding:14px; position:sticky; top:49px;
          height:calc(100vh - 49px); overflow-y:auto; }
  aside h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:18px 0 6px; }
  aside h2:first-child { margin-top:0; }
  .group { display:flex; align-items:center; gap:8px; padding:5px 6px; border-radius:6px; cursor:pointer; }
  .group.active { background:var(--bg); box-shadow:inset 0 0 0 1px var(--line); }
  .swatch { width:14px; height:14px; border-radius:4px; flex:none; }
  .group input { border:none; background:transparent; font:inherit; width:100%; }
  .group .count { color:var(--muted); font-size:12px; }
  .group .del, .pair .del { border:none; background:none; color:var(--muted); cursor:pointer; font-size:14px; }
  #newGroup { margin-top:6px; width:100%; border:1px dashed var(--line); background:none; border-radius:6px;
              padding:6px; cursor:pointer; color:var(--muted); }
  .pair { display:flex; justify-content:space-between; align-items:center; font-size:13px; padding:2px 6px; }
  .oplist label { display:flex; gap:6px; align-items:center; font-size:13px; padding:2px 6px; }
  .opset { font-size:13px; padding:2px 6px; display:flex; justify-content:space-between; }
  #notes { width:100%; min-height:56px; border:1px solid var(--line); border-radius:6px; font:inherit; padding:6px; }
  .actions { display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; }
  .actions button, .actions label { border:1px solid var(--line); background:var(--bg); border-radius:6px;
                                    padding:7px 10px; cursor:pointer; font-size:13px; }
  .actions .primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  #lightbox { position:fixed; inset:0; background:rgba(0,0,0,.88); display:none; z-index:50;
              align-items:center; justify-content:center; flex-direction:column; gap:10px; }
  #lightbox img { max-width:92vw; max-height:84vh; }
  #lightbox .cap { color:#ddd; font-size:13px; }
  kbd { background:var(--bg); border:1px solid var(--line); border-radius:3px; padding:0 4px; font-size:11px; }
  .hint { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>__SLUG__</h1>
  <div class="modes" id="modes">
    <button data-mode="assign" class="on">Assign rooms</button>
    <button data-mode="dup">Duplicates</button>
    <button data-mode="pref">Preferred pairs</button>
    <button data-mode="neg">Must-not-group</button>
  </div>
  <span class="hint">Assign: click photos to put them in the active room · Pair modes: click two photos · <kbd>N</kbd> new room · <kbd>Esc</kbd> close zoom</span>
  <span id="status"></span>
</header>
<main>
  <div id="grid"></div>
  <aside>
    <h2>Room instances</h2>
    <div id="groups"></div>
    <button id="newGroup">+ New room (N)</button>
    <h2>Open-plan connections</h2>
    <div class="hint">Tick rooms that form one connected open space, then link.</div>
    <div class="oplist" id="oplist"></div>
    <button id="linkOp">Link ticked rooms</button>
    <div id="opsets"></div>
    <h2>Duplicates</h2><div id="dups"></div>
    <h2>Preferred cinematic pairs</h2><div id="prefs"></div>
    <h2>Must not group</h2><div id="negs"></div>
    <h2>Notes</h2>
    <textarea id="notes" placeholder="anything unusual…"></textarea>
    <div class="actions">
      <button class="primary" id="export">Export ground_truth JSON</button>
      <label>Import JSON<input type="file" id="import" accept=".json" style="display:none"></label>
      <button id="reset">Reset</button>
    </div>
    <p class="hint">Progress autosaves in this browser. Export when done and send the JSON file back.</p>
  </aside>
</main>
<div id="lightbox"><img alt=""><div class="cap"></div></div>
<script>
const PHOTOS = __PHOTOS_JSON__;
const SLUG = "__SLUG__";
const STORE_KEY = "scene-gt-" + SLUG;
const COLORS = Array.from({length: 24}, (_, i) => `hsl(${(i * 137.5) % 360} 55% 45%)`);

let state = { groups: [], openPlan: [], duplicates: [], preferred: [], mustNot: [], notes: "" };
let activeGroup = null, mode = "assign", pairFirst = null;

function save() { localStorage.setItem(STORE_KEY, JSON.stringify(state)); }
function load() {
  try { const s = JSON.parse(localStorage.getItem(STORE_KEY)); if (s && s.groups) state = s; } catch (e) {}
  if (state.groups.length) activeGroup = state.groups[0].id;
}
function groupOf(pos) { return state.groups.find(g => g.positions.includes(pos)); }
function slugify(t) { return t.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""); }

function addGroup() {
  const id = Date.now() + Math.random();
  const g = { id, name: "room-" + (state.groups.length + 1), positions: [] };
  state.groups.push(g); activeGroup = id; render(); save();
  return g;
}

function togglePhoto(pos) {
  if (mode === "assign") {
    if (!activeGroup) addGroup();
    const g = state.groups.find(g => g.id === activeGroup);
    const owner = groupOf(pos);
    if (owner === g) { g.positions = g.positions.filter(p => p !== pos); }
    else {
      if (owner) owner.positions = owner.positions.filter(p => p !== pos);
      g.positions.push(pos); g.positions.sort((a, b) => a - b);
    }
    render(); save();
  } else {
    if (pairFirst === null) { pairFirst = pos; render(); return; }
    if (pairFirst !== pos) {
      const pair = [Math.min(pairFirst, pos), Math.max(pairFirst, pos)];
      const list = mode === "dup" ? state.duplicates : mode === "pref" ? state.preferred : state.mustNot;
      if (!list.some(p => p[0] === pair[0] && p[1] === pair[1])) list.push(pair);
    }
    pairFirst = null; render(); save();
  }
}

function pairRow(list, i, container) {
  const [a, b] = list[i];
  const div = document.createElement("div");
  div.className = "pair";
  div.innerHTML = `<span>#${a} + #${b}</span>`;
  const del = document.createElement("button");
  del.className = "del"; del.textContent = "✕";
  del.onclick = () => { list.splice(i, 1); render(); save(); };
  div.appendChild(del); container.appendChild(div);
}

function render() {
  const grid = document.getElementById("grid");
  if (!grid.childElementCount) {
    for (const p of PHOTOS) {
      const tile = document.createElement("div");
      tile.className = "tile"; tile.dataset.pos = p.position;
      tile.innerHTML = `<img src="${p.data}" loading="lazy"><span class="pos">#${p.position}</span><button class="zoom">🔍</button><span class="badge"></span>`;
      tile.onclick = e => { if (e.target.classList.contains("zoom")) return; togglePhoto(p.position); };
      tile.querySelector(".zoom").onclick = () => showLightbox(p);
      grid.appendChild(tile);
    }
  }
  for (const tile of grid.children) {
    const pos = +tile.dataset.pos;
    const g = groupOf(pos);
    const idx = g ? state.groups.indexOf(g) : -1;
    tile.style.outlineColor = g ? COLORS[idx % COLORS.length] : "transparent";
    tile.querySelector(".badge").textContent = g ? g.name : "";
    tile.querySelector(".badge").style.display = g ? "block" : "none";
    tile.classList.toggle("pairflash", pairFirst === pos);
  }
  const groups = document.getElementById("groups"); groups.innerHTML = "";
  state.groups.forEach((g, i) => {
    const row = document.createElement("div");
    row.className = "group" + (g.id === activeGroup ? " active" : "");
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = COLORS[i % COLORS.length];
    const input = document.createElement("input"); input.value = g.name;
    input.onchange = () => { g.name = slugify(input.value) || g.name; render(); save(); };
    input.onclick = e => e.stopPropagation();
    const count = document.createElement("span"); count.className = "count"; count.textContent = g.positions.length;
    const del = document.createElement("button"); del.className = "del"; del.textContent = "✕";
    del.onclick = e => { e.stopPropagation(); state.groups = state.groups.filter(x => x !== g);
      state.openPlan = state.openPlan.map(s => s.filter(n => n !== g.name)).filter(s => s.length > 1);
      if (activeGroup === g.id) activeGroup = state.groups[0]?.id ?? null; render(); save(); };
    row.append(sw, input, count, del);
    row.onclick = () => { activeGroup = g.id; setMode("assign"); render(); };
    groups.appendChild(row);
  });
  const oplist = document.getElementById("oplist"); oplist.innerHTML = "";
  state.groups.forEach(g => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${g.name}"> ${g.name}`;
    oplist.appendChild(label);
  });
  const opsets = document.getElementById("opsets"); opsets.innerHTML = "";
  state.openPlan.forEach((set, i) => {
    const div = document.createElement("div"); div.className = "opset";
    div.innerHTML = `<span>${set.join(" + ")}</span>`;
    const del = document.createElement("button"); del.className = "del"; del.textContent = "✕";
    del.onclick = () => { state.openPlan.splice(i, 1); render(); save(); };
    div.appendChild(del); opsets.appendChild(div);
  });
  const dups = document.getElementById("dups"); dups.innerHTML = "";
  state.duplicates.forEach((_, i) => pairRow(state.duplicates, i, dups));
  const prefs = document.getElementById("prefs"); prefs.innerHTML = "";
  state.preferred.forEach((_, i) => pairRow(state.preferred, i, prefs));
  const negs = document.getElementById("negs"); negs.innerHTML = "";
  state.mustNot.forEach((_, i) => pairRow(state.mustNot, i, negs));
  const assigned = state.groups.reduce((n, g) => n + g.positions.length, 0);
  document.getElementById("status").textContent =
    `${assigned}/${PHOTOS.length} assigned · ${state.groups.length} rooms · mode: ${mode}` + (pairFirst !== null ? ` · first: #${pairFirst}` : "");
  document.getElementById("notes").value = state.notes;
}

function setMode(m) {
  mode = m; pairFirst = null;
  document.querySelectorAll("#modes button").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  render();
}

function buildExport() {
  return {
    listing: SLUG,
    job_ids: [],
    photos: PHOTOS.map(p => ({ position: p.position, filename: p.filename })),
    room_instances: state.groups.filter(g => g.positions.length).map(g => ({ instance: g.name, positions: g.positions })),
    open_plan_groups: state.openPlan,
    duplicates: state.duplicates,
    must_not_group: state.mustNot,
    preferred_cinematic_pairs: state.preferred,
    notes: state.notes,
    tool: "label_scene_groups v1",
  };
}

function showLightbox(p) {
  const lb = document.getElementById("lightbox");
  lb.querySelector("img").src = p.data;
  lb.querySelector(".cap").textContent = `#${p.position} — ${p.filename}`;
  lb.style.display = "flex";
}

document.getElementById("lightbox").onclick = () => document.getElementById("lightbox").style.display = "none";
document.addEventListener("keydown", e => {
  if (e.key === "Escape") document.getElementById("lightbox").style.display = "none";
  if (e.key.toLowerCase() === "n" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") addGroup();
});
document.getElementById("modes").onclick = e => { if (e.target.dataset.mode) setMode(e.target.dataset.mode); };
document.getElementById("newGroup").onclick = addGroup;
document.getElementById("linkOp").onclick = () => {
  const ticked = [...document.querySelectorAll("#oplist input:checked")].map(i => i.value);
  if (ticked.length > 1) { state.openPlan.push(ticked); render(); save(); }
};
document.getElementById("notes").onchange = e => { state.notes = e.target.value; save(); };
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ground_truth-${SLUG}.json`;
  a.click();
};
document.getElementById("import").onchange = e => {
  const file = e.target.files[0]; if (!file) return;
  file.text().then(text => {
    const data = JSON.parse(text);
    state.groups = (data.room_instances || []).map((r, i) => ({ id: i + 1, name: r.instance, positions: r.positions }));
    state.openPlan = data.open_plan_groups || [];
    state.duplicates = data.duplicates || [];
    state.preferred = data.preferred_cinematic_pairs || [];
    state.mustNot = data.must_not_group || [];
    state.notes = data.notes || "";
    activeGroup = state.groups[0]?.id ?? null;
    render(); save();
  });
};
document.getElementById("reset").onclick = () => {
  if (confirm("Clear all labeling for this listing?")) {
    localStorage.removeItem(STORE_KEY);
    state = { groups: [], openPlan: [], duplicates: [], preferred: [], mustNot: [], notes: "" };
    activeGroup = null; render();
  }
};

load(); render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path.home() / "Desktop" / "scene-labeling")
    parser.add_argument("--slug", type=str, default=None)
    args = parser.parse_args()

    slug = args.slug or re.sub(r"[^a-z0-9]+", "-", args.folder.name.lower()).strip("-")
    photos = collect_photos(args.folder)
    html = (
        HTML_TEMPLATE
        .replace("__PHOTOS_JSON__", json.dumps(photos))
        .replace("__SLUG__", slug)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"label-{slug}.html"
    target.write_text(html)
    print(f"\nWrote {target} ({len(photos)} photos, {target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
