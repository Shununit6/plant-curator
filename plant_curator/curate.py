"""Visual curation loop: localhost browser app, one photo at a time.

Run via `plant-curator curate FOLDER`.

Keys (in the browser):
    1 = love   (mark liked + copy to picks/)
    2 = like   (mark liked, no copy)
    3 = no     (mark disliked)
    space = skip (no mark, advance)
    ←      = previous photo (no decision change)
    →      = next photo (no decision change)
    u      = clear current photo's decision

The queue includes every photo in the folder, so you can scroll back to
already-decided photos and overwrite the mark by pressing 1/2/3 again.
"""
import shutil
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import click
from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image, ImageOps

from . import cache as cache_mod
from .ingest import list_photos

_THUMB_PX = 200
_THUMB_CACHE: dict[int, bytes] = {}


def _thumb_bytes(path: Path) -> bytes:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((_THUMB_PX, _THUMB_PX))
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=72)
    return buf.getvalue()


def _send_to_trash(path: Path) -> str:
    """Move path into ~/.Trash (macOS). Returns destination name."""
    trash = Path.home() / ".Trash"
    trash.mkdir(exist_ok=True)
    dest = trash / path.name
    i = 1
    while dest.exists():
        dest = trash / f"{path.stem} {i}{path.suffix}"
        i += 1
    shutil.move(str(path), str(dest))
    return dest.name

_PORT = 5750

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>plant-curator</title>
<style>
  html, body { margin: 0; height: 100%; background: #0a0a0a; color: #ddd;
               font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif; }
  body { display: flex; flex-direction: column; }
  .stage { flex: 1; display: flex; align-items: center; justify-content: center;
           overflow: hidden; padding: 24px; position: relative; }
  .stage img { max-width: 100%; max-height: 100%; object-fit: contain;
               box-shadow: 0 12px 40px rgba(0,0,0,0.6); border-radius: 2px; }
  .badge { position: absolute; top: 18px; left: 18px;
           padding: 6px 12px; border-radius: 14px; font-size: 13px;
           font-weight: 500; letter-spacing: 0.3px;
           background: rgba(0,0,0,0.55); backdrop-filter: blur(6px);
           border: 1px solid rgba(255,255,255,0.08); display: none; }
  .badge.love { color: #ff6b88; border-color: rgba(255,107,136,0.35); display: inline-block; }
  .badge.like { color: #ddd; display: inline-block; }
  .badge.no   { color: #888; display: inline-block; }
  .bar { padding: 16px 28px; display: flex; justify-content: space-between;
         align-items: center; font-size: 13px; color: #888;
         border-top: 1px solid #1a1a1a; }
  .keys span { margin-right: 18px; }
  .keys .k { display: inline-block; min-width: 22px; padding: 2px 7px;
             margin-right: 6px; border: 1px solid #333; border-radius: 3px;
             background: #151515; color: #ddd; font-family: ui-monospace, Menlo, monospace;
             font-size: 11px; text-align: center; }
  .stats { color: #aaa; font-variant-numeric: tabular-nums; }
  .stats .liked { color: #ff6b88; }
  .stats .no { color: #777; }
  .stats .pending { color: #555; }
  .meta { font-variant-numeric: tabular-nums; }
  .done { font-size: 22px; text-align: center; line-height: 1.7; color: #ddd; }
  .done small { display: block; margin-top: 30px; color: #555; font-size: 13px; }
  .done .ask { margin-top: 28px; color: #aaa; font-size: 15px; }
  .done .btn-delete {
      margin-top: 14px; padding: 12px 22px; font-size: 14px;
      background: #1a1a1a; color: #ff8a9d; border: 1px solid #4a2030;
      border-radius: 4px; cursor: pointer; font-family: inherit; }
  .done .btn-delete:hover { background: #2a1218; border-color: #ff6b88; color: #fff; }
  .done .deleted { color: #777; font-size: 14px; margin-top: 22px; }
  .flash { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
           font-size: 80px; pointer-events: none; opacity: 0; transition: opacity 0.18s; }
  .flash.show { opacity: 1; }
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85);
             display: none; align-items: center; justify-content: center; z-index: 10; }
  .overlay.show { display: flex; }
  .card { background: #111; border: 1px solid #222; border-radius: 8px;
          padding: 32px 36px; min-width: 420px; max-width: min(900px, 90vw);
          max-height: 86vh; overflow-y: auto; text-align: center; }
  .card h2 { margin: 0 0 6px; font-size: 18px; font-weight: 500; color: #eee; }
  .card p { margin: 0 0 24px; color: #888; font-size: 13px; }
  .card button { display: block; width: 100%; margin: 8px 0;
                 padding: 12px 18px; font-size: 14px;
                 background: #1a1a1a; color: #ddd; border: 1px solid #333;
                 border-radius: 4px; cursor: pointer;
                 font-family: inherit; text-align: left; }
  .card button:hover { background: #222; border-color: #555; color: #fff; }
  .card .row { display: flex; gap: 8px; margin-top: 8px; }
  .card .row input { flex: 1; padding: 12px; background: #1a1a1a; color: #ddd;
                     border: 1px solid #333; border-radius: 4px; font-family: inherit;
                     font-size: 14px; }
  .card .row button { width: auto; padding: 0 18px; margin: 0; }
  .thumbs { margin-top: 16px; display: grid;
            grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
            gap: 8px; }
  .thumbs .thumb { position: relative; background: #0a0a0a;
                   border: 1px solid #1f1f1f; border-radius: 3px;
                   aspect-ratio: 1 / 1; overflow: hidden; cursor: pointer; }
  .thumbs .thumb:hover { border-color: #ff6b88; }
  .thumbs .thumb img { width: 100%; height: 100%; object-fit: contain;
                       background: #000; display: block; }
  .thumbs .thumb .n { position: absolute; bottom: 0; left: 0; right: 0;
                      background: linear-gradient(transparent, rgba(0,0,0,0.85));
                      color: #fff; font-size: 11px; font-family: ui-monospace, Menlo, monospace;
                      padding: 12px 4px 3px; text-align: center; }
  .thumbs .thumb.marked-love::after,
  .thumbs .thumb.marked-like::after,
  .thumbs .thumb.marked-no::after {
    content: ''; position: absolute; top: 4px; right: 4px;
    width: 8px; height: 8px; border-radius: 50%; }
  .thumbs .thumb.marked-love::after { background: #ff6b88; }
  .thumbs .thumb.marked-like::after { background: #ddd; }
  .thumbs .thumb.marked-no::after   { background: #555; }
  .thumb-hint { margin-top: 14px; color: #555; font-size: 11px; text-align: left; }
</style>
</head>
<body>
  <div class="stage" id="stage"></div>
  <div class="flash" id="flash"></div>
  <div class="overlay" id="overlay">
    <div class="card">
      <h2>Where do you want to start?</h2>
      <p id="overlay-summary"></p>
      <button id="btn-resume"></button>
      <button id="btn-beginning">Start from beginning (#1)</button>
      <div class="row">
        <input type="text" inputmode="numeric" id="jump-input" placeholder="jump to #… (type digits to filter)">
        <button id="btn-jump">Go</button>
      </div>
      <div class="thumbs" id="jump-results"></div>
      <div class="thumb-hint" id="jump-hint"></div>
    </div>
  </div>
  <div class="bar">
    <div class="meta" id="meta"></div>
    <div class="keys">
      <span><span class="k">1</span>love</span>
      <span><span class="k">2</span>like</span>
      <span><span class="k">3</span>no</span>
      <span><span class="k">space</span>skip</span>
      <span><span class="k">&larr;</span><span class="k">&rarr;</span>nav</span>
      <span><span class="k">u</span>clear</span>
    </div>
    <div class="stats" id="stats"></div>
  </div>
<script>
let current = null;
let busy = false;
let overlayOpen = false;
let allDecisions = [];  // one entry per photo, 0-indexed: 'love'|'like'|'no'|null

const BADGES = { love: ['love', '♥ loved'], like: ['like', '· liked'], no: ['no', '✕ no'] };
const MAX_THUMBS = 100;

async function showOverlay(d) {
  overlayOpen = true;
  const total = d.total || (d.current ? d.current.total : 0);
  document.getElementById('overlay-summary').textContent =
    `${d.stats.liked + d.stats.no} of ${total} already decided`;
  document.getElementById('btn-resume').textContent =
    `Resume from first undecided (#${d.default_idx + 1})`;
  const jump = document.getElementById('jump-input');
  jump.value = '';
  document.getElementById('jump-results').innerHTML = '';
  document.getElementById('jump-hint').textContent = '';
  document.getElementById('overlay').classList.add('show');
  setTimeout(() => jump.focus(), 50);
  // fetch the full decision list once so thumbnails can show prior marks
  try {
    const r = await fetch('/api/decisions');
    allDecisions = (await r.json()).decisions || [];
  } catch (_) { allDecisions = new Array(total).fill(null); }
}

function renderThumbs(prefix) {
  const results = document.getElementById('jump-results');
  const hint = document.getElementById('jump-hint');
  results.innerHTML = '';
  if (!prefix) { hint.textContent = ''; return; }
  const total = allDecisions.length;
  const matches = [];
  for (let i = 1; i <= total; i++) {
    if (String(i).startsWith(prefix)) {
      matches.push(i);
      if (matches.length >= MAX_THUMBS) break;
    }
  }
  if (matches.length === 0) {
    hint.textContent = `no photos match #${prefix}…`;
    return;
  }
  // count total matches without cap, for the hint
  let totalMatches = 0;
  for (let i = 1; i <= total; i++) if (String(i).startsWith(prefix)) totalMatches++;
  hint.textContent = totalMatches > MAX_THUMBS
    ? `showing first ${MAX_THUMBS} of ${totalMatches} matches`
    : `${totalMatches} match${totalMatches === 1 ? '' : 'es'} · click a thumbnail to jump`;
  for (const n of matches) {
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const d = allDecisions[n - 1];
    if (d) thumb.classList.add('marked-' + d);
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = `/thumb/idx/${n}`;
    img.alt = '#' + n;
    const label = document.createElement('div');
    label.className = 'n';
    label.textContent = '#' + n;
    thumb.appendChild(img);
    thumb.appendChild(label);
    thumb.onclick = () => start(n);
    results.appendChild(thumb);
  }
}

function hideOverlay() {
  overlayOpen = false;
  document.getElementById('overlay').classList.remove('show');
}

async function start(from) {
  await fetch('/api/start', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({from}) });
  hideOverlay();
  await loadState();
}

document.getElementById('btn-resume').onclick = () => start('resume');
document.getElementById('btn-beginning').onclick = () => start('beginning');
document.getElementById('btn-jump').onclick = () => {
  const n = parseInt(document.getElementById('jump-input').value, 10);
  if (Number.isInteger(n) && n >= 1) start(n);
};
document.getElementById('jump-input').addEventListener('input', (e) => {
  const cleaned = e.target.value.replace(/[^0-9]/g, '').slice(0, 4);
  if (e.target.value !== cleaned) e.target.value = cleaned;
  renderThumbs(cleaned);
});
document.getElementById('jump-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); document.getElementById('btn-jump').click(); }
});

async function loadState() {
  const r = await fetch('/api/state');
  const d = await r.json();
  const stage = document.getElementById('stage');
  const meta = document.getElementById('meta');
  const stats = document.getElementById('stats');

  if (!d.started) { showOverlay(d); }

  if (d.done) {
    const nos = d.stats.no;
    const noteHtml = window._deletedSummary
      ? `<div class="deleted">${window._deletedSummary}</div>`
      : '';
    const askHtml = nos > 0
      ? `<div class="ask">Move all ${nos} marked-no photos to Trash?</div>
         <button class="btn-delete" id="btn-delete-nos">Yes, move ${nos} to Trash</button>`
      : '';
    stage.innerHTML = `<div class="done">all done<br><br>
      <span style="color:#ff6b88">${d.stats.liked} liked</span> &middot;
      <span style="color:#777">${nos} no</span>
      ${askHtml}
      ${noteHtml}
      <small>close this tab when done</small></div>`;
    meta.textContent = '';
    stats.textContent = '';
    current = null;
    if (nos > 0) {
      document.getElementById('btn-delete-nos').onclick = async () => {
        if (!confirm(`Move ${nos} photos to ~/.Trash? You can restore from Trash if needed.`)) return;
        const r = await fetch('/api/delete-nos', { method: 'POST' });
        const result = await r.json();
        if (!result.ok) {
          alert(result.error || 'delete failed');
          return;
        }
        const errs = (result.errors || []).length;
        window._deletedSummary =
          `${result.deleted} moved to Trash` + (errs ? ` · ${errs} failed` : '');
        await loadState();
      };
    }
    return;
  }

  current = d.current;
  const img = new Image();
  img.src = '/image/' + current.hash;
  img.onload = () => {
    stage.innerHTML = '';
    stage.appendChild(img);
    if (current.decision && BADGES[current.decision]) {
      const b = document.createElement('div');
      const [cls, label] = BADGES[current.decision];
      b.className = 'badge ' + cls;
      b.textContent = label;
      stage.appendChild(b);
    }
  };
  meta.textContent = `${current.idx + 1} / ${current.total}` +
    `   ·   ${current.name}` +
    (current.captured_at ? `   ·   ${current.captured_at}` : '');
  stats.innerHTML =
    `<span class="liked">${d.stats.liked}</span> &middot; ` +
    `<span class="no">${d.stats.no}</span> &middot; ` +
    `<span class="pending">${d.stats.pending}</span>`;

  if (d.next_hash) { new Image().src = '/image/' + d.next_hash; }
  if (d.prev_hash) { new Image().src = '/image/' + d.prev_hash; }
}

function flash(symbol, color) {
  const f = document.getElementById('flash');
  f.textContent = symbol;
  f.style.color = color;
  f.classList.add('show');
  setTimeout(() => f.classList.remove('show'), 200);
}

async function decide(decision) {
  if (busy || !current) return;
  busy = true;
  const flashes = { love: ['♥', '#ff6b88'], like: ['·', '#ddd'],
                    no: ['✕', '#777'], skip: ['→', '#444'] };
  flash(...flashes[decision]);
  await fetch('/api/decide', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({hash: current.hash, decision}) });
  await loadState();
  busy = false;
}

async function nav(direction) {
  if (busy) return;
  busy = true;
  flash(direction === 'back' ? '←' : '→', '#666');
  await fetch('/api/nav', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({direction}) });
  await loadState();
  busy = false;
}

async function clearMark() {
  if (busy || !current) return;
  busy = true;
  flash('↺', '#888');
  await fetch('/api/clear', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({hash: current.hash}) });
  await loadState();
  busy = false;
}

document.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  if (overlayOpen) return;
  if (e.key === '1') decide('love');
  else if (e.key === '2') decide('like');
  else if (e.key === '3') decide('no');
  else if (e.key === ' ') { e.preventDefault(); decide('skip'); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); nav('back'); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); nav('forward'); }
  else if (e.key === 'u' || e.key === 'U') clearMark();
});

loadState();
</script>
</body>
</html>
"""


@dataclass
class _Item:
    hash: str
    path: Path
    captured_at: Optional[datetime]
    decision: Optional[str] = None  # 'love' | 'like' | 'no' | None


@dataclass
class _State:
    folder: Path
    picks_dir: Path
    queue: list = field(default_factory=list)
    idx: int = 0
    default_idx: int = 0  # first-undecided position computed at startup
    started: bool = False  # set True once the user picks a start position

    def current(self) -> Optional[_Item]:
        return self.queue[self.idx] if 0 <= self.idx < len(self.queue) else None

    def at(self, offset: int) -> Optional[_Item]:
        j = self.idx + offset
        return self.queue[j] if 0 <= j < len(self.queue) else None

    def find(self, h: str) -> Optional[_Item]:
        return next((it for it in self.queue if it.hash == h), None)

    def _reverse(self, item: _Item) -> None:
        if item.decision == "love":
            dest = self.picks_dir / item.path.name
            if dest.exists():
                dest.unlink()
        item.decision = None

    def decide(self, h: str, decision: str) -> bool:
        cur = self.current()
        if not cur or cur.hash != h:
            return False
        if decision not in ("love", "like", "no", "skip"):
            return False
        self._reverse(cur)
        if decision == "love":
            cache_mod.set_liked(h, True)
            self.picks_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cur.path, self.picks_dir / cur.path.name)
            cur.decision = "love"
        elif decision == "like":
            cache_mod.set_liked(h, True)
            cur.decision = "like"
        elif decision == "no":
            cache_mod.set_disliked(h, True)
            cur.decision = "no"
        elif decision == "skip":
            cache_mod.set_liked(h, False)  # clears to state 0
            cur.decision = None
        self.idx = min(len(self.queue), self.idx + 1)
        return True

    def clear(self, h: str) -> bool:
        cur = self.current()
        if not cur or cur.hash != h:
            return False
        self._reverse(cur)
        cache_mod.set_liked(h, False)  # state -> 0
        return True

    def back(self) -> None:
        self.idx = max(0, self.idx - 1)

    def forward(self) -> None:
        self.idx = min(len(self.queue), self.idx + 1)

    def stats(self) -> dict:
        s = {"liked": 0, "no": 0, "pending": 0}
        for it in self.queue:
            if it.decision in ("love", "like"):
                s["liked"] += 1
            elif it.decision == "no":
                s["no"] += 1
            else:
                s["pending"] += 1
        return s


def _build_queue(folder: Path, picks_dir: Path) -> tuple[list, int]:
    from .cli import _analyze  # late import: avoids circular
    photos = list(list_photos(folder))
    if not photos:
        return [], 0
    click.echo(f"Found {len(photos)} photos. Preparing…")
    rows = _analyze(photos, with_embeddings=True)
    queue: list[_Item] = []
    for ph, _s, _e in rows:
        h = cache_mod.file_hash(ph.path)
        state = cache_mod.get_state(h)
        if state == -1:
            decision = "no"
        elif state == 1:
            decision = "love" if (picks_dir / ph.path.name).exists() else "like"
        else:
            decision = None
        queue.append(_Item(hash=h, path=ph.path, captured_at=ph.captured_at, decision=decision))
    queue.sort(key=lambda it: it.captured_at or datetime.max)
    start = next((i for i, it in enumerate(queue) if it.decision is None), len(queue))
    return queue, start


def serve(folder: Path, picks_dir: Path, port: int = _PORT) -> None:
    state = _State(folder=folder, picks_dir=picks_dir)
    state.queue, state.default_idx = _build_queue(folder, picks_dir)
    state.idx = state.default_idx

    if not state.queue:
        click.echo("No photos in that folder.")
        return

    decided = sum(1 for it in state.queue if it.decision is not None)
    pending = len(state.queue) - decided
    if decided == 0 or pending == 0:
        state.started = True  # no meaningful resume/beginning choice
    click.echo(f"{decided} already decided, {pending} pending.")
    click.echo(f"Opening http://127.0.0.1:{port}/  …  Ctrl+C here to stop.\n")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(_HTML)

    @app.route("/api/state")
    def api_state():
        cur = state.current()
        if not cur:
            return jsonify({"done": True, "started": True, "stats": state.stats()})
        nxt = state.at(1)
        prv = state.at(-1)
        return jsonify({
            "done": False,
            "started": state.started,
            "default_idx": state.default_idx,
            "total": len(state.queue),
            "current": {
                "hash": cur.hash,
                "idx": state.idx,
                "total": len(state.queue),
                "captured_at": cur.captured_at.strftime("%Y-%m-%d %H:%M") if cur.captured_at else None,
                "decision": cur.decision,
                "name": cur.path.stem,
            },
            "next_hash": nxt.hash if nxt else None,
            "prev_hash": prv.hash if prv else None,
            "stats": state.stats(),
        })

    @app.route("/api/decide", methods=["POST"])
    def api_decide():
        body = request.get_json(force=True)
        ok = state.decide(body.get("hash", ""), body.get("decision", ""))
        return jsonify({"ok": ok})

    @app.route("/api/nav", methods=["POST"])
    def api_nav():
        body = request.get_json(force=True)
        d = body.get("direction", "")
        if d == "back":
            state.back()
        elif d == "forward":
            state.forward()
        return jsonify({"ok": True, "idx": state.idx})

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        body = request.get_json(force=True)
        ok = state.clear(body.get("hash", ""))
        return jsonify({"ok": ok})

    @app.route("/api/decisions")
    def api_decisions():
        return jsonify({"decisions": [it.decision for it in state.queue]})

    @app.route("/api/delete-nos", methods=["POST"])
    def api_delete_nos():
        # Refuse unless every photo in the queue has been decided. This is the
        # "you viewed the whole folder" safety check.
        pending = sum(1 for it in state.queue if it.decision is None)
        if pending > 0:
            return jsonify({"ok": False, "error": "still pending photos",
                            "pending": pending}), 400
        deleted, errors = 0, []
        kept: list[_Item] = []
        for it in state.queue:
            if it.decision != "no":
                kept.append(it)
                continue
            try:
                if it.path.exists():
                    _send_to_trash(it.path)
                deleted += 1
            except Exception as e:
                errors.append({"name": it.path.name, "error": str(e)})
                kept.append(it)  # keep in queue so user sees what failed
        state.queue = kept
        state.idx = len(state.queue)  # stays on the "done" screen
        return jsonify({"ok": True, "deleted": deleted, "errors": errors})

    @app.route("/api/start", methods=["POST"])
    def api_start():
        body = request.get_json(force=True)
        choice = body.get("from")
        if choice == "resume":
            state.idx = state.default_idx
        elif choice == "beginning":
            state.idx = 0
        elif isinstance(choice, int) and 1 <= choice <= len(state.queue):
            state.idx = choice - 1
        else:
            return jsonify({"ok": False, "error": "bad 'from'"}), 400
        state.started = True
        return jsonify({"ok": True, "idx": state.idx})

    @app.route("/image/<h>")
    def image(h: str):
        item = state.find(h)
        if not item:
            return "not found", 404
        return send_file(item.path, mimetype="image/jpeg")

    @app.route("/thumb/idx/<int:n>")
    def thumb_by_idx(n: int):
        if not (1 <= n <= len(state.queue)):
            return "out of range", 404
        if n not in _THUMB_CACHE:
            _THUMB_CACHE[n] = _thumb_bytes(state.queue[n - 1].path)
        return send_file(BytesIO(_THUMB_CACHE[n]), mimetype="image/jpeg")

    threading.Timer(0.7, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    try:
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=False)
    except KeyboardInterrupt:
        pass
    finally:
        s = state.stats()
        click.echo(f"\n{s['liked']} liked · {s['no']} no · {s['pending']} pending")
