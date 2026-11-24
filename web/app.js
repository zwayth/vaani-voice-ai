// Vaani dashboard: live call client + operations console.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const WS = (p) => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${p}`;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtMs = (v) => (v == null ? '–' : v >= 1000 ? (v / 1000).toFixed(2) + 's' : Math.round(v) + 'ms');
let STATE = { settings: {}, providers: [], keys: {} };

/* ---------------- navigation ---------------- */
const TITLES = {
  console: ['Live Console', 'Talk to the agent exactly as your customers will.'],
  calls: ['Active Calls', 'Every concurrent conversation, live. Operators can step in or end any call.'],
  studio: ['Agent Studio', 'Persona, knowledge, voices and expressive delivery.'],
  providers: ['Providers', 'Failover chains for each stage, plus their health and latency.'],
  logs: ['Call Logs', 'Transcripts and per-turn latency for every finished call.'],
};
$$('.nav').forEach((b) => b.onclick = () => {
  $$('.nav').forEach((x) => x.classList.toggle('on', x === b));
  $$('.view').forEach((v) => v.classList.toggle('on', v.id === 'v-' + b.dataset.view));
  const [t, s] = TITLES[b.dataset.view]; $('#viewTitle').textContent = t; $('#viewSub').textContent = s;
  if (b.dataset.view === 'logs') loadLogs();
  if (b.dataset.view === 'studio') loadVoices();
});

/* ---------------- state + monitor ---------------- */
async function loadState() {
  STATE = await (await fetch('/api/state')).json();
  fillStudio(); renderProviders(STATE.providers); renderKpis(STATE.stats); renderCalls(STATE.calls);
  $('#agentCompany').textContent = `${STATE.settings.company} · customer support`;
  const premium = ['groq', 'openai', 'gemini', 'anthropic', 'openrouter'].some((k) => STATE.keys[k]);
  if (!premium) {
    const b = $('#banner'); b.classList.remove('hidden');
    b.innerHTML = 'Running on the <b>free fallback stack</b>, so replies take 1–3s. For sub-second replies, add a free <code>GROQ_API_KEY</code> to <code>D:\\voice\\.env</code> and restart (optionally add <code>DEEPGRAM_API_KEY</code>, <code>ELEVENLABS_API_KEY</code> / <code>CARTESIA_API_KEY</code> / <code>SARVAM_API_KEY</code>).';
  }
}
function renderKpis(s) {
  if (!s) return;
  $('#kActive').textContent = s.active; $('#navActive').textContent = s.active;
  $('#kP50').textContent = fmtMs(s.p50); $('#kP90').textContent = fmtMs(s.p90); $('#kTotal').textContent = s.total;
  $('#liveDot').classList.toggle('on', s.active > 0);
}
function monitor() {
  const ws = new WebSocket(WS('/ws/monitor'));
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'tick') { renderKpis(ev.stats); renderCalls(ev.calls); renderProviders(ev.providers); }
    if (ev.type === 'transcript' && callLines[ev.call_id] !== undefined) callLines[ev.call_id] = `${ev.role === 'user' ? '🗣' : '✦'} ${ev.text}`;
  };
  ws.onclose = () => setTimeout(monitor, 1500);
}
const callLines = {};

/* ---------------- active calls grid ---------------- */
function renderCalls(calls) {
  const g = $('#callGrid');
  if (!calls || !calls.length) { g.innerHTML = '<div class="empty">No live calls right now.</div>'; return; }
  const focused = document.activeElement && document.activeElement.closest('.cc');
  if (focused) return; // don't clobber an operator typing
  g.innerHTML = calls.map((c) => {
    if (callLines[c.id] === undefined) callLines[c.id] = c.last_text || '';
    const m = Math.floor(c.duration / 60), s = String(c.duration % 60).padStart(2, '0');
    return `<div class="card cc">
      <div class="cc-h"><div><b>#${c.id}</b> <span class="muted">· ${esc(c.caller)} · ${c.channel}</span></div>
        ${c.state === 'speaking' ? '<div class="eq"><i></i><i></i><i></i><i></i></div>' : ''}
        <span class="chip s-${c.state}">${c.state}</span></div>
      <div class="cc-line">${esc(callLines[c.id] || c.last_text || '…')}</div>
      <div class="cc-stats"><div><span>Duration</span><b>${m}:${s}</b></div><div><span>Turns</span><b>${c.turns}</b></div>
        <div><span>Avg v2v</span><b>${fmtMs(c.avg_e2e)}</b></div></div>
      <div class="cc-stats"><div><span>STT</span><b>${c.providers.stt}</b></div><div><span>LLM</span><b>${c.providers.llm}</b></div><div><span>Lang</span><b>${c.lang.toUpperCase()}</b></div></div>
      <div class="cc-act"><input placeholder="Make the agent say…" data-id="${c.id}" />
        <button class="btn small" data-say="${c.id}">Say</button><button class="btn small danger" data-hang="${c.id}">End</button></div>
    </div>`;
  }).join('');
}
$('#callGrid').addEventListener('click', async (e) => {
  const say = e.target.dataset.say, hang = e.target.dataset.hang;
  if (say) {
    const inp = $(`input[data-id="${say}"]`); if (!inp.value.trim()) return;
    await fetch(`/api/calls/${say}/say`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: inp.value }) });
    inp.value = ''; inp.blur();
  }
  if (hang) await fetch(`/api/calls/${hang}/hangup`, { method: 'POST' });
});

/* ---------------- live call client ---------------- */
let call = null, lang = 'auto', engine = 'cascade';
$$('#engineSeg button').forEach((b) => b.onclick = () => {
  if (call) return;
  $$('#engineSeg button').forEach((x) => x.classList.toggle('on', x === b)); engine = b.dataset.engine;
});
function markEngineAvailability() {
  const ok = STATE.keys && STATE.keys.gemini, b = $('#engineSeg [data-engine=gemini_live]');
  b.classList.toggle('na', !ok);
  b.querySelector('small').textContent = ok ? 'Gemini Live · speech-native' : 'needs GEMINI_API_KEY · falls back';
}
$$('#langSeg button').forEach((b) => b.onclick = () => {
  if (call) return;
  $$('#langSeg button').forEach((x) => x.classList.toggle('on', x === b)); lang = b.dataset.lang;
});
const STATE_LABEL = { listening: 'Listening', hearing: 'Hearing you', thinking: 'Thinking', speaking: 'Speaking', connecting: 'Connecting', ended: 'Ended' };
let orbState = 'idle', micLevel = 0, outLevel = 0;
function setState(s) {
  orbState = s; const p = $('#statePill'); p.dataset.s = s; p.querySelector('span').textContent = STATE_LABEL[s] || 'Idle';
}

class Call {
  async start() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    await this.ctx.audioWorklet.addModule('/static/worklet.js');
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'capture');
    src.connect(this.node);
    this.out = this.ctx.createGain(); this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 512;
    this.out.connect(this.analyser); this.analyser.connect(this.ctx.destination);
    this.buf = new Float32Array(512); this.next = 0; this.sources = new Set(); this.muted = false;
    this.ws = new WebSocket(WS(`/ws/call?lang=${lang}&engine=${engine}&name=Console`)); this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (e) => (typeof e.data === 'string' ? this.onEvent(JSON.parse(e.data)) : this.play(e.data));
    this.ws.onclose = () => this.cleanup();
    this.node.port.onmessage = ({ data }) => {
      micLevel = micLevel * 0.7 + data.rms * 0.3;
      if (!this.ws || this.ws.readyState !== 1) return;
      // duplex echo gate: while the agent is audible, only pass mic frames clearly louder than its echo
      let pcm = data.pcm;
      if (this.muted || (outLevel > 0.015 && data.rms < Math.max(0.03, outLevel * 0.55))) pcm = new Int16Array(320).buffer;
      this.ws.send(pcm);
    };
    this.raf = requestAnimationFrame(() => this.meter());
  }
  meter() {
    if (!this.analyser) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    let s = 0; for (const v of this.buf) s += v * v;
    outLevel = outLevel * 0.6 + Math.sqrt(s / this.buf.length) * 0.4;
    this.raf = requestAnimationFrame(() => this.meter());
  }
  play(ab) {
    const i16 = new Int16Array(ab), f = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
    const b = this.ctx.createBuffer(1, f.length, 24000); b.copyToChannel(f, 0);
    const s = this.ctx.createBufferSource(); s.buffer = b; s.connect(this.out);
    const t = Math.max(this.ctx.currentTime + 0.03, this.next); s.start(t); this.next = t + b.duration;
    this.sources.add(s); s.onended = () => this.sources.delete(s);
  }
  clear() { for (const s of this.sources) { try { s.stop(); } catch {} } this.sources.clear(); this.next = 0; }
  onEvent(ev) {
    switch (ev.type) {
      case 'hello': $('#callIdTag').textContent = '#' + ev.call_id + ' · ' + (ev.engine === 'gemini_live' ? 'native S2S' : 'cascade'); break;
      case 'engine': if (ev.failover) addSys('Native engine unavailable → call continues on the cascade engine'); $('#callIdTag').textContent = '#' + ev.call_id + ' · cascade'; break;
      case 'state': setState(ev.state); break;
      case 'clear': this.clear(); break;
      case 'interrupted': addSys('caller interrupted · agent yielded'); break;
      case 'transcript': addMsg(ev.role, ev.role === 'agent' && ev.raw ? ev.raw : ev.text); break;
      case 'agent_partial': partial(ev.text); break;
      case 'metrics': showMetrics(ev); break;
      case 'hangup': stopCall(); break;
      case 'error': addSys(ev.message); break;
    }
  }
  send(o) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(o)); }
  cleanup() {
    cancelAnimationFrame(this.raf);
    try { this.stream.getTracks().forEach((t) => t.stop()); } catch {}
    try { this.ctx.close(); } catch {}
    this.analyser = null; outLevel = 0; micLevel = 0;
    if (call === this) stopCall(true);
  }
  hangup() { this.send({ type: 'hangup' }); try { this.ws.close(); } catch {} }
}

async function startCall() {
  $('#msgs').innerHTML = '';
  const c = new Call(); call = c;
  $('#callBtn').classList.replace('primary', 'danger'); $('#callLbl').textContent = 'End call'; $('#muteBtn').disabled = false;
  setState('connecting');
  try { await c.start(); } catch (e) { addSys('Microphone unavailable: ' + e.message); stopCall(); }
}
function stopCall(fromClose) {
  const c = call; call = null;
  if (c && !fromClose) c.hangup();
  $('#callBtn').classList.replace('danger', 'primary'); $('#callLbl').textContent = 'Start call';
  $('#muteBtn').disabled = true; $('#muteBtn').textContent = 'Mute'; setState('idle'); $('#callIdTag').textContent = 'no call';
}
$('#callBtn').onclick = () => (call ? stopCall() : startCall());
$('#muteBtn').onclick = () => { if (!call) return; call.muted = !call.muted; $('#muteBtn').textContent = call.muted ? 'Unmute' : 'Mute'; };
$('#typeForm').onsubmit = (e) => { e.preventDefault(); const v = $('#typeIn').value.trim(); if (!v || !call) return; call.send({ type: 'text', text: v }); $('#typeIn').value = ''; };

/* ---------------- transcript ---------------- */
let partialEl = null;
const fx = (t) => esc(t).replace(/\[(\/?)(breath|sigh|laugh|pause|whisper|hmm|inhale|exhale)\]/gi, (m) => `<span class="fx">${m}</span>`);
function addMsg(role, text) {
  if (role === 'agent' && partialEl) { partialEl.remove(); partialEl = null; }
  const m = document.createElement('div'); m.className = 'msg ' + role;
  m.innerHTML = `<small>${role === 'user' ? 'Caller' : 'Aria'}</small>${fx(text)}`;
  $('#msgs').appendChild(m); $('#msgs').scrollTop = 1e9;
}
function partial(text) {
  if (!partialEl) { partialEl = document.createElement('div'); partialEl.className = 'msg agent partial'; $('#msgs').appendChild(partialEl); }
  partialEl.innerHTML = `<small>Aria · speaking</small>${esc(text)}`; $('#msgs').scrollTop = 1e9;
}
function addSys(t) { const m = document.createElement('div'); m.className = 'msg sys'; m.textContent = t; $('#msgs').appendChild(m); $('#msgs').scrollTop = 1e9; }
function showMetrics(m) {
  const max = Math.max(1500, m.e2e_ms || 0, (m.stt_ms || 0) + (m.llm_ttft_ms || 0));
  const w = (v) => (v ? Math.min(100, (v / max) * 100) + '%' : '0');
  $('#bStt').style.width = w(m.stt_ms); $('#tStt').textContent = fmtMs(m.stt_ms);
  $('#bLlm').style.width = w(m.llm_ttft_ms); $('#tLlm').textContent = fmtMs(m.llm_ttft_ms);
  $('#bE2e').style.width = w(m.e2e_ms); $('#tE2e').textContent = fmtMs(m.e2e_ms);
  $('#e2eTag').textContent = m.e2e_ms ? `voice-to-voice ${fmtMs(m.e2e_ms)}` : '–';
  $('#provs').innerHTML = `<span>STT <b>${m.stt || 'typed'}</b></span><span>LLM <b>${m.llm || '–'}</b></span><span>TTS <b>${m.tts || '–'}</b></span>`;
}

/* ---------------- orb visual ---------------- */
const COLORS = { idle: [90, 96, 117], connecting: [138, 144, 163], listening: [34, 225, 208], hearing: [200, 255, 46], thinking: [124, 92, 255], speaking: [255, 77, 109], ended: [90, 96, 117] };
let cur = [90, 96, 117], lvl = 0;
function orb(t) {
  const cv = $('#orb'), g = cv.getContext('2d'), W = cv.width, H = cv.height, cx = W / 2, cy = H / 2;
  const target = COLORS[orbState] || COLORS.idle;
  cur = cur.map((c, i) => c + (target[i] - c) * 0.06);
  const want = orbState === 'speaking' ? outLevel * 6 : orbState === 'hearing' ? micLevel * 8 : orbState === 'thinking' ? 0.15 + 0.1 * Math.sin(t / 120) : 0.04;
  lvl += (Math.min(1, want) - lvl) * 0.18;
  g.clearRect(0, 0, W, H);
  const [r, gg, b] = cur.map(Math.round), col = (a) => `rgba(${r},${gg},${b},${a})`;
  const glow = g.createRadialGradient(cx, cy, 40, cx, cy, 270); glow.addColorStop(0, col(0.28 + lvl * 0.3)); glow.addColorStop(1, col(0));
  g.fillStyle = glow; g.fillRect(0, 0, W, H);
  for (let ring = 0; ring < 3; ring++) {
    g.beginPath(); const R = 150 + ring * 30 + lvl * 40 * (ring + 1);
    for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI / 90) {
      const n = Math.sin(a * (3 + ring) + t / (700 - ring * 150)) * (6 + lvl * 26) + Math.cos(a * 5 - t / 900) * (3 + lvl * 12);
      const x = cx + Math.cos(a) * (R + n), y = cy + Math.sin(a) * (R + n); a === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    }
    g.strokeStyle = col(0.5 - ring * 0.14); g.lineWidth = 1.6; g.stroke();
  }
  const core = g.createRadialGradient(cx - 30, cy - 40, 10, cx, cy, 140 + lvl * 30);
  core.addColorStop(0, 'rgba(255,255,255,.95)'); core.addColorStop(0.25, col(0.95)); core.addColorStop(1, 'rgba(10,12,20,.95)');
  g.beginPath(); g.arc(cx, cy, 128 + lvl * 26, 0, Math.PI * 2); g.fillStyle = core; g.fill();
  requestAnimationFrame(orb);
}
requestAnimationFrame(orb);

/* ---------------- simulated concurrent callers ---------------- */
const SCRIPTS = [
  { lang: 'en', name: 'Sim · Rahul', lines: ['Hi, my order still has not arrived.', 'The order ID is VN482917.', 'Okay, can I get a callback from a senior person?', 'Thanks, that is all.'] },
  { lang: 'hi', name: 'Sim · Priya', lines: ['मुझे अपने रिफंड के बारे में जानना है।', 'मेरा ऑर्डर नंबर VN550123 है।', 'ठीक है, धन्यवाद।'] },
  { lang: 'auto', name: 'Sim · Arjun', lines: ['I received a damaged product, this is really frustrating.', 'Yes please book a pickup.', 'Thank you.'] },
];
$('#spawnBtn').onclick = () => SCRIPTS.forEach((s, i) => setTimeout(() => simCaller(s), i * 700));
function simCaller(s) {
  const ws = new WebSocket(WS(`/ws/call?lang=${s.lang}&name=${encodeURIComponent(s.name)}`)); ws.binaryType = 'arraybuffer';
  let i = 0, timer = null;
  const next = () => { clearTimeout(timer); timer = setTimeout(() => { if (i < s.lines.length) ws.send(JSON.stringify({ type: 'text', text: s.lines[i++] })); else ws.close(); }, 1800 + Math.random() * 1200); };
  ws.onmessage = (e) => { if (typeof e.data !== 'string') return; const ev = JSON.parse(e.data); if (ev.type === 'state' && ev.state === 'listening') next(); };
}

/* ---------------- studio ---------------- */
function fillStudio() {
  const s = STATE.settings;
  $('#sCompany').value = s.company; $('#sGoal').value = s.goal; $('#sExtra').value = s.extra_instructions || '';
  $('#sGreetEn').value = s.greeting_en; $('#sGreetHi').value = s.greeting_hi;
  $('#sEndpoint').value = s.endpoint_ms; $('#epVal').textContent = s.endpoint_ms + 'ms';
  $('#sBarge').checked = s.barge_in; $('#sExpr').checked = s.expressive;
  $('#sEngine').value = s.engine; $('#sGemVoice').value = s.gemini_voice; markEngineAvailability();
  for (const [sel, v] of [['#sVoiceEn', s.voice_en], ['#sVoiceHi', s.voice_hi], ['#sKokEn', s.kokoro_en], ['#sKokHi', s.kokoro_hi]]) if (!$(sel).options.length) $(sel).innerHTML = `<option>${v}</option>`;
}
let voicesLoaded = false;
async function loadVoices() {
  if (voicesLoaded) return;
  try {
    const vs = await (await fetch('/api/voices')).json(); voicesLoaded = true;
    const opt = (v, cur) => `<option value="${v.id}" ${v.id === cur ? 'selected' : ''}>${v.id} · ${v.gender}</option>`;
    const edge = vs.filter((v) => v.engine === 'edge'), kok = vs.filter((v) => v.engine === 'kokoro');
    $('#sVoiceEn').innerHTML = edge.filter((v) => v.locale.startsWith('en')).map((v) => opt(v, STATE.settings.voice_en)).join('');
    $('#sVoiceHi').innerHTML = edge.filter((v) => v.locale === 'hi-IN' || v.id.includes('Multilingual')).map((v) => opt(v, STATE.settings.voice_hi)).join('');
    $('#sKokEn').innerHTML = kok.filter((v) => v.locale === 'en').map((v) => opt(v, STATE.settings.kokoro_en)).join('');
    $('#sKokHi').innerHTML = kok.filter((v) => v.locale === 'hi-IN').map((v) => opt(v, STATE.settings.kokoro_hi)).join('');
  } catch {}
}
$('#sEndpoint').oninput = (e) => ($('#epVal').textContent = e.target.value + 'ms');
$('#saveBtn').onclick = async () => {
  const body = { company: $('#sCompany').value, goal: $('#sGoal').value, extra_instructions: $('#sExtra').value,
    greeting_en: $('#sGreetEn').value, greeting_hi: $('#sGreetHi').value, voice_en: $('#sVoiceEn').value, voice_hi: $('#sVoiceHi').value, kokoro_en: $('#sKokEn').value, kokoro_hi: $('#sKokHi').value,
    endpoint_ms: +$('#sEndpoint').value, barge_in: $('#sBarge').checked, expressive: $('#sExpr').checked,
    engine: $('#sEngine').value, gemini_voice: $('#sGemVoice').value };
  await saveSettings(body); $('#saveBtn').textContent = 'Saved ✓'; setTimeout(() => ($('#saveBtn').textContent = 'Save & apply'), 1500);
};
async function saveSettings(body) {
  const r = await (await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  STATE.settings = r.settings; $('#agentCompany').textContent = `${r.settings.company} · customer support`;
}
let labLang = 'en';
$$('#labLang button').forEach((b) => b.onclick = () => { $$('#labLang button').forEach((x) => x.classList.toggle('on', x === b)); labLang = b.dataset.lang; });
$$('#tagBtns button').forEach((b) => b.onclick = () => { const ta = $('#labText'); const p = ta.selectionStart; ta.value = ta.value.slice(0, p) + b.dataset.t + ' ' + ta.value.slice(p); ta.focus(); });
$('#labPlay').onclick = async () => {
  $('#labPlay').textContent = 'Rendering…';
  const t0 = performance.now();
  const r = await fetch(`/api/preview?lang=${labLang}&text=${encodeURIComponent($('#labText').value)}`);
  const url = URL.createObjectURL(await r.blob()); $('#labAudio').src = url; $('#labAudio').play();
  $('#labPlay').textContent = `▶ Render & play  ·  ${Math.round(performance.now() - t0)}ms`;
};
const PRESETS = {
  hi: ['hi', '[sigh] मैं समझ सकती हूँ, ये सच में परेशान करने वाला है। [breath] चिंता मत कीजिए, मैं अभी आपका रिफंड प्रोसेस कर देती हूँ। [whisper]और हाँ, मैंने आपके लिए एक छोटा सा डिस्काउंट भी जोड़ दिया है।[/whisper]'],
  angry: ['en', "[sigh] I hear you, and honestly, you have every right to be upset. [pause] [breath] Here's what I'm doing right now: a replacement ships today, free of charge. [laugh] And I'll personally keep an eye on it."],
};
$$('.presets button').forEach((b) => b.onclick = () => { const [l, t] = PRESETS[b.dataset.p]; $('#labText').value = t; $$('#labLang button').find((x) => x.dataset.lang === l).click(); });

/* ---------------- providers ---------------- */
function renderProviders(list) {
  if (!list || !STATE.settings.llm_chain) return;
  const by = Object.fromEntries(list.map((p) => [`${p.kind}:${p.name}`, p]));
  const cols = [['llm', 'Language model', 'llm_chain'], ['stt', 'Speech-to-text', 'stt_chain'], ['tts', 'Text-to-speech', 'tts_chain']];
  $('#chains').innerHTML = cols.map(([k, title, key]) => `<div class="card"><div class="card-h"><h3>${title}</h3><span class="tag">failover order</span></div>
    ${STATE.settings[key].map((n, i) => { const p = by[`${k}:${n}`] || { state: 'unconfigured' };
      return `<div class="prov ${p.state === 'unconfigured' ? 'off' : ''}"><span class="n">${i + 1}</span>
        <div><b><span class="dot st-${p.state}"></span>${n}</b><small>${p.state}${p.p50 != null ? ` · p50 ${fmtMs(p.p50)}` : ''}${p.fails ? ` · ${p.fails} fail` : ''}</small></div>
        <div class="mv"><button data-k="${key}" data-i="${i}" data-d="-1">↑</button><button data-k="${key}" data-i="${i}" data-d="1">↓</button></div></div>`; }).join('')}
  </div>`).join('') + (() => {
    const p = by['s2s:gemini_live'] || { state: 'unconfigured' };
    return `<div class="card"><div class="card-h"><h3>Speech-to-speech</h3><span class="tag">native engine</span></div>
      <div class="prov ${p.state === 'unconfigured' ? 'off' : ''}"><span class="n">★</span>
      <div><b><span class="dot st-${p.state}"></span>Gemini Live</b><small>${p.state}${p.p50 != null ? ` · connect ${fmtMs(p.p50)}` : ''}${p.fails ? ` · ${p.fails} fail` : ''}</small></div><div></div></div>
      <p class="muted" style="font-size:12px;line-height:1.5;margin-top:6px">If it is down or drops mid-call, the call moves to the cascade engine with its conversation intact.</p></div>`;
  })();
}
$('#chains').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-k]'); if (!b) return;
  const arr = [...STATE.settings[b.dataset.k]], i = +b.dataset.i, j = i + +b.dataset.d;
  if (j < 0 || j >= arr.length) return; [arr[i], arr[j]] = [arr[j], arr[i]];
  await saveSettings({ [b.dataset.k]: arr }); renderProviders(STATE.providers);
});

/* ---------------- logs ---------------- */
async function loadLogs() {
  const logs = await (await fetch('/api/calls/log')).json();
  if (!logs.length) return;
  $('#logs').innerHTML = logs.map((l) => {
    const s = l.summary, d = new Date(s.started * 1000);
    return `<details class="card log"><summary><b>#${s.id}</b><span>${d.toLocaleTimeString()}</span><span>${esc(s.caller)} · ${s.channel} · ${s.lang.toUpperCase()}</span>
      <span>${s.turns} turns</span><b>${fmtMs(s.avg_e2e)}</b></summary>
      <div class="tr">${l.transcript.map((t) => `<p class="${t.role === 'user' ? 'u' : ''}"><em>${t.role === 'user' ? 'CALLER' : 'ARIA'}</em>${esc(t.text)}</p>`).join('')}</div></details>`;
  }).join('');
}

loadState(); monitor();
