// Vaani dashboard: live call client + operations console.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const WS = (p) => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${p}`;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtMs = (v) => (v == null ? '–' : (v / 1000).toFixed(2) + 's');   // one format everywhere
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
let STATE = { settings: {}, providers: [], keys: {} };

/* ---------------- navigation ---------------- */
const TITLES = {
  console: ['Live console', 'Talk to the agent exactly as your customers will'],
  calls: ['Active calls', 'Every conversation in progress. Step in or end any call'],
  studio: ['Agent studio', 'Persona, knowledge, voices and delivery'],
  providers: ['Providers', 'Failover order, health and latency for each stage'],
  logs: ['Call logs', 'Transcript and latency of every finished call'],
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
  $('#agentCompany').textContent = `${STATE.settings.company} customer support`;
  const premium = ['groq', 'openai', 'gemini', 'anthropic', 'openrouter'].some((k) => STATE.keys[k]);
  if (!premium) {
    const b = $('#banner'); b.classList.remove('hidden');
    b.innerHTML = '<b>Free fallback models in use.</b> Replies take 1 to 3 seconds. Add <code>GROQ_API_KEY</code> to <code>.env</code> and restart for sub-second replies';
  }
}
function renderKpis(s) {
  if (!s) return;
  $('#kActive').textContent = s.active; $('#navActive').textContent = s.active || '';
  $('#kP50').textContent = fmtMs(s.p50); $('#kP90').textContent = fmtMs(s.p90); $('#kTotal').textContent = s.total;
}
const callLines = {};
function monitor() {
  const ws = new WebSocket(WS('/ws/monitor'));
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'tick') { renderKpis(ev.stats); renderCalls(ev.calls); renderProviders(ev.providers); }
    if (ev.type === 'transcript') callLines[ev.call_id] = { role: ev.role, text: ev.text };
  };
  ws.onclose = () => setTimeout(monitor, 1500);
}

/* ---------------- active calls grid ---------------- */
const STATE_LABEL = { idle: 'Ready', listening: 'Listening', hearing: 'Hearing', thinking: 'Thinking', speaking: 'Speaking', connecting: 'Connecting', ended: 'Ended' };
function renderCalls(calls) {
  const g = $('#callGrid');
  if (!calls || !calls.length) { g.innerHTML = '<div class="empty">No live calls</div>'; return; }
  if (document.activeElement && document.activeElement.closest('.cc')) return; // operator is typing
  g.innerHTML = calls.map((c) => {
    const line = callLines[c.id] || { role: 'agent', text: c.last_text };
    return `<div class="card cc">
      <div class="cc-h"><div class="who">${esc(c.caller)} <span>· ${c.channel} · ${c.lang.toUpperCase()}</span></div>
        <span class="state-dot" data-s="${c.state}"><i></i>${STATE_LABEL[c.state] || c.state}</span></div>
      <div class="cc-line ${line.role === 'agent' ? 'agent' : ''}">${esc(line.text || '')}</div>
      <div class="cc-stats"><div><span>Duration</span><b>${mmss(c.duration)}</b></div><div><span>Turns</span><b>${c.turns}</b></div>
        <div><span>Voice-to-voice</span><b>${fmtMs(c.avg_e2e)}</b></div></div>
      <div class="cc-act"><input placeholder="Make Aria say" data-id="${c.id}" />
        <button class="btn small" data-say="${c.id}">Say</button><button class="btn small" data-hang="${c.id}">End</button></div>
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
let call = null, lang = 'auto', engine = 'cascade', callStart = 0;
$$('#engineSeg button').forEach((b) => b.onclick = () => {
  if (call) return;
  $$('#engineSeg button').forEach((x) => x.classList.toggle('on', x === b)); engine = b.dataset.engine;
});
function markEngineAvailability() {
  const ok = STATE.keys && STATE.keys.gemini, b = $('#engineSeg [data-engine=gemini_live]');
  b.classList.toggle('na', !ok);
  b.querySelector('small').textContent = ok ? 'Gemini Live' : 'Needs a Gemini key, uses cascade';
}
$$('#langSeg button').forEach((b) => b.onclick = () => {
  if (call) return;
  $$('#langSeg button').forEach((x) => x.classList.toggle('on', x === b)); lang = b.dataset.lang;
});
let uiState = 'idle', micLevel = 0, outLevel = 0;
function setState(s) {
  uiState = s; const p = $('#statePill'); p.dataset.s = s; p.querySelector('span').textContent = STATE_LABEL[s] || 'Ready';
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
      case 'hello': $('#callIdTag').textContent = `${ev.call_id} · ${ev.engine === 'gemini_live' ? 'native' : 'cascade'}`; break;
      case 'engine': if (ev.failover) addSys('Native engine unavailable, continuing on cascade'); $('#callIdTag').textContent = `${ev.call_id} · cascade`; break;
      case 'state': setState(ev.state); break;
      case 'clear': this.clear(); break;
      case 'interrupted': addSys('Caller interrupted, Aria stopped to listen'); break;
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
  const c = new Call(); call = c; callStart = performance.now(); samples.length = 0;
  $('#callBtn').classList.replace('primary', 'danger'); $('#callLbl').textContent = 'End call'; $('#muteBtn').disabled = false;
  setState('connecting');
  try { await c.start(); } catch (e) { addSys('Microphone unavailable: ' + e.message); stopCall(); }
}
function stopCall(fromClose) {
  const c = call; call = null;
  if (c && !fromClose) c.hangup();
  $('#callBtn').classList.replace('danger', 'primary'); $('#callLbl').textContent = 'Start call';
  $('#muteBtn').disabled = true; $('#muteBtn').textContent = 'Mute'; setState('idle'); $('#callIdTag').textContent = 'No call';
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
  partialEl.innerHTML = `<small>Aria</small>${esc(text)}`; $('#msgs').scrollTop = 1e9;
}
function addSys(t) { const m = document.createElement('div'); m.className = 'msg sys'; m.textContent = t; $('#msgs').appendChild(m); $('#msgs').scrollTop = 1e9; }
function showMetrics(m) {
  const max = Math.max(1500, m.e2e_ms || 0, (m.stt_ms || 0) + (m.llm_ttft_ms || 0));
  const w = (v) => `scaleX(${v ? Math.min(1, v / max) : 0})`;
  $('#bStt').style.transform = w(m.stt_ms); $('#tStt').textContent = fmtMs(m.stt_ms);
  $('#bLlm').style.transform = w(m.llm_ttft_ms); $('#tLlm').textContent = fmtMs(m.llm_ttft_ms);
  $('#bE2e').style.transform = w(m.e2e_ms); $('#tE2e').textContent = fmtMs(m.e2e_ms);
  $('#e2eTag').textContent = m.e2e_ms ? `Turn ${m.turn}` : '–';
  $('#provs').innerHTML = `<span>STT<b>${m.stt || 'typed'}</b></span><span>Model<b>${m.llm || '–'}</b></span><span>Voice<b>${m.tts || '–'}</b></span>`;
}

/* ---------------- conversation waveform ----------------
   The real audio of the call: Aria's voice above the line in the accent, the caller below in neutral. */
const samples = [];
const ACCENT = '#C8FF2E', CALLER = '#a1a1a6', GRID = 'rgba(237,237,237,0.08)';
setInterval(() => {
  if (!call) return;
  samples.push({ a: Math.min(1, outLevel * 7), c: Math.min(1, micLevel * 9) });
  if (samples.length > 600) samples.shift();
  $('#clock').textContent = mmss((performance.now() - callStart) / 1000);
}, 50);
function drawWave() {
  const cv = $('#wave'), g = cv.getContext('2d');
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth * dpr, H = cv.clientHeight * dpr;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  g.clearRect(0, 0, W, H);
  const mid = H / 2, step = 6 * dpr, bw = 3 * dpr, n = Math.floor(W / step), maxH = H * 0.42;
  g.fillStyle = GRID; g.fillRect(0, mid - dpr / 2, W, dpr);
  for (let x = W - step; x > 0; x -= step * 10) g.fillRect(x, mid - 4 * dpr, dpr, 8 * dpr);
  const start = Math.max(0, samples.length - n);
  for (let i = start; i < samples.length; i++) {
    const s = samples[i], x = W - (samples.length - i) * step;
    const ha = Math.max(dpr, s.a * maxH), hc = Math.max(dpr, s.c * maxH);
    g.fillStyle = ACCENT; g.beginPath(); g.roundRect(x, mid - ha - dpr, bw, ha, bw / 2); g.fill();
    g.fillStyle = CALLER; g.beginPath(); g.roundRect(x, mid + dpr, bw, hc, bw / 2); g.fill();
  }
  requestAnimationFrame(drawWave);
}
requestAnimationFrame(drawWave);

/* ---------------- test callers (run in parallel against the real pipeline) ---------------- */
const SCRIPTS = [
  { lang: 'en', name: 'Rahul', lines: ['Hi, my order still has not arrived.', 'The order number is 482917.', 'Okay, can I get a callback from a senior person?', 'Thanks, that is all.'] },
  { lang: 'hi', name: 'Priya', lines: ['मुझे अपने रिफंड के बारे में जानना है।', 'मेरा ऑर्डर नंबर 550123 है।', 'ठीक है, धन्यवाद।'] },
  { lang: 'auto', name: 'Arjun', lines: ['I received a damaged product, this is really frustrating.', 'Yes please book a pickup.', 'Thank you.'] },
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
  await saveSettings(body); $('#saveBtn').textContent = 'Saved'; setTimeout(() => ($('#saveBtn').textContent = 'Save'), 1500);
};
async function saveSettings(body) {
  const r = await (await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  STATE.settings = r.settings; $('#agentCompany').textContent = `${r.settings.company} customer support`;
}
let labLang = 'en';
$$('#labLang button').forEach((b) => b.onclick = () => { $$('#labLang button').forEach((x) => x.classList.toggle('on', x === b)); labLang = b.dataset.lang; });
$$('#tagBtns button').forEach((b) => b.onclick = () => { const ta = $('#labText'); const p = ta.selectionStart; ta.value = ta.value.slice(0, p) + b.dataset.t + ' ' + ta.value.slice(p); ta.focus(); });
$('#labPlay').onclick = async () => {
  const lbl = $('#labPlay span'); lbl.textContent = 'Rendering'; $('#labMs').textContent = '';
  const t0 = performance.now();
  const r = await fetch(`/api/preview?lang=${labLang}&text=${encodeURIComponent($('#labText').value)}`);
  const url = URL.createObjectURL(await r.blob()); $('#labAudio').src = url; $('#labAudio').play();
  lbl.textContent = 'Render and play'; $('#labMs').textContent = `Rendered in ${fmtMs(performance.now() - t0)}`;
};
const PRESETS = {
  hi: ['hi', '[sigh] मैं समझ सकती हूँ, ये सच में परेशान करने वाला है। [breath] चिंता मत कीजिए, मैं अभी आपका रिफंड प्रोसेस कर देती हूँ। [whisper]और हाँ, मैंने आपके लिए एक छोटा सा डिस्काउंट भी जोड़ दिया है।[/whisper]'],
  angry: ['en', "[sigh] I hear you, and honestly, you have every right to be upset. [pause] [breath] Here's what I'm doing right now: a replacement ships today at no cost. [laugh] And I'll keep an eye on it myself."],
};
$$('.presets button').forEach((b) => b.onclick = () => { const [l, t] = PRESETS[b.dataset.p]; $('#labText').value = t; $$('#labLang button').find((x) => x.dataset.lang === l).click(); });

/* ---------------- providers ---------------- */
const STATE_TEXT = { healthy: 'Healthy', cooling: 'Cooling down', unconfigured: 'No key' };
const NAMES = { groq: 'Groq', openai: 'OpenAI', gemini: 'Gemini', anthropic: 'Anthropic', openrouter: 'OpenRouter',
  pollinations: 'Pollinations (free)', scripted: 'Offline answers', deepgram: 'Deepgram', sarvam: 'Sarvam',
  google: 'Google (free)', whisper: 'Whisper (local)', elevenlabs: 'ElevenLabs', cartesia: 'Cartesia', nvidia: 'NVIDIA Magpie',
  kokoro: 'Kokoro (local)', edge: 'Microsoft Edge (free)', sapi: 'Windows voice' };
function renderProviders(list) {
  if (!list || !STATE.settings.llm_chain) return;
  const by = Object.fromEntries(list.map((p) => [`${p.kind}:${p.name}`, p]));
  const meta = (p, lat = 'p50') => `${STATE_TEXT[p.state] || p.state}${p.p50 != null ? ` · ${lat} ${fmtMs(p.p50)}` : ''}${p.fails ? ` · ${p.fails} failed` : ''}`;
  const cols = [['llm', 'Language model', 'llm_chain'], ['stt', 'Speech-to-text', 'stt_chain'], ['tts', 'Voice', 'tts_chain']];
  $('#chains').innerHTML = cols.map(([k, title, key]) => `<div class="card"><div class="card-h"><h3>${title}</h3><span class="meta">In order</span></div>
    ${STATE.settings[key].map((n, i) => { const p = by[`${k}:${n}`] || { state: 'unconfigured' };
      return `<div class="prov ${p.state === 'unconfigured' ? 'off' : ''}"><span class="n">${i + 1}</span>
        <div><b><span class="dot st-${p.state}"></span>${NAMES[n] || n}</b><small>${meta(p)}</small></div>
        <div class="mv"><button data-k="${key}" data-i="${i}" data-d="-1" aria-label="Move up">↑</button><button data-k="${key}" data-i="${i}" data-d="1" aria-label="Move down">↓</button></div></div>`; }).join('')}
  </div>`).join('') + (() => {
    const p = by['s2s:gemini_live'] || { state: 'unconfigured' };
    return `<div class="card"><div class="card-h"><h3>Speech-to-speech</h3><span class="meta">Native engine</span></div>
      <div class="prov ${p.state === 'unconfigured' ? 'off' : ''}"><span class="n">1</span>
      <div><b><span class="dot st-${p.state}"></span>Gemini Live</b><small>${meta(p, 'connect')}</small></div><div></div></div>
      <p class="note">If it is down or drops mid-call, the call moves to the cascade engine with its conversation intact</p></div>`;
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
    return `<details class="card log"><summary><b>${s.id}</b><span>${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span><span>${esc(s.caller)} · ${s.channel} · ${s.lang.toUpperCase()}</span>
      <span>${s.turns} turns</span><b>${fmtMs(s.avg_e2e)}</b></summary>
      <div class="tr">${s.recording ? `<audio class="rec" controls preload="none" src="/recordings/${encodeURIComponent(s.recording)}"></audio>` : ''}${l.transcript.map((t) => `<p class="${t.role === 'user' ? 'u' : ''}"><em>${t.role === 'user' ? 'Caller' : 'Aria'}</em><span>${esc(t.text)}</span></p>`).join('')}</div></details>`;
  }).join('');
}

loadState(); monitor();
