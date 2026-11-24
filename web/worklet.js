// Mic capture: downsample to 16 kHz mono, emit 20 ms Int16 frames + RMS.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.out = new Int16Array(320);
    this.n = 0;
    this.acc = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // simple box-filter decimation
      this.acc += ch[i];
      this.pos += 1;
      if (this.pos >= this.ratio) {
        const v = this.acc / this.pos;
        this.acc = 0; this.pos -= this.ratio;
        this.out[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
        if (this.n === 320) {
          let s = 0;
          for (let k = 0; k < 320; k++) { const f = this.out[k] / 32768; s += f * f; }
          this.port.postMessage({ pcm: this.out.slice(0).buffer, rms: Math.sqrt(s / 320) }, []);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('capture', Capture);
