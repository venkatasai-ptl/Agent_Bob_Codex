class PCMCollectorProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = Math.round(sampleRate * 0.02); // 20ms frames
    this.buffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channelData = input[0];
    const combined = new Float32Array(this.buffer.length + channelData.length);
    combined.set(this.buffer, 0);
    combined.set(channelData, this.buffer.length);
    let offset = 0;
    while (offset + this.frameSize <= combined.length) {
      const frame = combined.subarray(offset, offset + this.frameSize);
      const buffer = new ArrayBuffer(frame.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < frame.length; i++) {
        let s = Math.max(-1, Math.min(1, frame[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      this.port.postMessage(buffer);
      offset += this.frameSize;
    }
    this.buffer = combined.subarray(offset);
    return true;
  }
}

registerProcessor('pcm_collector', PCMCollectorProcessor);
