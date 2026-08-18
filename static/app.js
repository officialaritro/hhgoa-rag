// Microphone capture for the voice RAG demo.
//
// Captures raw 16kHz PCM (not MediaRecorder's compressed WebM output) to
// match app/stt.py's ElevenLabs Realtime call, which requires
// AudioFormat.PCM_16000 -- see plan Task 5 / Global Constraints.
//
// SHORTCUT: uses the deprecated ScriptProcessorNode instead of an
// AudioWorklet module for simplicity. Still functional in current Chrome.
// Upgrade trigger: if ScriptProcessorNode support is ever removed from the
// browsers used for the demo.

let audioContext;
let sourceNode;
let processorNode;
let mediaStream;
let capturedChunks = [];

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

async function startRecording() {
  capturedChunks = [];
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext({ sampleRate: 16000 });
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    capturedChunks.push(floatTo16BitPCM(input));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
  document.getElementById("status").textContent = "Recording...";
}

function concatenateChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const combined = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }
  return combined;
}

function displayResult(data) {
  document.getElementById("transcript").textContent = data.transcript || "";
  document.getElementById("answer").textContent = data.answer || data.refusal_reason || "";
  document.getElementById("latency").textContent = data.latency_ms
    ? `${data.latency_ms.toFixed(1)} ms`
    : "";
}

async function stopRecording() {
  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
  document.getElementById("status").textContent = "Processing...";

  const audioBytes = concatenateChunks(capturedChunks);
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: audioBytes,
  });
  const data = await response.json();
  displayResult(data);
  document.getElementById("status").textContent = "";
}

const recordButton = document.getElementById("recordBtn");
recordButton.addEventListener("mousedown", startRecording);
recordButton.addEventListener("mouseup", stopRecording);
