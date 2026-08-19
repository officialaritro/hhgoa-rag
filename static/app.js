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
let recording = false;
let selectedStrategy = "fixed_size";

const $ = (id) => document.getElementById(id);

// getUserMedia is a secure-context-only API: over plain HTTP
// navigator.mediaDevices is undefined and recording fails with no visible
// error. Detect it up front rather than throwing on first click.
function micAvailable() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function downsampleBuffer(buffer, sampleRate, outRate) {
  if (outRate === sampleRate) return buffer;
  const sampleRateRatio = sampleRate / outRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

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
  if (recording || !micAvailable()) return;
  recording = true;
  capturedChunks = [];
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    recording = false;
    $("status").textContent = "Microphone access denied.";
    return;
  }
  
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const resampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
    capturedChunks.push(floatTo16BitPCM(resampled));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
  $("status").textContent = "Recording…";
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

// Stage keys are whatever app/main.py reports, so the table follows the
// pipeline automatically if stages are added or renamed server-side.
const STAGE_LABELS = {
  stt: "Speech-to-text",
  guardrail_unsafe: "Guardrail · unsafe input",
  retrieval: "Retrieval (embed + FAISS)",
  guardrail_off_topic: "Guardrail · off-topic",
  generation: "Generation (Claude)",
  guardrail_groundedness: "Guardrail · groundedness",
};

function renderLatency(data) {
  const stages = data.stages_ms || {};
  const rows = Object.entries(stages)
    .map(
      ([key, ms]) =>
        `<tr><td>${STAGE_LABELS[key] || key}</td><td class="n">${ms.toFixed(1)} ms</td></tr>`,
    )
    .join("");
  const total =
    data.latency_ms != null
      ? `<tr class="total"><td>Total</td><td class="n">${data.latency_ms.toFixed(1)} ms</td></tr>`
      : "";
  $("latency").innerHTML = rows || total ? `<table>${rows}${total}</table>` : "";
}

function displayResult(data) {
  $("transcript").textContent = data.transcript || "";
  const answerEl = $("answer");
  if (data.answer) {
    answerEl.textContent = data.answer;
    answerEl.className = "";
  } else {
    // A refusal is a successful outcome, not an error -- show why. The top
    // retrieval score is shown alongside the reason so a refusal can be told
    // apart from a miscalibrated threshold without shell access to the box.
    const score =
      data.top_score != null ? ` (top match ${data.top_score.toFixed(3)})` : "";
    answerEl.textContent = data.refusal_reason
      ? `Cannot answer — ${data.refusal_reason}${score}`
      : "";
    answerEl.className = "refusal";
  }
  renderLatency(data);
}

async function stopRecording() {
  if (!recording || !processorNode) {
    recording = false;
    return;
  }
  recording = false;
  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
  $("status").textContent = "Processing…";

  const audioBytes = concatenateChunks(capturedChunks);
  try {
    const response = await fetch(
      `/api/ask?strategy=${encodeURIComponent(selectedStrategy)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: audioBytes,
      },
    );
    displayResult(await response.json());
  } catch (err) {
    $("answer").textContent = `Request failed: ${err.message}`;
    $("answer").className = "refusal";
  }
  $("status").textContent = "";
}

// Both chunking strategies are indexed; let the user pick which one answers.
// Hard-coding one would waste the second index entirely.
async function loadStrategies() {
  try {
    const { strategies, default: def } = await (
      await fetch("/api/strategies")
    ).json();
    selectedStrategy = def;
    $("strategyChoices").innerHTML = strategies
      .map(
        (s) =>
          `<label style="margin-right:1.2rem">
             <input type="radio" name="strategy" value="${s}" ${s === def ? "checked" : ""}>
             ${s.replace("_", "-")}
           </label>`,
      )
      .join("");
    for (const input of document.querySelectorAll('input[name="strategy"]')) {
      input.addEventListener("change", (e) => {
        selectedStrategy = e.target.value;
      });
    }
  } catch {
    $("strategyChoices").textContent = "unavailable (service not ready)";
  }
}

const recordButton = $("recordBtn");
if (!micAvailable()) {
  recordButton.disabled = true;
  const warning = $("micWarning");
  warning.hidden = false;
  warning.textContent = window.isSecureContext
    ? "This browser does not expose a microphone API."
    : "Microphone unavailable: this page must be served over HTTPS.";
} else {
  // Pointer events, not mousedown/mouseup: mouseup fires on the element the
  // cursor is over, so releasing off the button would leave it recording
  // forever. Pointer capture keeps the release bound to the button, and the
  // same handlers cover touch on a phone.
  recordButton.addEventListener("pointerdown", (e) => {
    recordButton.setPointerCapture(e.pointerId);
    if (!audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx({ sampleRate: 16000 });
    }
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }
    startRecording();
  });
  recordButton.addEventListener("pointerup", stopRecording);
  recordButton.addEventListener("pointercancel", stopRecording);
}

loadStrategies();
