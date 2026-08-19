let audioContext;
let sourceNode;
let processorNode;
let analyserNode;
let mediaStream;
let capturedChunks = [];
let recording = false;
let selectedStrategy = "fixed_size";
let animationId;

const $ = (id) => document.getElementById(id);

function setState(state) {
  document.body.className = `state-${state}`;
}

function parseMarkdown(text) {
  if (!text) return "";
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

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

function drawVisualizer() {
  const canvas = $("visualizer");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  
  if (!analyserNode || !recording) {
    ctx.clearRect(0, 0, width, height);
    return;
  }
  
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyserNode.getByteTimeDomainData(dataArray);
  
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
  ctx.beginPath();
  
  const centerX = width / 2;
  const centerY = height / 2;
  const baseRadius = 60;
  
  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0; 
    const mappedV = 1.0 + (v - 1.0) * 1.5; 
    const radius = baseRadius + (mappedV * 15);
    
    const angle = (i / bufferLength) * 2 * Math.PI;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  
  ctx.closePath();
  ctx.stroke();
  
  animationId = requestAnimationFrame(drawVisualizer);
}

async function startRecording() {
  if (recording || !micAvailable()) return;
  recording = true;
  capturedChunks = [];
  setState("recording");
  
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    recording = false;
    setState("error");
    $("statusText").textContent = "Microphone access denied.";
    return;
  }
  
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 128;
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const resampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
    capturedChunks.push(floatTo16BitPCM(resampled));
  };

  sourceNode.connect(analyserNode);
  analyserNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
  
  $("statusText").textContent = "Recording...";
  drawVisualizer();
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
  $("transcript").textContent = data.transcript ? `"${data.transcript}"` : "";
  const answerEl = $("answer");
  if (data.answer) {
    answerEl.innerHTML = parseMarkdown(data.answer);
    answerEl.className = "";
  } else {
    const score = data.top_score != null ? ` (top match ${data.top_score.toFixed(3)})` : "";
    answerEl.textContent = data.refusal_reason ? `Cannot answer — ${data.refusal_reason}${score}` : "";
    answerEl.className = "refusal";
  }
  renderLatency(data);
}

async function stopRecording() {
  if (!recording || !processorNode) {
    recording = false;
    setState("idle");
    return;
  }
  recording = false;
  setState("processing");
  if (animationId) cancelAnimationFrame(animationId);
  
  processorNode.disconnect();
  analyserNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
  $("statusText").textContent = "Processing...";

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
    setState("done");
  } catch (err) {
    $("answer").textContent = `Request failed: ${err.message}`;
    $("answer").className = "refusal";
    setState("error");
  }
}

async function loadStrategies() {
  try {
    const { strategies, default: def } = await (await fetch("/api/strategies")).json();
    selectedStrategy = def;
    $("strategyChoices").innerHTML = strategies
      .map(
        (s) =>
          `<input type="radio" id="strat_${s}" name="strategy" value="${s}" ${s === def ? "checked" : ""}>
           <label for="strat_${s}">${s.replace("_", "-")}</label>`
      )
      .join("");
    for (const input of document.querySelectorAll('input[name="strategy"]')) {
      input.addEventListener("change", (e) => {
        selectedStrategy = e.target.value;
      });
    }
  } catch {
    $("strategyChoices").innerHTML = '<span style="font-size: 0.8rem; padding: 6px 16px; color: var(--text-muted);">unavailable</span>';
  }
}

const recordButton = $("recordBtn");
if (!micAvailable()) {
  recordButton.disabled = true;
  setState("error");
  $("statusText").textContent = window.isSecureContext
    ? "No microphone API."
    : "Must be HTTPS.";
} else {
  recordButton.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
  
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
