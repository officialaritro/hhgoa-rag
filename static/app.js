let audioContext;
let sourceNode;
let processorNode;
let analyserNode;
let mediaStream;
let capturedChunks = [];
let recording = false;
let selectedStrategy = "fixed_size";
let animationId;
let compareMode = false;

const $ = (id) => document.getElementById(id);

function setState(state) {
  document.body.className = `state-${state}`;
}

function parseMarkdown(text) {
  // Defense-in-depth: the generation prompt (app/generation.py) asks for
  // plain spoken-style prose, but a model can still slip in a stray
  // heading/bullet/asterisk -- render it properly instead of leaking literal
  // `#`/`-`/`**` characters into the answer, and escape first since this
  // text becomes innerHTML.
  if (!text) return "";
  const lines = escapeHtml(text).split("\n");
  const htmlParts = [];
  let listItems = [];
  let paragraphLines = [];

  const flushList = () => {
    if (!listItems.length) return;
    htmlParts.push(`<ul>${listItems.map((item) => `<li>${item}</li>`).join("")}</ul>`);
    listItems = [];
  };
  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    htmlParts.push(`<p>${paragraphLines.join(" ")}</p>`);
    paragraphLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    const bolded = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const headingMatch = bolded.match(/^#{1,6}\s+(.*)/);
    const bulletMatch = bolded.match(/^[-*]\s+(.*)/);

    if (headingMatch) {
      flushParagraph();
      flushList();
      htmlParts.push(`<h4 class="answer-heading">${headingMatch[1]}</h4>`);
    } else if (bulletMatch) {
      flushParagraph();
      listItems.push(bulletMatch[1]);
    } else {
      flushList();
      paragraphLines.push(bolded);
    }
  }
  flushParagraph();
  flushList();
  return htmlParts.join("");
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

/** The live waveform, drawn as a horizontal wave across the horizon the sun
 *  sits on -- a sea line rather than the ring this used to draw. The ring
 *  was sized for a square canvas and would be clipped by the wide, short
 *  one the sun composition uses. */
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
  ctx.lineWidth = 2.5;
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#ff0080";
  ctx.beginPath();

  const mid = height / 2;
  for (let i = 0; i < bufferLength; i++) {
    const v = (dataArray[i] - 128) / 128;
    const x = (i / (bufferLength - 1)) * width;
    const y = mid + v * (height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
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
    $("statusText").textContent = "MICROPHONE BLOCKED";
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
  
  $("statusText").textContent = "LISTENING";
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
  rerank: "Rerank (cross-encoder)",
  guardrail_off_topic: "Guardrail · off-topic",
  generation: "Generation (Claude)",
  guardrail_groundedness: "Guardrail · groundedness",
};

/* ====================================================================
   The sun is the pipeline.
   One ray per stage, radiating from a half-disc sitting on the horizon.
   Ray length encodes that stage's share of total latency; the exact
   numbers stay in the latency card, so the sun carries the shape of the
   run and never becomes the only place a number is legible.

   Geometry matches the viewBox in index.html: 0 0 400 190, horizon at
   y=170, disc centred at (200,170) r=40.
   ==================================================================== */
const SUN_CX = 200;
const SUN_CY = 170;
const RAY_INNER = 52;
// A stage that measures 0.0 ms still gets a visible stub, so the corona
// never has a gap that reads as a missing stage. Everything above the
// floor is strictly proportional: len = RAY_MIN + share * RAY_SPAN, so
// (len - RAY_MIN) / share is the same constant for every ray.
const RAY_MIN = 15;
const RAY_SPAN = 68;
// Idle rays sit mid-arc so the corona reads as a sun before any query has
// run. At RAY_MIN alone they rendered as tick marks with a band of dead
// air above them.
const RAY_IDLE_FACTOR = 0.62;
// Rays fan across 160 degrees rather than a full 180 so the outermost
// pair never lies flat along the horizon.
const RAY_ARC_START = 10;
const RAY_ARC_SWEEP = 160;

let rayRevealTimers = [];

function rayAngle(index, count) {
  if (count <= 1) return Math.PI / 2;
  const deg = RAY_ARC_START + (index * RAY_ARC_SWEEP) / (count - 1);
  return (deg * Math.PI) / 180;
}

function buildRays(stageKeys, initialFactor = 0) {
  const group = document.getElementById("rayGroup");
  if (!group) return;
  const reach = RAY_INNER + RAY_MIN + RAY_SPAN * initialFactor;
  group.innerHTML = stageKeys
    .map((key, i) => {
      const a = rayAngle(i, stageKeys.length);
      const cos = Math.cos(a);
      const sin = Math.sin(a);
      const x1 = (SUN_CX + RAY_INNER * cos).toFixed(2);
      const y1 = (SUN_CY - RAY_INNER * sin).toFixed(2);
      const x2 = (SUN_CX + reach * cos).toFixed(2);
      const y2 = (SUN_CY - reach * sin).toFixed(2);
      return `<line class="ray" data-stage="${key}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"><title>${
        STAGE_LABELS[key] || key
      }</title></line>`;
    })
    .join("");
}

function clearRayReveal() {
  for (const t of rayRevealTimers) clearTimeout(t);
  rayRevealTimers = [];
}

function resetRays() {
  clearRayReveal();
  buildRays(Object.keys(STAGE_LABELS), RAY_IDLE_FACTOR);
}

/** Light the rays in stage order, sized by each stage's share of the total.
 *
 * The reveal is sequenced rather than instant because the order carries
 * real information -- it is the pipeline's execution order, straight off
 * the response's own key order. It deliberately does NOT claim live
 * per-stage progress: the API returns every stage at once, so animating
 * during the request would be inventing timing we never measured. The
 * in-flight state is an explicit indeterminate sweep instead.
 */
function revealRays(stagesMs) {
  clearRayReveal();
  const entries = Object.entries(stagesMs || {});
  if (!entries.length) return resetRays();

  buildRays(entries.map(([k]) => k));
  const total = entries.reduce((sum, [, ms]) => sum + ms, 0);
  const group = document.getElementById("rayGroup");
  if (!group) return;
  const rays = [...group.querySelectorAll(".ray")];

  rays.forEach((ray, i) => {
    const ms = entries[i] ? entries[i][1] : 0;
    const share = total > 0 ? ms / total : 0;
    const len = RAY_INNER + RAY_MIN + share * RAY_SPAN;
    const a = rayAngle(i, rays.length);
    rayRevealTimers.push(
      setTimeout(() => {
        ray.setAttribute("x2", (SUN_CX + len * Math.cos(a)).toFixed(3));
        ray.setAttribute("y2", (SUN_CY - len * Math.sin(a)).toFixed(3));
        ray.classList.add("is-lit");
        ray.dataset.ms = ms.toFixed(1);
        // Recorded at full precision, not re-derived from the rounded x2/y2:
        // on a stage worth 0.4% of the run, 2dp coordinate rounding moves the
        // implied proportionality constant by more than a unit.
        ray.dataset.share = share.toFixed(6);
        ray.dataset.len = (len - RAY_INNER).toFixed(4);
      }, i * 70)
    );
  });
}

/** Compare mode runs its strategies concurrently, so the wall clock the
 *  user actually waited is the shared stages plus the slowest branch at
 *  each stage -- not any single strategy's column. */
function mergeCompareStages(data) {
  const merged = { ...(data.shared_stages_ms || {}) };
  for (const result of Object.values(data.results || {})) {
    for (const [key, ms] of Object.entries(result.stages_ms || {})) {
      merged[key] = Math.max(merged[key] || 0, ms);
    }
  }
  return merged;
}


// Which check refused, keyed on the server's `refused_by`. The task asks the
// system to show that it knows when NOT to answer, so a refusal is a result
// worth presenting properly rather than an error string.
const GUARD_LABELS = {
  off_topic: "Off-topic guard",
  groundedness: "Groundedness guard",
  insufficient_context: "Not in the retrieved passages",
  unsafe_input: "Unsafe-input guard",
  stt: "Speech-to-text",
  generation: "Generation",
  internal_error: "Internal error",
};

function scoreNote(data) {
  // A similarity means nothing without the bar it had to clear, so both are
  // shown -- on success as well as refusal. This is what makes the per-index
  // calibration visible instead of a claim in a document.
  if (data.top_score == null) return "";
  if (data.offtopic_threshold == null) return `top match ${data.top_score.toFixed(3)}`;

  // Enough decimals to actually separate the two. At three, a score of 0.5637
  // against a threshold of 0.5641 rendered as "top match 0.564 · below
  // threshold 0.564" -- two identical numbers and a verdict that reads as a
  // bug. Widen only when rounding collapses them, so the common case stays
  // readable.
  let digits = 3;
  while (
    digits < 6 &&
    data.top_score.toFixed(digits) === data.offtopic_threshold.toFixed(digits)
  ) {
    digits += 1;
  }
  const score = data.top_score.toFixed(digits);
  const bar = data.offtopic_threshold.toFixed(digits);
  const verb = data.top_score >= data.offtopic_threshold ? "cleared" : "below";
  return `top match ${score} · ${verb} threshold ${bar}`;
}

function refusalHtml(data) {
  const guard = GUARD_LABELS[data.refused_by] || "Refused";
  const detail = data.refusal_reason || "";
  const note = scoreNote(data);
  return `<span class="refusal-guard">${guard}</span>
     <span class="refusal-detail">${detail}</span>
     ${note ? `<span class="refusal-score">${note}</span>` : ""}`;
}

function buildLatencyChartHtml(stagesMs, totalMs) {
  const entries = Object.entries(stagesMs || {});
  if (!entries.length && totalMs == null) return "";

  const rows = entries
    .map(([key, ms]) => {
      // A 2%-minimum bar keeps a 0.0ms guardrail stage visible instead of a
      // zero-width sliver -- the exact ms label next to it is what keeps that
      // sliver from being mistaken for a larger real stage.
      const width = totalMs > 0 ? Math.max(2, (ms / totalMs) * 100) : 2;
      return `
        <div class="latency-bar-row">
          <div class="latency-bar-topline">
            <span class="latency-bar-label">${STAGE_LABELS[key] || key}</span>
            <span class="latency-bar-value">${ms.toFixed(1)} ms</span>
          </div>
          <div class="latency-bar-track">
            <div class="latency-bar-fill" style="width: ${width}%"></div>
          </div>
        </div>`;
    })
    .join("");
  const total =
    totalMs != null
      ? `<div class="latency-bar-row total"><div class="latency-bar-topline"><span class="latency-bar-label">Total</span><span class="latency-bar-value">${totalMs.toFixed(1)} ms</span></div></div>`
      : "";
  return rows || total ? `${rows}${total}` : "";
}

function renderLatency(data) {
  $("latency").innerHTML = buildLatencyChartHtml(data.stages_ms || {}, data.latency_ms);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function buildCitationsHtml(passages) {
  if (!passages || !passages.length) {
    return '<p class="citations-empty">No passages retrieved.</p>';
  }
  return passages
    .map((p, i) => {
      const truncated = p.text.length > 220 ? `${p.text.slice(0, 220)}...` : p.text;
      const relevance = Math.min(100, Math.max(0, p.score * 100));
      return `
        <div class="citation-row" title="${escapeHtml(p.text)}">
          <div class="citation-header">
            <span class="citation-index">Passage ${i + 1}</span>
            <div class="latency-bar-track citation-relevance-track">
              <div class="latency-bar-fill" style="width: ${relevance}%"></div>
            </div>
          </div>
          <p class="citation-text">${escapeHtml(truncated)}</p>
        </div>`;
    })
    .join("");
}

function renderCitations(passages, containerEl) {
  containerEl.innerHTML = buildCitationsHtml(passages);
}

function displayResult(data) {
  $("transcript").textContent = data.transcript ? `"${data.transcript}"` : "";
  const answerEl = $("answer");
  if (data.answer) {
    answerEl.innerHTML = parseMarkdown(data.answer);
    answerEl.className = "";
  } else {
    answerEl.innerHTML = data.refusal_reason ? refusalHtml(data) : "";
    answerEl.className = "refusal refusal-block";
  }
  const note = data.answer ? scoreNote(data) : "";
  $("answerScore").textContent = note;
  $("answerScore").style.display = note ? "block" : "none";
  renderLatency(data);
  renderCitations(data.passages, $("citations"));
  revealRays(data.stages_ms);
  $("outputContainer").style.display = "grid";
  $("compareContainer").style.display = "none";
}

function displayCompareResult(data) {
  $("compareTranscript").textContent = data.transcript ? `"${data.transcript}"` : "";
  const refusalEl = $("compareRefusal");
  const grid = document.querySelector("#compareContainer .compare-grid");

  if (data.results == null) {
    refusalEl.textContent = `Cannot answer — ${data.refusal_reason}`;
    refusalEl.style.display = "block";
    grid.style.display = "none";
  } else {
    refusalEl.style.display = "none";
    grid.style.display = "";

    // Build a column per strategy the server actually compared. This used to
    // loop over a hardcoded ["fixed_size", "semantic"], which threw the moment
    // the compared set changed -- data.results[strategy] came back undefined.
    const compared = Object.keys(data.results);
    grid.innerHTML = compared
      .map((s) => {
        const axis = (strategyDetails[s] || {}).axis;
        const badge = axis ? `<span class="axis-badge">${axis}</span>` : "";
        return `<div class="card compare-column" id="compareCol_${s}">
             <h2>${prettyStrategy(s)}${badge}</h2>
             <div class="answer" aria-live="polite"></div>
             <div class="latency"></div>
             <div class="citations"></div>
           </div>`;
      })
      .join("");

    for (const strategy of compared) {
      const result = data.results[strategy];
      const col = $(`compareCol_${strategy}`);
      if (!result || !col) continue;
      const answerEl = col.querySelector(".answer");
      if (result.answer) {
        answerEl.innerHTML = parseMarkdown(result.answer);
        answerEl.className = "answer";
        const colNote = scoreNote(result);
        if (colNote) {
          answerEl.insertAdjacentHTML(
            "afterend",
            `<div class="refusal-score">${colNote}</div>`
          );
        }
      } else {
        answerEl.innerHTML = refusalHtml(result);
        answerEl.className = "answer refusal refusal-block";
      }
      const totalMs = Object.values(result.stages_ms || {}).reduce((a, b) => a + b, 0);
      col.querySelector(".latency").innerHTML = buildLatencyChartHtml(result.stages_ms, totalMs);
      col.querySelector(".citations").innerHTML = buildCitationsHtml(result.passages);
    }
  }
  revealRays(mergeCompareStages(data));
  $("outputContainer").style.display = "none";
  $("compareContainer").style.display = "grid";
}

function teardownAudioGraph() {
  if (animationId) cancelAnimationFrame(animationId);
  processorNode.disconnect();
  analyserNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
}

function cancelRecording() {
  if (!recording || !processorNode) return;
  recording = false;
  teardownAudioGraph();
  capturedChunks = [];
  resetRays();
  setState("idle");
  $("statusText").textContent = "HOLD TO SPEAK";
}

async function stopRecording() {
  if (!recording || !processorNode) {
    recording = false;
    setState("idle");
    return;
  }
  recording = false;
  // Show the destination container (dimmed via the state-processing CSS
  // rule) before the fetch resolves -- otherwise the very first query of a
  // fresh page load has no inline display set yet, so the "dimmed skeleton"
  // effect silently doesn't render on that first request.
  if (compareMode) {
    $("compareContainer").style.display = "grid";
    $("outputContainer").style.display = "none";
  } else {
    $("outputContainer").style.display = "grid";
    $("compareContainer").style.display = "none";
  }
  setState("processing");
  teardownAudioGraph();
  $("statusText").textContent = "RETRIEVING";

  const audioBytes = concatenateChunks(capturedChunks);
  const endpoint = compareMode
    ? "/api/compare"
    : `/api/ask?strategy=${encodeURIComponent(selectedStrategy)}`;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: audioBytes,
    });
    const data = await response.json();
    if (compareMode) {
      displayCompareResult(data);
    } else {
      displayResult(data);
    }
    setState("done");
  } catch (err) {
    if (compareMode) {
      $("compareTranscript").textContent = "";
      $("compareRefusal").textContent = `Request failed: ${err.message}`;
      $("compareRefusal").style.display = "block";
      document.querySelector("#compareContainer .compare-grid").style.display = "none";
      $("outputContainer").style.display = "none";
      $("compareContainer").style.display = "grid";
    } else {
      $("answer").textContent = `Request failed: ${err.message}`;
      $("answer").className = "refusal";
      $("outputContainer").style.display = "grid";
      $("compareContainer").style.display = "none";
    }
    setState("error");
  }
}

// Populated from /api/strategies so the UI never restates what the registry
// already knows. Ten strategy names mean nothing on their own; the server
// ships a description and an axis for each.
let strategyDetails = {};

function prettyStrategy(name) {
  // replaceAll, not replace: `replace` with a string pattern substitutes only
  // the first match, so `query_aware_heldout` came out as `query-aware_heldout`.
  return name.replaceAll("_", "-");
}

async function loadStrategies() {
  try {
    const { strategies, default: def, details } = await (
      await fetch("/api/strategies")
    ).json();
    selectedStrategy = def;
    strategyDetails = details || {};
    // Grouped by axis, not a flat list. The axes -- what is split, what unit is
    // returned, what the embedding is enriched with, what is aggregated -- are
    // the actual argument for a "vast" chunking strategy; ten undifferentiated
    // pills hide it.
    const byAxis = new Map();
    for (const s of strategies) {
      const axis = (strategyDetails[s] || {}).axis || "other";
      if (!byAxis.has(axis)) byAxis.set(axis, []);
      byAxis.get(axis).push(s);
    }
    $("strategyChoices").innerHTML = [...byAxis.entries()]
      .map(([axis, members]) => {
        const pills = members
          .map((s) => {
            const detail = strategyDetails[s] || {};
            // The description is the registry's own, surfaced as a tooltip
            // rather than duplicated here.
            const tip = detail.description
              ? ` title="${detail.description.replace(/"/g, "&quot;")}"`
              : "";
            return `<input type="radio" id="strat_${s}" name="strategy" value="${s}" ${s === def ? "checked" : ""}>
               <label for="strat_${s}"${tip}>${prettyStrategy(s)}</label>`;
          })
          .join("");
        return `<div class="axis"><span class="axis-name">${axis}</span>${pills}</div>`;
      })
      .join("");
    for (const input of document.querySelectorAll('input[name="strategy"]')) {
      input.addEventListener("change", (e) => {
        selectedStrategy = e.target.value;
      });
    }
  } catch {
    $("strategyChoices").innerHTML = '<span class="axis-name">UNAVAILABLE</span>';
  }
}

function primeAudioContextAndStart() {
  if (!audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx({ sampleRate: 16000 });
  }
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }
  startRecording();
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
    primeAudioContextAndStart();
  });
  recordButton.addEventListener("pointerup", stopRecording);
  recordButton.addEventListener("pointercancel", stopRecording);

  // Hold-space-to-speak, Wispr Flow-style: works from anywhere on the page,
  // not just while the mic button itself is focused.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    // preventDefault on every Space keydown, not just the first -- held keys
    // fire repeated keydown events at the OS repeat rate, and skipping this
    // for the `recording`-guarded repeats (as an early-return above it would)
    // left every repeat free to page-scroll even though the first didn't.
    e.preventDefault();
    if (e.repeat || recording) return;
    primeAudioContextAndStart();
  });
  document.addEventListener("keyup", (e) => {
    if (e.code !== "Space") return;
    e.preventDefault();
    stopRecording();
  });
  // Esc drops an in-progress recording without submitting it, so the user
  // can re-record fresh instead of waiting out a bad take.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Escape" || !recording) return;
    e.preventDefault();
    cancelRecording();
  });
}

$("compareToggle").addEventListener("change", (e) => {
  compareMode = e.target.checked;
  $("strategyChoices").style.opacity = compareMode ? "0.4" : "1";
  $("strategyChoices").style.pointerEvents = compareMode ? "none" : "";
});

resetRays();
loadStrategies();
