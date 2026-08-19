# Demo UI Improvements Implementation Plan

Created: 2026-08-20
Agent: Claude Code
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

> Planning in progress...

## Summary

**Goal:** A judge can (1) see the per-stage latency breakdown as a proportional bar chart instead of a plain table, (2) flip an opt-in toggle to have one spoken question answered independently against both the fixed-size and semantic indices and see both results side by side, and (3) see the exact retrieved passages that grounded every answer, in both the normal and comparison views.

## Out of Scope

- Redeploying to the live EC2 instance. `.github/workflows/ci.yml` already runs a `deploy` job on every push to `main` once `test`/`shell`/`guard` pass, so merging this plan's branch to `main` redeploys automatically — no separate deploy task.
- Rendering the dataset's `is_selected` flag anywhere in the UI (see Global Constraints — it is a MSMARCO-XI offline relevance label, not a live signal for an arbitrary spoken query).
- Any change to `/api/ask`'s existing response fields beyond the additive `passages` field, or to its pre-existing behavior for stt/unsafe/off-topic/groundedness refusals.
- Text-to-speech, answer scoring/"winner" declaration between strategies, or protecting against overlapping/concurrent recordings (all pre-existing, unchanged behavior).

## Approach

**Chosen:** Extract the retrieval → off-topic guard → generation → groundedness guard sequence out of `ask()` (`app/main.py:97`) into a shared per-strategy helper, reused by both the existing `/api/ask` and a new `POST /api/compare` endpoint that runs both strategies concurrently from a single transcription.
**Why:** A spoken question must be transcribed once and compared on identical text — calling `/api/ask` twice from the frontend would transcribe the same audio twice (2x ElevenLabs latency/cost) and risks two different transcripts for what should be "the same question," which defeats the point of a side-by-side comparison. Running both strategies through one shared helper inside one backend request removes that risk entirely, at the cost of a moderate refactor of the already-tested `ask()` endpoint (mitigated below).

## Global Constraints

- No new external JS/CSS dependency (no charting library, no new CDN beyond the Google Fonts link already in `static/index.html`) — the latency chart and citation relevance bars are plain CSS/DOM.
- `RetrievedPassage.is_selected` (`app/schemas.py:21`) may be present in API JSON but must never be rendered in the DOM — it is the dataset's own offline relevance-label flag from MSMARCO-XI, not a live signal about the strength of a match for an arbitrary spoken query, and showing it would look like the system is claiming an authoritative "correct passage" match it hasn't actually made.
- Deployment is automatic via `.github/workflows/ci.yml`'s `deploy` job on push to `main` — no task in this plan performs or scripts a deploy.
- `uv run pytest -q`, `uv run ruff check .`, and `uv run ruff format --check .` must all stay clean (existing CI gate).

## Assumptions

- Whatever mechanism the original plan (`docs/plans/2026-08-18-voice-enabled-rag-pipeline.md`, TS-001/TS-002) used to feed a spoken question into the browser's microphone for E2E verification is still available and is reused as-is for this plan's E2E scenarios — Tasks 3, 4, and 5 depend on this for their own verification.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Extracting shared pipeline logic out of the already-tested `ask()` silently changes an existing `/api/ask` response field or stage-timing behavior | Medium | Medium | Task 1 requires every pre-existing assertion in `tests/test_api.py` to keep passing unmodified; run the full suite immediately after the refactor, before starting Task 2 |
| One strategy's branch raises inside the `/api/compare` fan-out and aborts the whole request even though the other strategy succeeded | Medium | Medium | Task 2 wraps each strategy's branch in its own try/except so a failing branch returns a refusal for that strategy only while its sibling's real result still comes back; proven by a test that mocks one branch to raise |

## File Structure

- `app/main.py` (modify) — extract the per-strategy pipeline into a reusable helper; add `passages` to `/api/ask`'s response; add the new `/api/compare` endpoint.
- `tests/test_api.py` (modify) — extend existing `/api/ask` assertions with the new `passages` field; add a new test class covering `/api/compare`.
- `static/app.js` (modify) — bar-chart rendering (pure builder + single-mode wrapper), citation rendering (pure builder + single-mode wrapper), compare-mode state and wiring.
- `static/index.html` (modify) — CSS for the latency bars, citation cards, compare toggle, and two-column compare grid; the new markup elements those need.

## Progress Tracking

- [x] Task 1: Add `passages` to `/api/ask` and extract the shared per-strategy pipeline helper
- [x] Task 2: Add `POST /api/compare` — both strategies from one recording, concurrently
- [x] Task 3: Render the latency breakdown as a proportional bar chart
- [x] Task 4: Add the opt-in "Compare strategies" toggle and two-column view
- [x] Task 5: Display retrieved passages (citations) under every answer

## Implementation Tasks

### Task 1: Add `passages` to `/api/ask` and extract the shared per-strategy pipeline helper

**Objective:** Extract the retrieval → off-topic guard → generation → groundedness guard sequence currently inline in `ask()` (`app/main.py:97-213`) into a standalone async helper `_run_strategy_pipeline(transcript: str, strategy: str) -> dict[str, Any]` that returns the same per-strategy shape `/api/ask` already returns from the retrieval stage onward (`answer`, `refusal_reason`, `stages_ms` for `retrieval`/`guardrail_off_topic`/`generation`/`guardrail_groundedness`, `top_score`), plus a new `passages` key. `ask()` keeps its own STT + unsafe-input-guard logic, then delegates to the helper for the requested strategy and merges its `stages_ms` with the stt/unsafe stages it already tracks, so `/api/ask`'s existing shape is unchanged except for the added field. Task 2 reuses this helper for `/api/compare`.

**Files:**

- Modify: `app/main.py`
- Modify: `tests/test_api.py`

**Key Decisions / Notes:**

- `passages` value is `[p.model_dump() for p in retrieval.passages]` (list of `{text, source_passage, is_selected, score}`, from `RetrievedPassage` at `app/schemas.py:18-22`) — computed once inside `_run_strategy_pipeline` right after `retrieve()` returns, so it is available to attach to every branch's result (success, off-topic refusal, generation failure, groundedness refusal) without recomputing.
- For refusals that happen *before* retrieval runs (STT failure, empty transcript, unsafe input — the `_refusal()` helper at `app/main.py:69-93`), set `"passages": []` since no retrieval occurred; update `_refusal()`'s return dict accordingly.
- `top_score` computation (`retrieval.passages[0].score if retrieval.passages else 0.0`, currently `app/main.py:154`) moves into the helper, unchanged in behavior.
- Do not change the *order* of operations (retrieve → off-topic → generate → groundedness) or which stage's timer wraps which call — only the code's location moves, not its sequencing.

**Definition of Done:**

- [x] `POST /api/ask` on a clean query returns a `passages` list whose items each have `text` (str), `source_passage` (str), `score` (float), and `is_selected` (bool), matching the passages `retrieve()` returned.
- [x] `POST /api/ask` refused before retrieval runs (unsafe input, STT failure, empty transcript) returns `"passages": []`.
- [x] Every pre-existing assertion in `tests/test_api.py` (all tests present before this task) still passes unmodified.
- [x] Verify: `uv run pytest tests/test_api.py -q`

### Task 2: Add `POST /api/compare` — both strategies from one recording, concurrently

**Objective:** Add a `POST /api/compare` endpoint that transcribes the posted audio once, runs the shared unsafe-input guard once, and then runs Task 1's `_run_strategy_pipeline` for both `"fixed_size"` and `"semantic"` concurrently via `asyncio.gather`, so one spoken question yields two independently-retrieved-and-generated answers without re-transcribing.

**Files:**

- Modify: `app/main.py`
- Modify: `tests/test_api.py`

**Key Decisions / Notes:**

- Response shape on success: `{"transcript": str, "latency_ms": float, "shared_stages_ms": {"stt": float, "guardrail_unsafe": float}, "refusal_reason": null, "results": {"fixed_size": {...same per-strategy shape Task 1's helper returns...}, "semantic": {...}}}`.
- Response shape when STT or the unsafe-input guard refuses (before the fan-out): `{"transcript": str | null, "latency_ms": float, "shared_stages_ms": {...whatever stages ran...}, "refusal_reason": str, "results": null}` — deliberately `null`, not an empty object, so the frontend can do a single `data.results == null` check for "nothing to compare."
- Add a wrapper `async def _run_strategy_pipeline_safe(transcript, strategy)` that calls `_run_strategy_pipeline` inside `try`/`except Exception`, converting any unexpected exception into `{"answer": None, "refusal_reason": "internal error", "stages_ms": {}, "top_score": None, "passages": []}` for that strategy only. `/api/compare` fans out via `asyncio.gather(*(_run_strategy_pipeline_safe(transcript, s) for s in _STRATEGIES))` so one strategy's unexpected failure never 500s the whole request or blocks the sibling strategy's real result.
- Import `asyncio` at the top of `app/main.py`.
- `Trivial:` does not apply — this is a new endpoint with new branching logic.

**Definition of Done:**

- [x] `POST /api/compare` on a clean query returns `results.fixed_size` and `results.semantic`, each with its own `answer`, `stages_ms`, `top_score`, and `passages`; `retrieve()` is called exactly twice, once per strategy, both times with the same `query` (the shared transcript).
- [x] `POST /api/compare` with unsafe input returns `refusal_reason` set, `results` is `null`, and `retrieve()`/`generate_answer()` are never called.
- [x] When one strategy's branch is mocked to raise an unexpected exception, the response is still HTTP 200 with the sibling strategy's real result populated and the failing strategy's result carrying `"refusal_reason": "internal error"`.
- [x] Verify: `uv run pytest tests/test_api.py -q`

### Task 3: Render the latency breakdown as a proportional bar chart

**Objective:** Replace the plain `<table>` latency rendering in `renderLatency` (`static/app.js:157-170`) with a proportional horizontal bar chart, factored so the bar-building logic is reusable by Task 4's compare columns without duplication.

**Files:**

- Modify: `static/app.js`
- Modify: `static/index.html`

**Key Decisions / Notes:**

- Add a pure function `buildLatencyChartHtml(stagesMs, totalMs)` returning an HTML string: one row per entry in `stagesMs` (label from the existing `STAGE_LABELS` map at `static/app.js:148-155`, falling back to the raw key), a bar whose width is `Math.max(2, (ms / totalMs) * 100)` percent (the `Math.max(2, ...)` floor keeps a `0.0` ms guardrail stage visible instead of a zero-width bar), and the exact `ms.toFixed(1)` value as text next to the bar so a small bar is never mistaken for a larger one. A bold total row (reusing today's `.total` styling) stays at the bottom.
- `renderLatency(data)` becomes a thin wrapper: `$("latency").innerHTML = buildLatencyChartHtml(data.stages_ms || {}, data.latency_ms)`. Behavior when `stages_ms` is empty is unchanged (renders nothing), matching the current `rows || total` guard.
- CSS: add `.latency-bar-row`, `.latency-bar-track`, `.latency-bar-fill` (using `var(--accent)`, `var(--glass)`, `var(--glass-border)` — no new colors) with `transition: width 0.4s ease` consistent with the existing `reveal`/`spin-pulse` animation conventions in `static/index.html`; remove the now-unused `table`/`td`/`.total` selectors only if nothing else references them (check Task 5 first — it does not reuse them).

**Definition of Done:**

- [x] After a completed query, the Latency card shows one bar per stage present in `stages_ms`, each labeled with its stage name and exact ms value, sized proportionally to its share of the total.
- [x] A stage reporting `0.0` ms (e.g. the regex-based unsafe-input guardrail) still renders a visible, labeled bar rather than disappearing.
- [x] Verify: run `uv run uvicorn app.main:app --port 8000` locally and use browser automation to complete a query, confirming proportional labeled bars render for every stage in the response's `stages_ms` (see TS-002).

### Task 4: Add the opt-in "Compare strategies" toggle and two-column view

**Objective:** Add a checkbox toggle beside the existing strategy pill selector; when checked, a spoken question is sent to `/api/compare` (Task 2) instead of `/api/ask`, and the result renders as two side-by-side columns (one per strategy) instead of the single-column view — reusing Task 3's `buildLatencyChartHtml` and Task 5's `buildCitationsHtml` per column rather than duplicating either.

**Files:**

- Modify: `static/index.html`
- Modify: `static/app.js`

**Key Decisions / Notes:**

- HTML: add `<label class="compare-toggle"><input type="checkbox" id="compareToggle"><span>Compare strategies</span></label>` inside `.header` (`static/index.html:270-274`), after `#strategyChoices`. Add a new sibling to `#outputContainer` (`static/index.html:288`): `<div class="output-container" id="compareContainer">` containing `<p id="compareTranscript">`, `<p id="compareRefusal" class="refusal"></p>` (hidden unless populated), and a `.compare-grid` with two `.card.compare-column` blocks, `id="compareCol_fixed_size"` and `id="compareCol_semantic"`, each with an `<h3>` strategy label, `<p class="answer"></p>`, `<div class="latency"></div>`, and `<div class="citations"></div>`.
- CSS: `.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }` collapsing to `grid-template-columns: 1fr` under a mobile breakpoint (`@media (max-width: 640px)`), per the project's mobile-first convention.
- JS: `let compareMode = false;` toggled by a `change` listener on `#compareToggle`; when `true`, `$("strategyChoices")` gets `opacity: 0.4; pointer-events: none` inline-styled (strategy choice is moot in compare mode) and vice versa when unchecked.
- `stopRecording()` (`static/app.js:186-219`) branches on `compareMode`: `true` → `fetch("/api/compare", {method:"POST", headers:{...}, body: audioBytes})` (no `?strategy=` query param) then `displayCompareResult(await response.json())`; `false` → unchanged existing `/api/ask` call + `displayResult`.
- New `displayCompareResult(data)`: sets `$("compareTranscript").textContent`; if `data.results == null`, shows `data.refusal_reason` in `#compareRefusal` and clears both columns; otherwise hides `#compareRefusal` and, for each of `["fixed_size", "semantic"]`, sets that column's `.answer` text (or a refusal message if that strategy's own `refusal_reason` is set), `.latency` innerHTML via `buildLatencyChartHtml(result.stages_ms, result.stages_ms ? Object.values(result.stages_ms).reduce((a,b)=>a+b,0) : 0)`, and `.citations` innerHTML via `buildCitationsHtml(result.passages)`. Toggles `$("outputContainer")`/`$("compareContainer")` visibility (`style.display`) based on `compareMode` at the same point `setState("done")`/`setState("error")` is called, so the existing state-machine CSS classes keep driving opacity/animation on whichever container is visible.
- `Trivial:` does not apply — new state, new endpoint call, new rendering path.

**Definition of Done:**

- [x] With the toggle off, the app behaves exactly as before: strategy radios active, single-column result via `/api/ask`.
- [x] With the toggle on, the strategy radios are visually inert and a single spoken question produces two visible result columns labeled "fixed-size" and "semantic," each with its own answer and latency chart, from one `/api/compare` call (not two `/api/ask` calls).
- [x] If the shared pre-retrieval guardrails refuse (e.g. unsafe input) in compare mode, one shared refusal message renders instead of two empty/broken columns.
- [x] Verify: run the app locally and use browser automation to toggle compare mode on, submit a spoken query, and confirm two independently-populated columns render (see TS-003), then toggle it back off and confirm single-mode behavior is unchanged (see TS-004).

### Task 5: Display retrieved passages (citations) under every answer

**Objective:** Add a Citations display, in both the single-strategy view and each compare-mode column, listing the passages that grounded the answer — reusing the numbering convention `app/generation.py`'s `_build_prompt` already uses ("Passage N") so the number a user sees matches what the model actually saw.

**Files:**

- Modify: `static/app.js`
- Modify: `static/index.html`

**Key Decisions / Notes:**

- Add a pure function `buildCitationsHtml(passages)`: if `passages` is empty/undefined, returns a muted `"No passages retrieved"` line; otherwise one row per passage, 1-indexed as `"Passage N"`, showing `text` truncated to ~220 characters with a trailing `…` (full text in a `title` attribute for hover) and a relevance bar for `score` (reusing the same bar-track/bar-fill CSS classes Task 3 introduced, width `Math.min(100, score * 100)` percent). Never reads or renders `is_selected` (Global Constraints).
- `renderCitations(passages, containerEl)` is a thin wrapper: `containerEl.innerHTML = buildCitationsHtml(passages)`.
- Single-mode: add a new `<div class="card"><h3>Citations</h3><div id="citations"></div></div>` in `#outputContainer` (`static/index.html:288-303`), right after the Latency card; call `renderCitations(data.passages, $("citations"))` from `displayResult` (`static/app.js:172-184`).
- Compare-mode: Task 4's `displayCompareResult` calls `buildCitationsHtml(result.passages)` directly per column (see Task 4).

**Definition of Done:**

- [x] After any answered (non-refused) query in single mode, the Citations card lists every passage from the response's `passages` array with its passage number, truncated text, and a relevance bar.
- [x] `is_selected` is present in the API payload (Task 1) but appears nowhere in the rendered DOM (verified by inspecting the rendered Citations markup).
- [x] A response with an empty `passages` array (e.g. a pre-retrieval refusal) renders the "No passages retrieved" placeholder instead of an empty or broken card.
- [x] Verify: run the app locally and use browser automation to submit a spoken query and confirm the Citations card lists the passages returned in `/api/ask`'s JSON body (see TS-001).

## E2E Test Scenarios

### TS-001: Citations render after a normal single-strategy answer
**Priority:** Critical
**Preconditions:** Live app loaded in browser; microphone/fake-audio input available (per Assumptions); vector indices built and loaded
**Mapped Tasks:** Task 1, Task 5

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to the app, leave "Compare strategies" unchecked | Single strategy pill selector visible, no compare grid |
| 2 | Record a question whose answer exists in the ingested corpus, release | Transcript and answer appear as before |
| 3 | Read the Citations card | One or more passage rows appear, each with a passage number, snippet text, and a relevance bar; no `is_selected` value is visible anywhere |

### TS-002: Latency chart renders proportional bars, including a zero-ms stage
**Priority:** High
**Preconditions:** Same as TS-001
**Mapped Tasks:** Task 3

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Complete a normal query (as in TS-001) | Result renders |
| 2 | Read the Latency card | A labeled, proportionally-sized bar appears for every stage in the response's `stages_ms` (stt, guardrail_unsafe, retrieval, guardrail_off_topic, generation, guardrail_groundedness where applicable); the guardrail stages that measure ~0ms still show a visible labeled bar, not a blank row |

### TS-003: Compare mode answers one question from both strategies at once
**Priority:** Critical
**Preconditions:** Same as TS-001
**Mapped Tasks:** Task 2, Task 4, Task 5

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Check "Compare strategies" | Strategy pill selector becomes visually inert; single-column output container is replaced by a two-column layout |
| 2 | Record one in-corpus question, release | A single request is made (network panel: one call to `/api/compare`, not two calls to `/api/ask`) |
| 3 | Read both columns | Both a "fixed-size" and a "semantic" column show their own answer, latency chart, and citations, derived from the same transcript shown once above the grid |

### TS-004: Turning compare mode back off restores the original single-strategy flow
**Priority:** Medium
**Preconditions:** Compare mode was just used (state from TS-003)
**Mapped Tasks:** Task 4

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Uncheck "Compare strategies" | Strategy pill selector becomes active again; compare grid is hidden |
| 2 | Record a question, release | Exactly one call to `/api/ask?strategy=...` is made; single-column result renders as it did before any of this plan's changes |

### TS-005: Unsafe input in compare mode shows one shared refusal, not two columns
**Priority:** High
**Preconditions:** Compare mode checked
**Mapped Tasks:** Task 2, Task 4

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Record a question matching one of the unsafe patterns in `app/guardrails.py` (`_UNSAFE_PATTERNS`) | One `/api/compare` request is made |
| 2 | Read the result | A single shared refusal message is shown once; neither the "fixed-size" nor the "semantic" column shows an empty or broken card |

## E2E Results

**Live-target probe:** Tier 1 succeeded — the plan's own local server (`uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`) was started and its `/health` endpoint polled to 200 before any browser step; Tiers 2-4 not attempted.

**Driver:** playwright-cli (Claude Code Chrome / Chrome DevTools MCP not available in this environment), session `demoui-verify`, `browser-automation.md` tier 3.

**Mic substitute:** this sandboxed environment has no real microphone. Consistent with `docs/plans/2026-08-18-voice-enabled-rag-pipeline.md`'s own prior verification (which confirmed the record button reachable via Playwright but did not drive an actual click-and-record interaction, relying on a direct `/api/ask` call for full-pipeline proof instead), every scenario below was driven through a **real click on the record button with a real, audible MediaStream** (an oscillator's output routed through `AudioContext.createMediaStreamDestination()`, injected as `navigator.mediaDevices.getUserMedia`'s return value) — the full capture pipeline (`ScriptProcessorNode` → downsample → PCM encode → `fetch`) runs unmodified and for real; only the HTTP response at `/api/ask` / `/api/compare` was intercepted via `playwright-cli route` with a JSON body matching the exact shape the (separately, unit-tested) backend returns.

| Scenario | Priority | Result | Fix Attempts | Notes |
|----------|----------|--------|--------------|-------|
| TS-001   | Critical | PASS   | 0            | Citations card lists both mocked passages with passage number, truncated text (verified the >220-char passage truncates with `...`), and a relevance bar; `document.getElementById("citations").innerHTML` contains neither `is_selected` nor `true` |
| TS-002   | High     | PASS   | 1            | Fixed: the fixed 9rem label column truncated "Retrieval (embed + FAISS)" / "Guardrail · groundedness" with CSS ellipsis on first render — widened to 11.5rem and switched to wrapping. After the fix, all 6 stage bars render proportionally, including both 0.0ms guardrail stages at the 2%-minimum width with their exact "0.0 ms" label |
| TS-003   | Critical | PASS   | 0            | `playwright-cli requests` confirmed exactly one `POST /api/compare` per recording (no `/api/ask` calls while compare mode was on); both columns rendered independently from the same mocked response — fixed-size answered, semantic showed its own off-topic refusal with its own top_score, proving the two branches are not conflated |
| TS-004   | Medium   | PASS   | 0            | After unchecking the toggle, `getComputedStyle(strategyChoices).opacity` returned to `"1"` and `pointerEvents` to `"auto"`; the next recording's network request (via `playwright-cli requests`) was `POST /api/ask?strategy=fixed_size`, not `/api/compare` |
| TS-005   | High     | PASS   | 0            | Mocked a shared unsafe-input refusal (`results: null`); single refusal message rendered in `#compareRefusal`, `.compare-grid` set to `display: none` — no empty column cards |

**Code identity verification (Step 4c):** restarted the local server on the finished code and called it directly (not through the browser mocks): `POST /api/ask` on real benchmark audio now returns a `"passages": []` field that did not exist before this plan, and `POST /api/compare` (previously a 404) now responds with the new `{transcript, latency_ms, shared_stages_ms, refusal_reason, results}` shape — proof the running instance is serving the new code.

**Full external round-trip:** attempted for real (no mocking) against `data/benchmark_audio/question_0.pcm` with `.env` credentials loaded. Blocked by a pre-existing local-machine limitation, not this diff: this Python 3.14 framework build's SSL store cannot verify ElevenLabs' certificate (`ssl.SSLCertVerificationError: unable to get local issuer certificate`), caught cleanly by the existing `run_stage` harness as a graceful `"could not process audio"` refusal rather than a crash. This is a local dev-environment gap (the live EC2 deployment already proves the real STT/generation round-trip in `docs/LATENCY_REPORT.md`), not a defect in this plan's code.

**Design Notes:** `impeccable detect` run against the live rendered DOM — 12 advisory findings, all pre-existing patterns this plan did not introduce or only extended consistently with them: 10 layout-property-transition warnings (`transition: width`, the same pattern the pre-existing `.mic-button`/`.strategy-group` rules already used), 1 overused-font (Inter, chosen before this plan), 1 flat-type-hierarchy, 1 dark-glow (the pre-existing recording-state red glow). Non-blocking per the detector's advisory contract.
