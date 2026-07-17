# Tiered ("leveled") inference — design plan

Status: **plan only, not implemented.** Captures the agreed design for progressive
refinement STT so we can build it in stages.

## Goal

Dictation where text appears fast and *improves in place*: a rough local draft the
instant you pause, then it's overwritten by a high-quality large-v3 result a moment
later. "Speculative transcription" — show something now, refine as better models catch up.

## Why (grounded in measurements)

From the latency/quality work on the GTX 1080 (see git history, `bench_latency.py`):

- **Word-by-word streaming is laggy anyway.** LocalAgreement confirms a word a
  *median 2.5s* after it's spoken (turbo), up to 5–7s — it needs two agreeing
  re-decodes, and tail words wait for the pause. So streaming's "live" feel doesn't
  actually buy low latency.
- **Pause-based decode is both simpler and lower-latency:** a finished utterance
  decodes once (no re-decode), appearing ~1.5–2s after you stop (large-v3 on GPU),
  ~0.3s (tiny locally).
- **large-v3 beats turbo on code-switching** (хост vs хвост, Linear vs линии,
  Review Tasks) but its full decoder is too slow to re-decode every chunk live
  (borderline even at min_chunk 2.0). Decode-once sidesteps that entirely.

Conclusion: build tiers on **pause-based, per-utterance** decoding, not streaming.

## Locked decisions

- **Display: terminal TUI**, in-place redraw. Each utterance is a slot that updates draft→final.
- **Draft timing: on-pause** (local tiny decodes the finished utterance ~0.3s after
  you stop). No during-speech streaming — simpler, lower latency, trivial alignment.
- **Core = 2 tiers** (local draft + remote large-v3 final). The turbo "middle tier"
  is deferred (it lands ~2.5s, near the large-v3 final, so it's mostly redundant
  except on long utterances).

## Tiers

| tier | where / model | lands | role |
|------|---------------|-------|------|
| 1 draft | laptop CPU, `tiny`/`base` | ~0.3s after pause | instant feedback, works offline |
| 3 final | GPU box, `large-v3` (int8) | ~1.5–2s after pause | authoritative text, overwrites the draft |
| 2 refine *(deferred)* | GPU box, `turbo` streaming | mid-utterance | only for long sentences, before the pause |

Tier 1 is deliberately **local** so the first feedback never depends on the network
and works with the GPU box offline (degrades to draft-only).

## Architecture

Single laptop-side orchestrator process (`tiered.py`) owning all I/O:

```
 mic ─ffmpeg─▶ VAD segmenter ──utterance(id, pcm, span)──┬─▶ local worker (tiny)  ──draft(id,text)──┐
 (laptop)                    (one segmentation,          │                                          ├─▶ TUI (slots keyed by id)
                             ids + time spans)           └─▶ ssh→ remote worker (large-v3) ─final(id,text)─┘
```

- **Segment once, on the laptop.** webrtcvad (reuse the loop from `transcribe.py`)
  cuts utterances; each gets an incrementing `id` and a `[start,end]` audio span.
  Both tiers decode the *same* segment, so alignment is just the `id` — no fuzzy
  text matching.
- **Local worker**: `WhisperModel(tiny/base, cpu)`, decodes each utterance, emits
  `draft(id, text)`.
- **Remote worker**: a persistent `ssh` subprocess running a decode server on the GPU
  (model stays loaded — never reload per utterance). Laptop writes framed PCM to its
  stdin, reads results from its stdout.
- **TUI**: consumes `draft`/`final` events, updates the slot for `id`, redraws.

### Remote transport (framing)

Persistent connection so large-v3 loads once:

- Laptop → GPU (binary stdin): per utterance `[u32 id][u32 n_bytes][n_bytes PCM s16le]`.
- GPU → laptop (text stdout): one JSON line per result `{"id": N, "text": "..."}`.

Reuses the `gpu-server.sh` env (LD_LIBRARY_PATH for CUDA). New remote mode
`./gpu-server.sh --serve-utterances large-v3`.

### TUI model

- **Committed history**: once an utterance has its final *and* all earlier utterances
  are final, print it as a permanent scrollback line and drop it from the live region.
- **Live region**: the last few utterances still awaiting a final — redrawn in place
  (ANSI cursor-up + clear-line). Bounded height, so no whole-screen redraw.
- Per slot: show final if present, else draft (dim), else `… listening`. A small tag
  marks tier (e.g. `·` draft, ` ` final) and network state.
- Handle terminal width (truncate/wrap the live region); degrade gracefully if not a TTY (append-only).

## Concurrency

Threads in `tiered.py`, communicating via queues, all events tagged with `id`:

1. **capture+VAD** (main read loop) → utterance queue
2. **local decode worker** → event queue (`draft`)
3. **remote writer** (utterance queue → ssh stdin framing)
4. **remote reader** (ssh stdout JSON → event queue `final`)
5. **display** (event queue → TUI redraw)

Display orders by `id` regardless of arrival order.

## Files

- `tiered.py` — laptop orchestrator (capture, VAD, local worker, remote client, TUI).
- `utterance_server.py` — GPU: load model once, read framed PCM, emit JSON results.
- `tui.py` — mutable terminal display (or inline in `tiered.py` if small).
- reuse: VAD/segmentation + `read_exact` + `start_ffmpeg` from `transcribe.py`
  (refactor the segmenter into a shared helper), recording/`RECORD` from `stream-remote.sh`.
- `gpu-server.sh` — add `--serve-utterances` mode.

## Milestones

- **M0** Refactor VAD utterance segmentation out of `transcribe.py` into a reusable
  generator yielding `(id, pcm, span)`.
- **M1** Tier 1 + TUI only (local tiny, no remote): capture → segment → draft → mutable
  display. Proves the display model end-to-end.
- **M2** Remote `utterance_server.py` + `gpu-server.sh --serve-utterances`; `tiered.py`
  sends utterances, receives finals, overwrites drafts. **Full 2-tier system.**
- **M3** Recording + per-tier transcript logs (reuse `RECORD`); latency instrumentation
  (measure draft-latency and final-latency per utterance, like `bench_latency.py`).
- **M4** *(optional)* Tier 2 turbo mid-utterance refinement for long utterances; apply
  a post-correction map at the final tier for the residual acoustic errors (гид→гит etc.).
- **M5** *(optional)* TUI polish: colors per tier, scrollback, copy, width/resize handling.

## Open questions / risks

- **TUI is the main risk** (redraw, wrapping, scroll). Mitigation: keep the live region
  to the last few utterances; commit finalized ones to plain scrollback.
- **Long utterances (no pause)** show nothing until the pause. Mitigation: a
  max-utterance cutoff (as in `transcribe.py`) to force a segment, or Tier 2.
- **GPU backlog** if speech outpaces large-v3: bound the remote queue; if it grows,
  keep drafts and mark finals as pending/skipped rather than falling behind silently.
- **Network drop**: drafts still show; mark utterances as "not upgraded"; auto-reconnect ssh.
- **Correction layer** placement (final tier) and whether it's worth the brittleness — TBD.

## Validation

Reuse recordings: a 1×-paced replay harness feeds a recording through `tiered.py`
and reports, per utterance, draft-latency, final-latency, and draft-vs-final text —
so the tiered UX is measured on fixed audio, not live takes.
