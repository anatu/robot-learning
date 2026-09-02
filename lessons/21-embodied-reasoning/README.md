# Lesson 21 — Embodied Reasoning as a Planning Layer

A slow reasoner plans and verifies, a fast policy acts; the principle is that a hierarchical system is only debuggable if every layer was measured *before* the system was, and the proof is a failure-attribution table your component measurements predicted.

> **Version note (Aug 2026).** This course was scaffolded against Gemini Robotics-ER 1.5. That model line has rolled: `gemini-robotics-er-1.6-preview` is deprecated (shutting down end of Aug 2026) and the current model is **`gemini-robotics-er-2-preview`** (plus a `-streaming-` variant), with a new `client.interactions.create` API surface. Everything below targets ER 2. Re-verify the model string and SDK shape at ai.google.dev/gemini-api/docs/robotics-overview before writing code — this is the fastest-moving dependency in the course.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~2 sessions: API warm-up + grounding eval (2–3 h), planner-executor loop + comparison (3–4 h), attribution (1–2 h). Desk time assumes AI-assisted coding |
| **Cost** | Gemini API usage — modest at this volume (low hundreds of image calls); check current free-tier/rate-limit docs. $0 GPU (executor runs your existing checkpoints on `mps`) |
| **Prerequisites** | 14 or 15 (a working sim policy checkpoint + the `evaluate(...)` harness), 02 (you can script sim scenes), 20 (learned verification — ER's success detection is the same problem, hosted) |
| **Feeds into** | 22 (capstone options gain a planning layer), H-track finale (swap the sim executor for the real SO-101 policy) |

## Learning objectives

After this lesson you can:

1. **Quantify** an embodied-reasoning model's grounding accuracy on your own images against hand labels, including its sensitivity to prompt phrasing.
2. **Explain** the planner-executor contract — typed subtasks, grounding at dispatch time, verification gates, bounded replanning — and why each piece is where it is.
3. **Predict** hierarchical-vs-flat performance on multi-step tasks from component measurements, then test the prediction at matched budgets.
4. **Attribute** every system failure to exactly one layer — grounding, planning, execution, verification-FP, verification-FN — and defend the label from logs.

## Principles

**Why hierarchy.** A flat language-conditioned policy must pack goal decomposition, grounding, and control into one reactive mapping; multi-step instructions ("put both cubes in the bowl, red first") stress exactly the parts it lacks — persistent task state and compositional language. The hierarchical bet (PI's Hi Robot; DeepMind's Gemini Robotics line): a slow, general VLM handles semantics and sequencing at ~1 Hz; a fast, task-tuned policy handles contact and control at 30–50 Hz. The interface between them is tiny — subtask strings, points, and success verdicts — which is what makes the architecture buildable in one lesson.

**What the ER API gives you** (verify per version note):

- **Pointing / boxes:** ask for objects and get `[{"point": [y, x], "label": "..."}]` with coordinates **normalized to 0–1000, y first**. Convert with `px = x/1000 * W`, `py = y/1000 * H` — the y-first order is the classic bug.
- **Planning:** free-form or JSON-constrained multi-step decompositions of a language goal against the current image.
- **Verification:** success/failure judgments from frames (ER 2 adds video progress classification — moment finding and progress states from continuous feeds).
- **Thinking budget:** ER 2 exposes a `thinking_level` (or equivalent) knob — latency vs reasoning depth; grounding-heavy calls tolerate low, planning benefits from high.

The call shape as currently documented (confirm before use):

```python
from google import genai
client = genai.Client()  # GEMINI_API_KEY in env; key must not be unrestricted
resp = client.interactions.create(
    model="gemini-robotics-er-2-preview",
    input=[{"type": "image", "uri": f.uri, "mime_type": f.mime_type},
           {"type": "text",  "text": PROMPT}],
    generation_config={"thinking_level": "high"},
)
```

**The contract that makes it debuggable.** Plans are typed subtask lists, not prose. Grounding happens at *dispatch* time, against the current frame, not once at planning time — the scene moves after every subtask. A verifier, not the executor, decides whether a subtask succeeded; the loop trusts the verdict and replans a bounded number of times. Verification errors are asymmetric: a false positive advances the loop on a lie (silent, compounding), a false negative wastes a replan (loud, bounded). Every one of these choices maps to a failure class in Exercise 5, which is why measuring grounding and verification standalone (Exercises 2–3) *predicts* the attribution table.

**Carry forward**

- Hierarchy trades one hard problem (reactive compositional control) for three measurable ones: grounding, planning, verification. Measure each before wiring them together.
- Ground at dispatch time; the frame you planned from is stale by the second subtask.
- Verification FPs are worse than FNs in a sequential loop; threshold accordingly.
- Schemas and evals survive a model roll; prompts don't. Put the model string in one file.

| Source | Read for |
|---|---|
| ai.google.dev robotics docs (overview + video-progress guide) | current model string, call shape, point/box formats, safety guidance — Exercise 1's ground truth |
| PI Hi Robot post | the hierarchical framing: what the high level should *not* try to do |
| Gemini Robotics 2 announcement (DeepMind blog, Jul 2026 — locate and verify) | what moved between ER 1.5 and ER 2; note claims you can test yourself |
| Your Lesson 20 reward-model results | the calibration lens you'll reapply to ER's verification verdicts |

## Exercise 1 — Access + a schema-validated client [Build]

Tests the "schemas survive rolls" principle: one client, one place the model string lives.

1. Create an API key in AI Studio; note the docs' warning that unrestricted keys are rejected with `403 Forbidden` — restrict it.
2. Reproduce the call above on one rendered frame from your Lesson 02 MuJoCo scene. Parse the JSON, convert coordinates, draw the points on the image.
3. Spec for `er_client.py` (AI-drafted; you read it against the docs): `point(image, phrase) -> list[Point]`, `plan(image, goal) -> Plan`, `verify(image, question) -> bool` (Plan schema in Exercise 3). Every response is validated with `pydantic`; a schema failure triggers one retry with a schema reminder in the prompt. Retry + exponential backoff on `429`. Per-call log: prompt, image hash, response, latency. A call counter — you report total calls and cost. The model string appears exactly once. The check: force a malformed response with a deliberately ambiguous prompt and confirm the retry path runs and the log records both attempts.

**✅ Checkpoint:** annotated image with points on the right objects; the malformed-JSON path exercised and handled by one retry-with-schema-reminder; `grep -c "gemini-robotics-er" er_client.py` prints 1.

## Exercise 2 — Grounding accuracy on your own images [Predict → Run]

Tests objective 1: quantify grounding before trusting it inside a loop.

1. Build a 15-image eval set: 10 renders from your sim scenes (≥ 480p — low-res renders tank grounding) + 5 real photos of your desk/objects (H2-style frames if the hardware track has started). 2–4 nameable objects each.
2. Hand-label ground truth: a bounding polygon (or box) per object. `labelme` or a 30-line matplotlib clicker (AI-drafted) both work.
3. **Write first** in `RESULTS.md`: predicted point-in-mask hit rate for sim vs real, and which of two phrasings — bare noun ("red cube") vs descriptive ("the small red cube left of the bowl") — you expect to score higher, and why.
4. Metric script (AI-drafted from spec): fraction of returned points falling inside the GT polygon of the named object, per image source and per object category; miss distance (px, normalized by image diagonal) for failed points. Run both phrasings over all 15 images.
5. Reconcile against your predictions.

**✅ Checkpoint:** hit rate ≥ ~80% on sim images for unambiguous objects (if far below, inspect renders — resolution and lighting first); the phrasing table shows a measurable effect in some direction; predictions and outcomes both in `RESULTS.md`.

## Exercise 3 — Plans and verification as typed objects [Build]

Tests objective 2: the planner-executor contract, and verification measured against ground truth before the loop depends on it.

1. The plan schema (this *is* the contract):
   ```python
   class Subtask(BaseModel):
       action: Literal["pick", "place", "push", "move_to"]
       target: str                  # object phrase to ground at dispatch time
       destination: str | None
       success_check: str           # question ER answers from an after-frame
   class Plan(BaseModel):
       goal: str
       subtasks: list[Subtask]
   ```
2. Prompt ER 2 to emit plans in this schema for 5 goals against your scene ("stack red on blue", "clear the table into the bowl", …). Validate; log the raw text of any schema failure.
3. **Verification study, prediction first:** write your expected precision and recall for ER's success verdicts, and whether you expect it to beat your Lesson 20 reward model. Then from your Lesson 14 eval videos extract 20 before/after frame pairs (10 successes, 10 failures, GT known); ask ER the `success_check` question per pair; compute precision/recall against GT. Add the row to Lesson 20's calibration table — a hosted reasoner and a local reward model, same yardstick.

**✅ Checkpoint:** 5/5 goals produce schema-valid plans (with ≤ 1 retry each); verification precision/recall computed on 20 pairs and compared in writing to your Lesson 20 reward model.

## Exercise 4 — The planner-executor loop, hierarchical vs flat [Predict → Run]

Tests objective 3: does hierarchy buy what the component numbers say it should? Executor = your Lesson 14/15 checkpoint behind the `evaluate(...)` harness interface, or a scripted pick/place controller (Lesson 02's machinery) if your policy is single-task — the loop, not the executor, is under test.

```
loop(goal, max_replans=2):
  frame ← observe()
  plan  ← ER.plan(goal, frame)                     # Exercise 3 schema
  for sub in plan.subtasks:
      pt   ← ER.point(sub.target, observe())        # ground at dispatch, not plan, time
      ok   ← executor.run(sub, pt, timeout=T)       # policy rollout or scripted skill
      ok   ← ER.verify(sub.success_check, observe())# trust the verifier, not the executor
      if not ok:
          replans += 1; if replans > max_replans: return FAIL(sub)
          plan ← ER.replan(goal, observe(), history)
  return SUCCESS
```

1. Spec for `loop.py` (AI-drafted from the pseudocode): full logging — every ER call, every subtask outcome, every replan, per-layer wall-clock; a fixed seed list; `tasks/` defines two multi-step tasks in your MuJoCo scene, e.g. (a) "put the red cube in the bowl, then the blue cube", (b) "stack blue on red" (order constraint). 10 episodes each, randomized object placements.
2. **Write first:** predicted success rate per task for hierarchical and for flat, and which task the flat baseline will fail outright. Derive the hierarchical prediction from Exercise 2's hit rate and Exercise 3's verification precision (roughly: per-subtask success ≈ grounding hit × executor success × verifier agreement, compounded over subtasks and replans).
3. Run hierarchical. Then the flat baseline at matched budget: the same language-conditioned policy given the raw goal, same episode/timeout budget, no ER in the loop. (If your executor is scripted, the honest flat baseline is the policy alone — state this asymmetry in `RESULTS.md`.)
4. Reconcile against your predictions.

**✅ Checkpoint:** hierarchical completes ≥ 1 task the flat baseline can't; per-layer time budget logged (expect ER calls ~1–5 s each — the 1 Hz/50 Hz split made concrete); predicted vs measured success in `RESULTS.md`.

## Exercise 5 — Failure attribution [Diagnose]

Tests objective 4: the diagnostic skill that makes hierarchical systems debuggable.

1. Label every failed episode from Exercise 4 with exactly one primary cause, from the logs: **grounding** (pointed wrong), **planning** (wrong/impossible decomposition), **execution** (right subtask, policy failed), **verification-FP** (claimed success falsely — the silent killer: the loop advances on a lie), **verification-FN** (denied success → wasted replans).
2. Table: task × failure layer, with counts.
3. One paragraph: which single layer, if perfected, buys the most success — and does the Exercise 2/3 measured accuracy predict the attribution split? (It should: that's the point of measuring components first. Where it doesn't, say which component measurement was unrepresentative and why.)

**✅ Checkpoint:** every failure has exactly one label; the component-measurement → system-failure comparison is written.

## Exercise 6 — Where verification should live [Decide]

Tests the verification-asymmetry principle. Using Exercise 3's ER precision/recall and Lesson 20's reward-model calibration, decide which verifier you would deploy in this loop and at what threshold, given that FPs compound and FNs cost one replan. Defend it with the two rows side by side, and state the condition under which you'd switch.

**✅ Checkpoint:** the decision and its two supporting rows are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `er_client.py` | retries, schema validation, call log + counter; model string in one place; no raw API calls anywhere else |
| `grounding_eval/` | 15 labeled images, metric script, per-source + per-phrasing tables |
| `loop.py` + `tasks/` | the pseudocode realized; 2 tasks × 10 seeded episodes rerunnable from one command |
| `RESULTS.md` | Exercise 2/3/4 predictions with reconciliations; hierarchical-vs-flat table (±CI); attribution table; verification P/R vs Lesson 20's reward model; the Exercise 6 decision; total API calls + cost |

## Done when

- [ ] Grounding, planning, and verification each have a standalone measured number *before* the system result, with predictions written first.
- [ ] The hierarchical system beats flat on ≥ 1 of 2 multi-step tasks, or the attribution table explains precisely why not.
- [ ] Every failure is attributed to one layer; verification errors are split FP/FN.
- [ ] The verifier decision is defended with both rows.
- [ ] API cost and call counts are reported.

## Self-check

1. Why ground `sub.target` at dispatch time rather than once at planning time? Which failure class does the wrong choice inflate?
2. Verification false-positives are worse than false-negatives in this loop. Why? What threshold asymmetry follows?
3. Your flat baseline lacks ER entirely. Name the two distinct advantages the hierarchical system gets, and an experiment that isolates each.
4. When would you move verification from the hosted reasoner to Lesson 20's local reward model, and what do you lose?
5. The ER model line rolled once during this course already. Which parts of your build survive the next roll untouched, and why (hint: schemas and evals, not prompts)?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` on every call | unrestricted API key | restrict the key in AI Studio per the docs |
| `429` mid-episode kills the loop | rate limits at loop cadence | backoff in `er_client.py`; cache grounding within a subtask; batch the offline evals |
| Points land systematically off-object | y/x order or 0–1000 normalization mishandled | unit-test the conversion on a synthetic image with a known target |
| Plans valid but physically absurd | prompt lacks scene/embodiment constraints | include reachable-workspace and gripper limits in the planning prompt; re-run Exercise 3 |
| Grounding great on photos, poor on renders | sim render quality | ≥ 480p, lighting on, textured objects; re-measure before blaming the model |
| Model string 404s | the preview line rolled again | back to the docs page; only `er_client.py` should need the edit |
| Attribution split disagrees with component numbers | eval images unrepresentative of loop-time frames (occlusion by gripper, mid-motion blur) | add loop-time frames to the grounding/verification eval sets and re-measure |

## Going deeper

- **Third task + phrasing ablation at scale:** add "move everything that is not a bowl to the left half" (a set-valued goal that stresses planning) and a functional phrasing ("the object you would pick to fill the bowl") across a 30-image grounding set.
- **Streaming verification:** wire the `-streaming-preview` variant for continuous progress monitoring instead of before/after verification: detect subtask completion mid-execution and cut executor timeouts dynamically. Measure the wall-clock saving per episode.

## References

- Gemini API robotics docs: ai.google.dev/gemini-api/docs/robotics-overview (+ video-progress guide, rate-limits page).
- Physical Intelligence, *Hi Robot* (hierarchical VLM-over-policy framing).
- Google DeepMind, Gemini Robotics / Gemini Robotics 2 announcements (verify current post).
- Your Lesson 20 `reward_calibration/` — the comparison baseline for hosted verification.
