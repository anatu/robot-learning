# Lesson 21 — Embodied Reasoning as a Planning Layer

Build the "reasoner plans, policy acts" architecture hands-on: a hosted embodied-reasoning VLM decomposes language goals, points at objects, dispatches subtasks to your own low-level policy, verifies outcomes, and replans — and you measure which layer fails when the system does.

> **Version note (Aug 2026).** This course was scaffolded against Gemini Robotics-ER 1.5. That model line has rolled: `gemini-robotics-er-1.6-preview` is deprecated (shutting down end of Aug 2026) and the current model is **`gemini-robotics-er-2-preview`** (plus a `-streaming-` variant), with a new `client.interactions.create` API surface. Everything below targets ER 2. Re-verify the model string and SDK shape at ai.google.dev/gemini-api/docs/robotics-overview before writing code — this is the fastest-moving dependency in the course.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~3 sessions: API warm-up + pointing eval (3–4 h), planner-executor loop (4–6 h), comparison + attribution (2–3 h) |
| **Cost** | Gemini API usage — modest at this volume (hundreds of image calls); check current free-tier/rate-limit docs. $0 GPU (executor runs your existing checkpoints on `mps`) |
| **Prerequisites** | 14 or 15 (a working sim policy checkpoint + eval harness), 02 (you can script sim scenes), 20 (learned verification — ER's success detection is the same problem, hosted) |
| **Feeds into** | 22 (capstone options gain a planning layer), H-track finale (swap the sim executor for the real SO-101 policy) |

## Learning objectives

After this lesson you can:

1. **Drive** an embodied-reasoning model through its API for pointing, spatial grounding, plan generation, and success verification, with schema-validated outputs.
2. **Quantify** grounding accuracy on your own images against hand labels, including its sensitivity to prompt phrasing.
3. **Build** a planner-executor loop with explicit interfaces: plans as typed subtask lists, verification gates between steps, bounded replanning.
4. **Compare** hierarchical vs flat execution on multi-step tasks with matched budgets.
5. **Attribute** every system failure to grounding, planning, execution, or verification — the diagnostic skill that makes hierarchical systems debuggable.

## Background

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

| Source | Read for |
|---|---|
| ai.google.dev robotics docs (overview + video-progress guide) | current model string, call shape, point/box formats, safety guidance — Part 0's ground truth |
| PI Hi Robot post | the hierarchical framing: what the high level should *not* try to do |
| Gemini Robotics 2 announcement (DeepMind blog, Jul 2026 — locate and verify) | what moved between ER 1.5 and ER 2; note claims you can test yourself |
| Your Lesson 20 reward-model results | the calibration lens you'll reapply to ER's verification verdicts |

## Part 0 — Access + first grounded call (~1 h)

1. Create an API key in AI Studio; note the docs' warning that unrestricted keys are rejected with `403 Forbidden` — restrict it.
2. Reproduce the call above on one rendered frame from your Lesson 02 MuJoCo scene. Parse the JSON, convert coordinates, draw the points on the image.
3. Wrap the client in `er_client.py` with: retry + exponential backoff on rate limits, JSON-schema validation of every response (`pydantic`), a per-call log (prompt, image hash, response, latency), and a call counter — you'll report total calls and cost.

**✅ Checkpoint:** annotated image with points on the right objects; malformed-JSON path exercised (force it with an ambiguous prompt) and handled by one retry-with-schema-reminder.

## Part 1 — Pointing accuracy on your own images (~2–3 h)

Quantify grounding before trusting it inside a loop.

1. Build a 30-image eval set: ~20 renders from your sim scenes (≥ 480p — low-res renders tank grounding) + ~10 real photos of your desk/objects (H2-style frames if the hardware track has started). 2–4 nameable objects each.
2. Hand-label ground truth: a bounding polygon (or box) per object. `labelme` or a 30-line matplotlib clicker both work.
3. Metric: fraction of returned points falling inside the GT polygon of the named object (point-in-mask hit rate), reported per image source (sim vs real) and per object category. Also record miss distances (px, normalized by image diagonal) for failed points.
4. Prompt-sensitivity mini-ablation: three phrasings per object — bare noun ("red cube"), descriptive ("the small red cube left of the bowl"), functional ("the object you would pick to fill the bowl"). Same images, same metric, three numbers.

**✅ Checkpoint:** hit rate ≥ ~80% on sim images for unambiguous objects (if far below, inspect renders — resolution and lighting first); the ablation table shows a measurable phrasing effect in some direction.

## Part 2 — Plans and verification as typed objects (~2 h)

1. Define the plan schema (this is the planner-executor contract):
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
3. Verification study: from your Lesson 14 eval videos, extract ~40 before/after frame pairs (half successes, half failures, GT known). Ask ER the `success_check` question per pair; compute precision/recall against GT. Cross-reference the numbers into Lesson 20's calibration table — a hosted reasoner and a local reward model, same yardstick.

**✅ Checkpoint:** 5/5 goals produce schema-valid plans (with ≤ 1 retry each); verification precision/recall computed on ≥ 40 pairs and compared in writing to your Lesson 20 reward model.

## Part 3 — The planner-executor loop (~4–6 h)

The core build. Executor = your Lesson 14/15 checkpoint behind the Lesson 14 harness interface, or a scripted pick/place controller (Lesson 02's machinery) if your policy is single-task — the loop, not the executor, is under test.

```
loop(goal, max_replans=2):
  frame ← observe()
  plan  ← ER.plan(goal, frame)                     # Part 2 schema
  for sub in plan.subtasks:
      pt   ← ER.point(sub.target, observe())        # ground at dispatch, not plan, time
      ok   ← executor.run(sub, pt, timeout=T)       # policy rollout or scripted skill
      ok   ← ER.verify(sub.success_check, observe())# trust the verifier, not the executor
      if not ok:
          replans += 1; if replans > max_replans: return FAIL(sub)
          plan ← ER.replan(goal, observe(), history)
  return SUCCESS
```

1. Implement with full logging: every ER call, every subtask outcome, every replan, per-layer wall-clock.
2. Three multi-step tasks in your MuJoCo scene, e.g.: (a) "put the red cube in the bowl, then the blue cube", (b) "stack blue on red" (order constraint), (c) "move everything that is not a bowl to the left half". 10 episodes each, randomized object placements, fixed seed list.
3. Flat baseline, matched budget: the same language-conditioned policy given the raw goal, same episode/timeout budget, no ER in the loop. (If your executor is scripted, the honest flat baseline is the policy alone — state this asymmetry in RESULTS.md.)

**✅ Checkpoint:** hierarchical completes ≥ 1 task the flat baseline can't; per-layer time budget logged (expect ER calls ~1–5 s each — the 1 Hz/50 Hz split made concrete).

## Part 4 — Failure attribution (~2 h)

1. Label every failed episode with exactly one primary cause: **grounding** (pointed wrong), **planning** (wrong/impossible decomposition), **execution** (right subtask, policy failed), **verification-FP** (claimed success falsely — the silent killer: the loop advances on a lie), **verification-FN** (denied success → wasted replans).
2. Table: task × failure layer, with counts. One paragraph: which single layer, if perfected, buys the most success — and does your Part 1/2 measured accuracy predict the attribution split? (It should: that's the point of measuring components first.)

**✅ Checkpoint:** every failure has exactly one label; the component-measurement → system-failure comparison is written.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `er_client.py` | retries, schema validation, call log + counter; no raw `generate` calls anywhere else |
| `grounding_eval/` | 30 labeled images, metric script, per-source + per-phrasing tables |
| `loop.py` + `tasks/` | the pseudocode realized; 3 tasks × 10 seeded episodes rerunnable |
| `RESULTS.md` | hierarchical-vs-flat table (±CI); attribution table; verification P/R vs Lesson 20's reward model; total API calls + cost |

## Done when

- [ ] Grounding, planning, and verification each have a standalone measured number *before* the system result.
- [ ] The hierarchical system beats flat on ≥ 2 of 3 multi-step tasks, or the attribution table explains precisely why not.
- [ ] Every failure is attributed to one layer; verification errors are split FP/FN.
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
| Plans valid but physically absurd | prompt lacks scene/embodiment constraints | include reachable-workspace and gripper limits in the planning prompt; re-run Part 2 |
| Grounding great on photos, poor on renders | sim render quality | ≥ 480p, lighting on, textured objects; re-measure before blaming the model |
| Model string 404s | the preview line rolled again | back to the docs page; only `er_client.py` should need the edit |

## Stretch

Wire the streaming variant (`-streaming-preview`) for continuous progress monitoring instead of before/after verification: detect subtask completion mid-execution and cut executor timeouts dynamically. Measure the wall-clock saving per episode.

## References

- Gemini API robotics docs: ai.google.dev/gemini-api/docs/robotics-overview (+ video-progress guide, rate-limits page).
- Physical Intelligence, *Hi Robot* (hierarchical VLM-over-policy framing).
- Google DeepMind, Gemini Robotics / Gemini Robotics 2 announcements (verify current post).
- Your Lesson 20 `reward_calibration/` — the comparison baseline for hosted verification.
