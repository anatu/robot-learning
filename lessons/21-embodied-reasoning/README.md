# Lesson 21 — Embodied Reasoning as a Planning Layer

This lesson builds a hierarchical robot system in which a hosted embodied-reasoning model handles the slow, semantic parts of a task and a policy you trained earlier handles the fast, contact-rich parts. The reasoning model decomposes a language goal into subtasks, points at the objects each subtask needs, judges whether each subtask succeeded, and replans when it did not; the policy executes each subtask. The method of the lesson is to measure each layer on its own before assembling the system, so that when the assembled system fails you can say which layer failed and check that verdict against the component measurements. That discipline, rather than the particular model used, is what carries over to any hierarchical system you build later.

> **Version note (Aug 2026).** This course was scaffolded against Gemini Robotics-ER 1.5. That model line has since rolled: `gemini-robotics-er-1.6-preview` is deprecated and shuts down at the end of August 2026, and the current model is **`gemini-robotics-er-2-preview`** (with a `-streaming-` variant), served through a new `client.interactions.create` API surface. Everything below targets ER 2. This is the fastest-moving dependency in the course, so re-verify the model string and the SDK call shape at ai.google.dev/gemini-api/docs/robotics-overview before writing any code.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~2 sessions: API warm-up + grounding eval (2–3 h), planner-executor loop + comparison (3–4 h), attribution (1–2 h). Desk time assumes AI-assisted coding |
| **Cost** | Gemini API usage, modest at this volume (low hundreds of image calls); check the current free-tier and rate-limit docs. $0 GPU (the executor runs your existing checkpoints on `mps`) |
| **Prerequisites** | 14 or 15 (a working sim policy checkpoint and the `evaluate(...)` harness), 02 (you can script sim scenes), 20 (learned verification; ER's success detection is the same problem, hosted) |
| **Feeds into** | 22 (capstone options gain a planning layer), the hardware-track finale (swap the sim executor for the real SO-101 policy) |

## Learning objectives

After this lesson you can:

1. **Quantify** an embodied-reasoning model's grounding accuracy on your own images against hand labels, including its sensitivity to prompt phrasing.
2. **Explain** the planner-executor contract (typed subtasks, grounding at dispatch time, verification gates, bounded replanning) and why each element is placed where it is.
3. **Predict** hierarchical-versus-flat performance on multi-step tasks from component measurements, then test the prediction at matched budgets.
4. **Attribute** every system failure to exactly one layer (grounding, planning, execution, verification false positive, verification false negative) and defend the label from the logs.

## Principles

### Why a hierarchy

A flat language-conditioned policy has to do everything in one reactive mapping from observation and instruction to action: it must decompose the goal, find the relevant objects, and produce motor commands, all at control rate. Multi-step instructions such as "put both cubes in the bowl, red first" stress exactly the capabilities such a policy lacks, namely persistent task state (which cube has already been placed) and compositional language (the ordering constraint). The hierarchical alternative, which Physical Intelligence's Hi Robot and DeepMind's Gemini Robotics line both adopt, splits the problem by timescale. A slow, general vision-language model handles semantics and sequencing at roughly 1 Hz, and a fast, task-tuned policy handles contact and control at 30–50 Hz. The interface between the two layers is small, consisting of subtask strings, image points, and success verdicts, and that small interface is what makes the architecture buildable in one lesson: each side can be tested against the interface without the other.

### What the embodied-reasoning API provides

The hosted model exposes four capabilities that the loop in this lesson uses. Verify each against the docs page named in the version note before relying on it.

- **Pointing and boxes.** You ask for named objects and receive a list of the form `[{"point": [y, x], "label": "..."}]`. The coordinates are normalized to the range 0–1000, and the y coordinate comes first. To convert to pixels use `px = x/1000 * W` and `py = y/1000 * H`. The y-first ordering is the most common source of systematically misplaced points, so it deserves a unit test.
- **Planning.** The model produces multi-step decompositions of a language goal against the current image, either as free text or constrained to a JSON schema you supply.
- **Verification.** The model returns success or failure judgments from one or more frames. ER 2 also classifies progress from video, which allows moment finding and progress states to be read from a continuous feed.
- **Thinking budget.** ER 2 exposes a `thinking_level` setting (or its equivalent) that trades latency for reasoning depth. Grounding-heavy calls tolerate a low setting; planning benefits from a high one.

The call shape as currently documented is shown below. Confirm it before use.

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

### The planner-executor contract

Given those capabilities, the question is how to wire them together so that the resulting system can be debugged. The contract this lesson uses has four parts, and each part exists to make a specific class of failure visible.

First, plans are typed subtask lists rather than prose, so that a malformed or physically impossible plan is caught by schema validation before anything moves. Second, grounding happens at dispatch time, against the frame captured just before a subtask runs, rather than once when the plan was made. The scene changes after every subtask, so a point computed at planning time is stale by the second subtask. Third, a separate verifier, not the executor, decides whether a subtask succeeded. The executor's own notion of completion (for example, a policy rollout reaching its timeout) says nothing about whether the world is in the intended state. Fourth, replanning is bounded: the loop trusts the verifier's verdict and replans at most a fixed number of times before declaring failure, so that a persistent misjudgment cannot consume unbounded time.

Because each part of the contract corresponds to a failure class, measuring grounding accuracy and verification accuracy on their own (Exercises 2 and 3) gives you a prediction of how the assembled system's failures will be distributed (Exercise 5). If the prediction and the attribution table disagree, the disagreement tells you which component measurement was unrepresentative of the conditions the loop actually encounters.

### Verification errors are asymmetric

The two kinds of verification error have very different costs in a sequential loop. A false positive, in which the verifier reports success for a subtask that failed, advances the loop to the next subtask on a false premise. The error is silent, and every later subtask inherits it, so its cost compounds. A false negative, in which the verifier reports failure for a subtask that succeeded, triggers a replan. The cost is one wasted replan, which is loud and bounded. This asymmetry means the verifier's threshold should favour precision over recall, and it is the reason Exercise 6 asks you to choose a verifier on the basis of precision rather than overall accuracy.

**Carry forward**

- A hierarchy replaces one hard problem, reactive compositional control, with three problems that can each be measured on their own: grounding, planning, and verification. Measure each before wiring them together, because the component measurements predict how the assembled system will fail.
- Ground each subtask's target at dispatch time rather than at planning time, because the scene changes after every subtask and a point computed from the planning frame is stale by the second subtask.
- In a sequential loop a verification false positive compounds silently across later subtasks, whereas a false negative costs one bounded replan; set the verification threshold to favour precision accordingly.
- Schemas and evaluation sets survive a model roll and prompts do not, so keep the model string in exactly one file and invest in the parts that survive.

| Source | Read for |
|---|---|
| ai.google.dev robotics docs (overview + video-progress guide) | the current model string, call shape, point and box formats, and safety guidance; this is Exercise 1's ground truth |
| PI Hi Robot post | the hierarchical framing, and in particular what the high-level model should not try to do |
| Gemini Robotics 2 announcement (DeepMind blog, Jul 2026; locate and verify) | what changed between ER 1.5 and ER 2, with attention to claims you can test yourself |
| Your Lesson 20 reward-model results | the calibration lens you will reapply to ER's verification verdicts |

## Exercise 1 — Set up access and a schema-validated client [Build]

In this exercise you obtain API access, make one grounded call by hand, and then wrap the API in a client module that validates every response against a schema. The client is the only place the model string appears, which is what allows the rest of the build to survive the next model roll; the principle being exercised is that schemas and evaluations outlive prompts.

1. Create an API key in AI Studio. The docs warn that unrestricted keys are rejected with `403 Forbidden`, so restrict the key before using it.
2. Reproduce the call shown in the Principles section on one rendered frame from your Lesson 02 MuJoCo scene. Parse the JSON, convert the coordinates to pixels, and draw the returned points on the image.
3. Write the specification for `er_client.py` and have an AI tool draft it; then read the draft against the docs. The interface is `point(image, phrase) -> list[Point]`, `plan(image, goal) -> Plan`, and `verify(image, question) -> bool`, with the `Plan` schema defined in Exercise 3. Every response is validated with `pydantic`, and a schema failure triggers exactly one retry with a schema reminder added to the prompt. Rate-limit responses (`429`) are retried with exponential backoff. Each call is logged with its prompt, image hash, response, and latency, and a call counter accumulates so that you can report total calls and cost. The model string appears exactly once in the file. The check is to force a malformed response with a deliberately ambiguous prompt and confirm that the retry path runs and the log records both attempts.

**✅ Checkpoint:** the annotated image shows points on the correct objects; the malformed-JSON path has been exercised and handled by one retry with a schema reminder; `grep -c "gemini-robotics-er" er_client.py` prints 1.

## Exercise 2 — Measure grounding accuracy on your own images [Predict → Run]

Before grounding is trusted inside a loop, it should be measured on images like the ones the loop will see. In this exercise you build a small labelled evaluation set, predict how well the model will point at named objects in simulated and real images, and then measure it. The prediction matters because the difference between simulated renders and real photographs is one of the main things the attribution table in Exercise 5 will have to explain.

1. Build a 15-image evaluation set: 10 renders from your simulation scenes at 480p or higher, since low-resolution renders degrade grounding badly, and 5 real photographs of your desk or objects (frames in the style of H2 if the hardware track has started). Each image should contain two to four objects that can be named.
2. Hand-label the ground truth as a bounding polygon or box per object. Either `labelme` or a thirty-line matplotlib clicker (which an AI tool can draft) is sufficient.
3. Before running any calls, write in `RESULTS.md` your predicted point-in-mask hit rate for simulated images and for real photographs, and state which of two phrasings you expect to score higher and why: the bare noun ("red cube") or the descriptive phrase ("the small red cube left of the bowl").
4. Have an AI tool draft the metric script from this specification: for each returned point, record whether it falls inside the ground-truth polygon of the named object, and report the hit rate per image source and per object category; for points that miss, record the distance to the polygon in pixels, normalized by the image diagonal. Run both phrasings over all 15 images.
5. Reconcile the measured rates against your predictions.

**✅ Checkpoint:** the hit rate is at or above roughly 80% on simulated images for unambiguous objects; if it is far below that, inspect the renders for resolution and lighting problems before drawing conclusions about the model. The phrasing table shows a measurable effect in one direction or the other, and both predictions and outcomes are in `RESULTS.md`.

## Exercise 3 — Define plans and verification as typed objects [Build]

Here you fix the planner-executor contract as code, then measure the verifier against ground truth before the loop is allowed to depend on it. The plan schema is the contract between the two layers, and the verification study reuses the calibration method from Lesson 20 so that a hosted reasoner and a local reward model can be compared on the same yardstick.

1. Define the plan schema. This schema is the contract:
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
2. Prompt ER 2 to emit plans in this schema for five goals against your scene, for example "stack red on blue" and "clear the table into the bowl". Validate each plan, and log the raw text of any response that fails the schema.
3. Run the verification study, prediction first. Write down your expected precision and recall for ER's success verdicts and whether you expect it to outperform your Lesson 20 reward model. Then, from your Lesson 14 evaluation videos, extract 20 before-and-after frame pairs (10 successes and 10 failures, with ground truth known), ask ER the `success_check` question for each pair, and compute precision and recall against the ground truth. Add the resulting row to Lesson 20's calibration table so that the hosted reasoner and the local reward model sit side by side.

**✅ Checkpoint:** all five goals produce schema-valid plans with at most one retry each; verification precision and recall are computed on the 20 pairs and compared in writing to the Lesson 20 reward model.

## Exercise 4 — Run the planner-executor loop against a flat baseline [Predict → Run]

This is the assembled system. You realize the loop below, run it on two multi-step tasks, and compare it against a flat baseline that receives the same goal with no reasoning layer. The prediction step asks you to derive the expected hierarchical success rate from the component measurements in Exercises 2 and 3, which is the concrete form of the principle that component measurements predict system behaviour.

The executor is either your Lesson 14 or 15 checkpoint behind the `evaluate(...)` harness interface, or a scripted pick-and-place controller built from Lesson 02's machinery if your policy is single-task. The loop, not the executor, is what is under test.

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

1. Write the specification for `loop.py` and have an AI tool draft it from the pseudocode. It must log every ER call, every subtask outcome, every replan, and the wall-clock time spent in each layer, and it must accept a fixed seed list. A `tasks/` module defines two multi-step tasks in your MuJoCo scene, for example (a) "put the red cube in the bowl, then the blue cube" and (b) "stack blue on red", which carries an ordering constraint. Each task runs for 10 episodes with randomized object placements.
2. Before running, write in `RESULTS.md` your predicted success rate per task for the hierarchical system and for the flat baseline, and name the task you expect the flat baseline to fail outright. Derive the hierarchical prediction from Exercise 2's hit rate and Exercise 3's verification precision. A serviceable approximation is that per-subtask success is roughly the product of the grounding hit rate, the executor's success rate, and the verifier's agreement rate, compounded over the number of subtasks and allowing for the bounded replans.
3. Run the hierarchical system. Then run the flat baseline at a matched budget: the same language-conditioned policy given the raw goal, with the same episode and timeout budget and no reasoning model in the loop. If your executor is scripted, the honest flat baseline is the policy alone; state that asymmetry in `RESULTS.md`.
4. Reconcile the measured rates against your predictions.

**✅ Checkpoint:** the hierarchical system completes at least one task that the flat baseline cannot; the per-layer time budget is logged, and ER calls take roughly 1–5 s each, which is the 1 Hz versus 50 Hz split of the Principles section made concrete; predicted and measured success rates are both in `RESULTS.md`.

## Exercise 5 — Attribute every failure to one layer [Diagnose]

A hierarchical system is only debuggable if each failure can be assigned to the layer that caused it. In this exercise you label every failed episode from Exercise 4 with a single primary cause, tabulate the causes, and compare the distribution against what the component measurements led you to expect.

1. From the logs, label every failed episode with exactly one primary cause: **grounding** (the model pointed at the wrong location), **planning** (the decomposition was wrong or impossible), **execution** (the subtask was correct and the policy failed to perform it), **verification-FP** (the verifier claimed success falsely, so the loop advanced on a false premise), or **verification-FN** (the verifier denied a real success, so replans were wasted).
2. Produce a table of task against failure layer, with counts.
3. Write one paragraph answering two questions. Which single layer, if it were perfect, would buy the most additional success? And do the accuracies measured in Exercises 2 and 3 predict the attribution split? They should, since that is why the components were measured first. Where they do not, say which component measurement was unrepresentative of loop-time conditions and why.

**✅ Checkpoint:** every failure carries exactly one label, and the comparison between component measurements and system failures is written.

## Exercise 6 — Decide where verification should live [Decide]

The loop needs a verifier, and you now have calibration numbers for two candidates: the hosted reasoner from Exercise 3 and the local reward model from Lesson 20. Using those two rows, decide which verifier you would deploy in this loop and at what threshold, taking into account that false positives compound while false negatives cost a single replan. Defend the decision with the two rows placed side by side, and state the condition under which you would switch to the other verifier.

**✅ Checkpoint:** the decision and its two supporting rows are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `er_client.py` | retries, schema validation, call log and counter; the model string in one place; no raw API calls anywhere else |
| `grounding_eval/` | 15 labeled images, the metric script, per-source and per-phrasing tables |
| `loop.py` + `tasks/` | the pseudocode realized; 2 tasks × 10 seeded episodes rerunnable from one command |
| `RESULTS.md` | Exercise 2, 3, and 4 predictions with reconciliations; the hierarchical-vs-flat table (±CI); the attribution table; verification P/R against Lesson 20's reward model; the Exercise 6 decision; total API calls and cost |

## Done when

- [ ] Grounding, planning, and verification each have a standalone measured number, recorded before the system result, with predictions written first.
- [ ] The hierarchical system beats the flat baseline on at least one of the two multi-step tasks, or the attribution table explains precisely why not.
- [ ] Every failure is attributed to one layer, with verification errors split into false positives and false negatives.
- [ ] The verifier decision is defended with both calibration rows.
- [ ] API cost and call counts are reported.

## Self-check

1. Why ground `sub.target` at dispatch time rather than once at planning time? Which failure class does the wrong choice inflate?
2. Verification false positives are worse than false negatives in this loop. Why, and what threshold asymmetry follows?
3. Your flat baseline lacks the reasoning model entirely. Name the two distinct advantages the hierarchical system gets, and an experiment that isolates each.
4. When would you move verification from the hosted reasoner to Lesson 20's local reward model, and what would you lose?
5. The ER model line rolled once during this course already. Which parts of your build survive the next roll untouched, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` on every call | unrestricted API key | restrict the key in AI Studio per the docs |
| `429` mid-episode kills the loop | rate limits at loop cadence | backoff in `er_client.py`; cache grounding within a subtask; batch the offline evals |
| Points land systematically off-object | y/x order or 0–1000 normalization mishandled | unit-test the conversion on a synthetic image with a known target |
| Plans valid but physically absurd | prompt lacks scene/embodiment constraints | include reachable-workspace and gripper limits in the planning prompt; re-run Exercise 3 |
| Grounding good on photos, poor on renders | sim render quality | ≥ 480p, lighting on, textured objects; re-measure before blaming the model |
| Model string returns 404 | the preview line rolled again | back to the docs page; only `er_client.py` should need the edit |
| Attribution split disagrees with component numbers | eval images unrepresentative of loop-time frames (occlusion by gripper, mid-motion blur) | add loop-time frames to the grounding/verification eval sets and re-measure |

## Going deeper

- **A third task and a larger phrasing study.** Add "move everything that is not a bowl to the left half", a set-valued goal that stresses planning, and add a functional phrasing ("the object you would pick to fill the bowl") across a 30-image grounding set.
- **Streaming verification.** Wire the `-streaming-preview` variant for continuous progress monitoring in place of before-and-after verification, so that subtask completion is detected mid-execution and executor timeouts can be cut dynamically. Measure the wall-clock saving per episode.

## References

- Gemini API robotics docs: ai.google.dev/gemini-api/docs/robotics-overview (with the video-progress guide and the rate-limits page).
- Physical Intelligence, *Hi Robot* (the hierarchical VLM-over-policy framing).
- Google DeepMind, Gemini Robotics and Gemini Robotics 2 announcements (verify the current post).
- Your Lesson 20 `reward_calibration/`, which is the comparison baseline for hosted verification.
