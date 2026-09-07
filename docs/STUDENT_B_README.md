# Student B: make a plan with Qwen

Your job is to turn a request into steps the robot can try. A tells you what
the camera sees. You choose the object, destination and action order. C moves
the robot and reports what happened. If a step fails, you make a new plan.

The Qwen interface is implemented. **No real Qwen request has been tested yet.**
You can develop and test the interface now, without an API key or finished A/C.

```mermaid
flowchart LR
    A["A: objects, regions and positions"] --> B["B: instruction + Qwen"]
    B --> J["JSON: goal IDs + ordered skills"]
    J --> V["Python: check plan and fill positions"]
    V --> C["C: attempt the actions"]
    C -->|"result + new scene"| B
```

## 1. Run B offline first

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m planner.run --offline-demo --out runs/qwen_b_offline
```

Use a new output folder on each run, for example `runs/qwen_b_offline_2`.
The command refuses to mix results with an existing run.

Expected output:

```text
READY: APPROACH -> GRASP -> MOVE_TO -> PLACE -> VERIFY -> STOP
Source: fixture
```

This command uses a hand-written response through `FakeTransport`.
It exercises the same parser, compiler, validator and audit code as a real
request. It does not contact Qwen or test language understanding.

| File in the output folder | What to look at |
| --- | --- |
| `input.json` | The instruction and A's scene |
| `scene.json` | Reusable `SceneDescription` input, without the instruction |
| `response.json` | Model-shaped JSON: goal IDs and skills; no coordinates |
| `schema.json` | The exact allowed JSON format |
| `plan.json` | The compiled backbone `Plan` that C receives |
| `diagnostics.json` | Source, model, prompt, raw response, validation, tokens and timing |
| `audit/<episode-id>/001.json` | The first planning call; later calls have increasing numbers |

The sample request is “Move the stone beside the blue cube to the red area.”
The hand-written answer selects stone `p0`. Cube `p1` is a reference object;
region `p2` is the destination. This illustrates the intended answer, not
evidence that Qwen understands this sentence.

## 2. Input and output

The public interface stays the same:

```python
plan = planner.plan(instruction, scene)
plan = planner.replan(instruction, new_scene, history, context, clarification=None)
planner.reset()
```

| Input | Meaning |
| --- | --- |
| `instruction` | The original user request; keep it unchanged during replanning |
| `scene` | A's `SceneDescription`: IDs, classes, colors, status and world positions |
| `history` | Previous action results, including failures such as `GRASP_MISSED` |
| `context.held_instance_id` | The object the system believes is in the gripper |
| `context.last_release_instance_id` | The most recently released object |
| `clarification` | The user's answer to B's earlier question, if any |

The model receives text and structured scene data. B does not need images.
IDs come from A; B never uses simulator object IDs or oracle positions.
The last 12 action results are included, with the total history length.
Executor-private `info` fields are excluded from model input.

Qwen returns a `student-b-plan-v1` JSON object with these required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `student-b-plan-v1` |
| `status` | `READY`, `NEEDS_SEARCH`, `NEEDS_CLARIFICATION`, or `INFEASIBLE` |
| `goal` | `object_id`, `object_name`, `object_color`, `region_id`, `region_name` |
| `actions` | At most 16 actions; each has `skill`, `target`, `object`, `region`, `condition` |
| `reason` | A short reason, required for refusal |
| `clarification_question` | A question when clarification is needed |

Unused strings must be `""`. Unknown IDs must be empty, not invented.
An extra field such as model-generated `params.pos` is rejected.
For example, the model writes:

```json
{"skill":"PLACE","target":"p2","object":"p0","region":"","condition":""}
```

Python uses A's region support point to produce:

```python
Action(Skill.PLACE, target="p2", params={"object": "p0", "pos": [0.4, 0.3, 0.853]})
```

Coordinates use the world frame and meters. Object positions are centers;
region positions are support points. `MOVE_TO` uses the region support point
plus the public 0.18 m transport clearance. Every positional action is bound
from the current scene, never from remembered coordinates.
C may refresh an action's position from newer perception before moving;
see `executor/demo_skills.py` for an example.

## 3. What is checked before C runs

- The JSON fields and types match the schema.
- Goal IDs exist and match the stated class, role and requested color.
- Metric actions use current, unambiguous `LOCALIZED` references.
- An empty-hand plan approaches or reaches before grasping the goal object.
- Transport and placement use that same object and the goal region.
- A ready plan ends with placement `VERIFY`, then `STOP`.
- A replan preserves every nonempty field of the accepted original goal.
- If a different object is held, B cannot silently place it as the requested object.
- A correctly held object is not grasped again. After release, B may verify
  or plan a correction using a new observation.

A held or recently released object may be absent from the current camera
view. Its tracked ID can still be used for a **future verification**. This
does not create a position, make it visible, or pass the final visual check.

`NEEDS_SEARCH` contains only SEARCH actions with canonical class names.
Search is class-based in the current C interface. B keeps color constraints
in its goal and must resolve the right instance after receiving a new scene.
Ambiguity produces a question and no actions. Unsupported requests produce
`INFEASIBLE`, a reason, and no actions.

An invalid model output gets **one repair request**, containing the rejected
output and exact validation error. A second invalid output raises `PlannerError`.
HTTP errors and timeouts also raise `PlannerError`; they are not task refusal.
The orchestrator records an error and stops. There is no automatic rule fallback.

These checks do not prove that the model understood the sentence. A consistent
plan for the wrong object can still pass on the first call. They also do not
prove that IK, grasping or placement will succeed. Language labels and actual
execution tests are both needed.

## 4. Run the offline tests

```bash
cd /home/jiamo/EE4705/project1.3
.venv/bin/python -m pytest -q tests/test_student_b.py tests/test_llm_client.py tests/test_validation.py tests/test_architecture.py
```

These tests cover schema errors, invented IDs, forbidden coordinates, wrong
action order, bounded repair, service failures, goal preservation, fresh
positions, search, clarification, holding state, config and cache provenance.
A guard makes `RequestsTransport.post` fail if a B test tries live HTTP.

Run just the A/B/C integration check:

```bash
.venv/bin/python -m pytest -q tests/test_student_b.py::test_fixture_student_b_connects_to_real_simulated_demo_a_and_c
```

This uses a fixed response through the real B interface, demo RGB-D A and
demo motion C in MuJoCo. It checks the visual claim and an independent
evaluator's object identity, release, destination and stability checks.
It does not use Qwen. The fixed scene is not a randomized benchmark.

Run all project checks:

```bash
.venv/bin/python -m pytest -q
```

## 5. Connect Qwen later

Do this section only when you want to make real API requests.
No new SDK is needed; the shared client uses the existing `requests` dependency.

The default model is `qwen3.7-plus-2026-05-26`. The default output mode is
`json_schema`. Use the base URL supplied by your Model Studio console;
it must end in `/v1`, not `/chat/completions`. The currently documented
Singapore pattern is:

```text
https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
```

Replace the whole placeholder with your actual console URL. Your key and
endpoint must belong to the matching region/workspace.

In Bash, these commands read values without putting the key into shell history:

```bash
cd /home/jiamo/EE4705/project1.3
read -r -p 'Qwen base URL from your console: ' EE4705_QWEN_BASE_URL
export EE4705_QWEN_BASE_URL
read -r -s -p 'Qwen API key: ' DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY
export EE4705_QWEN_MODEL='qwen3.7-plus-2026-05-26'
.venv/bin/python -m planner.run --check-config
```

`--check-config` checks settings only. It sends no request and does not prove
that credentials, the model or structured output work at the endpoint.
The program reads environment variables, not `.env` files automatically.
Do not put keys in source files, command arguments, screenshots or reports.

Make one real planning request using the scene saved by Section 1:

```bash
.venv/bin/python -m planner.run \
  --scene runs/qwen_b_offline/scene.json \
  --instruction 'Move the stone beside the blue cube to the red area.' \
  --out runs/qwen_b_first_api
```

Inspect `diagnostics.json` before running the robot simulation. Then try
your B with the existing working demo A/C and record a video:

```bash
.venv/bin/python -m demo.run --student B --out runs/qwen_b_episode
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory runs/qwen_b_episode
```

Open [the local replay](http://127.0.0.1:8766/). This demo's evaluator expects
the stone in the red region. Use equivalent wording when testing this demo.
The separate planning evaluation uses mock A/C and may use other prepared trials:

```bash
.venv/bin/python -m eval.runner --mode planning --trials eval/trials/smoke --out runs/qwen_b_smoke
```

Optional settings:

| Environment variable | Default / use |
| --- | --- |
| `EE4705_QWEN_API_KEY` | Optional override of `DASHSCOPE_API_KEY`; no OpenAI-key fallback |
| `EE4705_QWEN_OUTPUT_MODE` | `json_schema`; explicitly choose `json_object` for a model without schema support |
| `EE4705_QWEN_THINKING` | Unset: omit provider flag; set `true` or `false` only if the chosen model supports it |
| `EE4705_QWEN_TIMEOUT_S` | `60` per HTTP attempt |
| `EE4705_QWEN_MAX_RETRIES` | `1` additional attempt for retryable HTTP/transport failures |
| `EE4705_QWEN_MAX_TOKENS` | `2048` |
| `EE4705_QWEN_CACHE_DIR` | `runs/qwen_cache`; empty string disables the cache |
| `EE4705_QWEN_CACHE_ONLY` | `false`; `true` requires an existing exact cached response and sends no HTTP |
| `EE4705_QWEN_AUDIT_DIR` | `runs/qwen_b_audit` |

The client never silently changes output mode or model. Both output modes
still run local validation. The default bounds allow at most four HTTP
attempts per planning call: initial + retry, then one repair + retry.
The orchestrator separately bounds replans and robot action attempts.

Cache entries distinguish `live`, `fixture` and `custom_transport` sources.
A live client cannot replay a fixture cache entry. `cached=true` means no new
model call; token/timing fields on a cache hit are zero for that call.
Cache keys include endpoint, model, messages, schema, mode and generation
settings. Invalid cached text is validated again. Client statistic
`live_requests` is a legacy name for uncached transport calls; with a fixture
it does not mean actual HTTP. Use `source` together with `cached` in reports.

Provider references, checked September 7, 2026:
[Qwen structured output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output),
[compatible API](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope),
[Qwen3.7-Plus](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus).
Real endpoint compatibility remains untested in this project.

## 6. Your next development work

Start with `planner/prompts.py` for wording changes. Use `planner/contract.py`
for supported plan shapes and checks, `planner/config.py` for settings and
`planner/student_b.py` for request/repair/replan behavior. Keep the A/B/C
method signatures unchanged.

Prepare at least 20 independently labelled instructions before measuring Qwen:
include paraphrases, distractors, color/relations, missing objects, ambiguity,
unsupported requests and recovery states. Label the intended object, region,
status and action dependencies before seeing the response.

Report language planning accuracy separately from schema validity, repair
rate, latency and execution success. Keep `first_pass_valid`, final validity
after repair and `source`/`cached` in the records. Do not report the fixed
offline responses as correct Qwen predictions. See the general
[student guide](STUDENT_README.md#3-student-b-turn-instructions-into-actions)
for the assignment's evaluation requirements.
