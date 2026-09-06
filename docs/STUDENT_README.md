# Student A / B / C 开发与测试指南

Owner: backbone (ALL) · 更新：2026-09-07 · `CONTRACT_VERSION = 2`

本指南依据 [Project 1.3 作业要求](../EE4705_Project1.3%20S1%20AY2627.pdf)
Task 2–5（PDF 第 3–5 页）。A 负责 Task 2，B 负责 Task 3，C 负责 Task 4；
Task 1 和 Task 5 由三人共同完成。下文给出的阈值、CSV 字段和测试分组是本项目建议的评测协议，
不是课程额外规定的评分线。

**现在可以并行开发三个模块。真实 A/B/C 仍是 stub；本次完成的是公共接口修复与测试，
不是三位学生的模型、策略或正式实验。** 默认使用 G1 + 双 Robotiq 2F-85、右臂操作、weld attachment。
课程明确允许 grasp 或 attach，因此无需先解决 experimental physical grasp 才能开展 Task 2–4。

## 1. 本次修复和开发边界

| 修复 | 现在的行为 | 代码 / 回归测试 |
| --- | --- | --- |
| B 输出目标 ID，C 执行时缺少坐标 | `resolve_action_position(action, scene)` 用当前感知的精确 ID 解析坐标；mock executor 已接入。`params.pos` 保留为显式 waypoint override；校验器检查坐标并拒绝没有 3D/坐标的 PLACE | `core/action_targets.py`、`core/validation.py`；`test_id_only_plan_validates_and_executes` |
| STOP 后机械臂仍继续旧轨迹 | `env.stop_motion()` 取消底盘、腰部、机械臂和夹爪的运动目标，保持测得姿态；保留 attachment。所有 episode 终止路径都会调用它，停止失败记录为 ERROR | `core/env.py`、`core/g1.py`、`core/orchestrator.py`；`test_stop_cancels_motion_without_teleport_and_can_resume`、`test_refusal_also_stops_an_active_reach` |
| 最终验证可能误报成功 | 必须是精确 object/region ID、已释放、有当前 3D 证据、处于区域范围和支撑高度内；相隔 0.2 模拟秒复查，位移不超过 2 cm。计划内 VERIFY 的成功不能替代最终检查 | `core/verification.py`；悬空、越界、错误 ID、旧帧、漂移和虚假 VERIFY 回归测试 |

完整修复测试在 `tests/test_student_handoff.py`。STOP 是**非阻塞命令**，不会把 qpos/qvel 清零，
也不会推进时间；之后需要 `env.step()` 让物理系统减速。它不会自动放下已抓取物体。
physical 模式已持物时保留右夹爪的夹持命令，其他夹爪保持测得开度。

本次升到 contract v2，以区分新的停止和最终验证行为。旧 dataclass 构造方式仍可用；
新增 `GroundedObject.region_half_extents_xy` 是可选字段。自定义环境适配器应实现公共
`RobotEnvProtocol`，尤其是 `stop_motion()`、`timestep()` 和观测/本体状态方法。
旧 v1 实验记录保留原版本，不应直接和 v2 的成功率混算。

三位学生可使用 `core/types.py`、`core/interfaces.py`、`core/validation.py`、
`core/action_targets.py`、`core/verification.py`、`core/skills.py`、`core/llm_client.py`。
学生模块之间不要互相 import；不要在学生实现中导入 `core.oracle`、`core.mocks`，
或绕过 RobotEnv 读取 `SimWorld`/MuJoCo 的物体真值、调用 teleport。
真值仅供 `eval/`、测试和 backbone mocks 使用。

## 2. 先运行公共基线

下列命令在项目目录执行；已有 `.venv` 时无需重新安装。

```bash
cd /home/jiamo/EE4705/project1.3

# 新机器需要时安装（Python >= 3.10）
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
bash scripts/fetch_menagerie.sh

# 离线检查：接口、相机、真实仿真控制、错误处理
.venv/bin/python -m pytest -q

# 本次三项修复的集中回归
.venv/bin/python -m pytest -q tests/test_student_handoff.py

# 5 个基础连通性场景：全部使用 mocks
.venv/bin/python -m eval.runner --mode e2e --trials eval/trials/smoke --mock-all
```

通过条件：pytest 无失败；smoke 输出 `Trials: 5  passed: 5  failed: 0`，退出码 0。
physical grasp 有一项明确标记为 experimental 的 skip；这不代表物理夹持已通过。
这些测试不访问付费模型 API。

本次完整验证为 **121 passed、1 experimental skip，smoke 5/5**；
原始输出和适用范围见 [验证记录](validation/student_handoff/REPORT.md)。

| 模式 | A | B | C | 用途 |
| --- | --- | --- | --- | --- |
| `grounding` | 真实 | RulePlanner | TeleportExecutor | 检查 A 能否接入流程 |
| `planning` | GTPerception | 真实 | TeleportExecutor | 检查 B 能否接入流程 |
| `manipulation` | GTPerception | RulePlanner | 真实 | 检查 C 在受控感知/计划下执行 |
| `e2e` | 真实 | 真实 | 真实 | 全系统集成 |

四种模式均运行完整 Orchestrator，
**不是四个独立评分器**。任何模式加 `--mock-all` 都会覆盖三个真实模块。
不加该参数而所需模块仍是 stub 时，runner 会报未实现并退出码 2，不会偷偷换成 mock。
含有任何 mock 的记录都标为 `infrastructure_check: true`；做组件实验时应明确哪些模块真实、哪些是测试替身。

## 3. 三人共享的 input/output

数据流：`Observation → A: SceneDescription → B: Plan → C: ExecutionResult → 新 Observation`。
失败结果和新 scene 会进入 B 的 `replan()`；最终 success 由 Orchestrator 独立调用感知验证后决定。

### 相机和坐标

| 输入字段 / API | 格式与含义 |
| --- | --- |
| `env.get_obs(camera="onboard")` | `Observation`；默认 onboard 是 head 的别名 |
| `obs.rgb` | `uint8[480,640,3]`，RGB |
| `obs.depth` | `float32[480,640]`，单位 m，沿相机光轴的 Z 深度；无效/背景为 NaN |
| `obs.intrinsics` | `float64[3,3]`，针孔相机 K |
| `obs.t_world_camera` | `float64[4,4]`，相机坐标到世界坐标变换 |
| `frame_id / sim_time / camera_name` | 当前观测身份、模拟秒数、相机名；感知输出必须携带对应值 |
| `env.get_obs_multi(["head", "left_wrist", "right_wrist"])` | 同一物理时刻的多视角 dict；相同 capture 批次前缀和 sim_time，不同 frame_id |
| `env.get_robot_state()` | base_pose `(x,y,yaw)`、TCP 世界位姿、左右夹爪测得开度、attached 等本体反馈；没有场景物体真值 |

世界坐标为右手系、Z 向上、位置单位 m、角度 rad。公共相机坐标为 +X 向右、+Y 向下、+Z 向前。
像素 `(u,v)` 的有效光轴深度为 `d` 时：

```text
p_camera = d * inv(K) @ [u, v, 1]
p_world  = (T_world_camera @ [p_camera.x, p_camera.y, p_camera.z, 1])[:3]
```

不要把包围盒中心的单个深度无条件当成物体中心：它可能落在背景、边缘或前表面。
A 应过滤无效深度、使用物体内部证据，并说明如何从表面估计物体中心。

### 感知对象

`SceneDescription` 含 `objects`、`regions`、`caption`、`ambiguities`、`frame_id`、`sim_time`。
每个 `GroundedObject` 至少正确填写以下字段：

| 字段 | 约定 |
| --- | --- |
| `instance_id` | A 分配的 episode 内稳定 ID，如 `p0`；换视角、物体移动后仍跟踪同一实例，reset 后重建。不同于 MuJoCo body/GT ID |
| `name` | `assets/objects.yaml` 中的标准类别；两个石头可同为 `stone`，但必须有不同 instance_id |
| `kind` | `object` 或 `region` |
| `bbox_xyxy` | 原图像素 `(xmin,ymin,xmax,ymax)`；缩放/裁剪推理后先换回原图坐标 |
| `pos_world` | object 的估计中心；region 的**支撑面中心点**，不是 region 薄片几何中心，也不是空中的搬运点 |
| `status` | `LOCALIZED` 有可靠 3D；`UNLOCALIZED` 看见但没有可靠 3D；`AMBIGUOUS` 无法区分候选；`NOT_FOUND` 没有视觉证据 |
| `confidence / source / frame_id` | 0–1 置信度、来源标签、当前观测 frame_id |
| `region_half_extents_xy` | 可选、仅 region：世界坐标中轴对齐区域的 XY 半宽，单位 m，必须为两个正数 |

可见对象不等于可抓取：`UNLOCALIZED` 可以用于回答“看到了什么”，不能直接 REACH/GRASP。
未看见目标时 `ground()` 返回 `None`；多个相似目标无法消歧时返回 `AMBIGUOUS`，不要随意选一个。
精确目标在新帧丢失时，报告丢失或重新观察，不要拿另一个同名对象继承旧 ID。

已知固定 `red_region` 不填半宽时，验证器使用 `assets/objects.yaml` 的类别先验 `(0.08,0.08)` m。
它不读取 region 真值位置。区域改变尺寸或新增类别时，必须由感知/公开场景定义提供正确半宽；
未知区域尺寸会验证失败。当前几何判断只支持轴对齐区域。

### 动作

| Skill | `target` / `params` input | 执行输出应表示什么 |
| --- | --- | --- |
| SEARCH | target 是类别名/搜索词，如 `stone`；status 为 NEEDS_SEARCH | 找到可用视觉证据；或 SEARCH_NOT_FOUND / SEARCH_FATAL |
| APPROACH | 已定位 instance_id，或 `params.pos=[x,y,z]` | 底盘到达可操作位置 |
| REACH | 已定位 object/region ID；可显式提供 pos | TCP 到达目标附近，或 UNREACHABLE/TIMEOUT |
| GRASP | 已定位 object ID；手为空 | attachment 成功；否则 GRASP_MISSED 等。ID 不是 attachment handle |
| MOVE_TO | 已定位 ID 或显式 pos | 到达运输目标；ID 指向 region 时默认取支撑点 + 0.18 m |
| PLACE | region ID；已持物；建议 `params.object` 为所持 object ID | 移至释放位置、释放、检查放置；region 默认坐标为支撑点 |
| VERIFY | `params.condition` 是 holding / object_visible / object_in_region | 该条件检查结果。object_visible 使用 target ID；object_in_region 必须给 params.object 与 params.region |
| STOP | 无 params；若出现在计划中必须最后 | 取消运动、保持姿态，不自动释放物体 |

除 SEARCH 外，引用对象时用 perceived ID。显式 `params.pos` 是世界坐标，并优先于 ID 解析位置；
B 通常只输出 ID，让 C 用新观测更新位置。显式坐标适合已定义的 waypoint，使用前由 C 检查是否过期/可达。
`resolve_action_position` 不检查机器人持物状态，也不替代 `validate_plan` 或动作后的反馈检查。

## 4. Student A — Task 2：VLM / visual AI 感知和 grounding

**修改入口：** `perception/student_a.py`，保留类名 `StudentAPerception`。

| 方法 | Input | Output |
| --- | --- | --- |
| `describe(obs, query=None)` | 一张带标定的 RGB-D Observation；可选自然语言问题 | 当前帧 SceneDescription；caption 可回答 VQA，objects/regions 供 B/C 使用 |
| `ground(obs, target)` | 当前 Observation + 类别或自然语言描述 | GroundedObject / None；应与 describe 使用同一套实例跟踪 |
| `reset()` | 新 episode 开始 | 清空实例跟踪、缓存的场景关联，避免把上一局 p0 当成当前物体 |

开发顺序：

1. 接上 VLM 或等价视觉 AI，完成结构化检测、scene description 和 VQA；保存 prompt、raw response、解析失败。
2. 把模型 bbox 映射到原图像素，通过 RGB-D/标定估计 3D；处理 NaN 和置信度不足。
3. 维护对象/区域 ID，支持同类消歧与不可见目标；让新观测更新位置而不改变身份。
4. 实现上面的两个接口和 reset；真实功能完成后设置 `IMPLEMENTED = True`。

**具体 input/output 例子（数值仅示意，不是固定答案）：** 输入是 head 图像和
`"Where is the stone?"`；输出 `GroundedObject(instance_id="p0", name="stone",
status=LOCALIZED, bbox_xyxy=(280,220,315,255), pos_world=(0.40,-0.15,0.88),
confidence=0.91, source="vlm", frame_id=obs.frame_id)`。
如果 RGB 看到了石头但对应深度不可用，应输出 UNLOCALIZED 和 `pos_world=None`。

**测试方法：**

- 接口/几何准备：`.venv/bin/python -m pytest -q tests/test_env.py tests/test_cam_sync.py tests/test_llm_client.py`。
  这些检查 backbone，不是 A 的模型正确率。
- A 自己的离线单元测试应保存真实 RGB-D fixtures，并注入固定模型响应，覆盖 bbox 坐标换算、深度 NaN、两个同类目标、不可见目标、跨帧 ID、reset、非法 JSON。
- 完成后跑 `.venv/bin/python -m eval.runner --mode grounding --trials eval/trials/smoke`，确认能接入已有 B/C mocks。
- 正式评测至少 **20 个 grounding trials**。建议 12 个可见且唯一目标（变化物体、位置、视角），
  4 个相似目标/属性消歧，4 个不可见目标；额外演示 scene description 和 VQA。
  输入只给图像、标定和 query，不能给 expected GT ID/位置。
- 独立评测应在 evaluation 脚本中直接调用 A。同一仿真时刻调用
  `oracle.associate(scene, camera=obs.camera_name)`，依据 bbox IoU 将 perceived ID 与 GT ID 配对，
  再比较 expected target。associate 当前只关联 objects；region 的 bbox/支撑点需单独比较评测标签。
  感知与关联之间不要 `env.step()`，否则真值时刻不同会报错。

建议输出 `grounding.csv`，每行记录
`trial_id, seed, camera, frame_id, query, expected_status, expected_gt_id, predicted_id,
predicted_status, associated_gt_id, bbox_iou, position_error_m, correct, latency_s`，并保存原图和预测叠图。
明确采用的 Grounding Accuracy 定义：例如“唯一可定位目标中，正确选中目标的 trials / 该类 trials”；
将不可见检测、消歧处理正确率和 3D 误差另外报告，避免把拒绝全部目标算成高准确率。
oracle 的默认匹配门槛为 IoU ≥ 0.30；这是本项目关联阈值，不是课程规定。

## 5. Student B — Task 3：语言理解和 action planning

**修改入口：** `planner/student_b.py`，保留类名 `StudentBPlanner`。

| 方法 | Input | Output |
| --- | --- | --- |
| `plan(instruction, scene)` | 原始自然语言 + A 的 SceneDescription | Plan |
| `replan(instruction, scene, history, context, clarification=None)` | 新 scene、历史 ExecutionResult、held_instance_id 等当前状态、可选澄清回答 | 替换剩余动作的新 Plan；不能盲目从 GRASP 重启 |
| `reset()` | episode 开始 | 清空本局记忆 |

接入 LLM/VLM/VLA 的结构化输出并转换成 `core.types`，不要只交付 RulePlanner 的关键词规则。
`validate_plan(plan, ExecutionContext(scene)) == []` 仅说明结构和前置条件允许尝试，
不能说明自然语言理解正确或机械臂实际可达。

**具体 input/output：** 输入 instruction 为 `"move the stone onto the red region"`，
scene 中 `p0` 是已定位 stone，`p1` 是已定位 red_region，当前没有持物。模型 JSON 可设计成：

```json
{
  "status": "READY",
  "actions": [
    {"skill": "APPROACH", "target": "p0", "params": {}},
    {"skill": "GRASP", "target": "p0", "params": {}},
    {"skill": "MOVE_TO", "target": "p1", "params": {}},
    {"skill": "PLACE", "target": "p1", "params": {"object": "p0"}},
    {"skill": "VERIFY", "params": {"condition": "object_in_region", "object": "p0", "region": "p1"}},
    {"skill": "STOP", "params": {}}
  ]
}
```

这是一份建议的模型输出 schema；B 负责解析成 `Plan`/`Action`，把字符串转为对应枚举。
字段缺失、未知技能、非法坐标、额外说明文字等都需要有界错误处理。

| 场景 | 应有状态和行为 |
| --- | --- |
| 目标唯一、已定位、请求支持 | READY + 有序动作 |
| 目标不在当前视野 | NEEDS_SEARCH + SEARCH；搜索成功后基于新 scene 重规划 |
| 两个 stone，指令未区分 | NEEDS_CLARIFICATION + 非空 clarification_question，actions 为空 |
| 不支持/明确不可能的请求 | INFEASIBLE + reason，actions 为空 |
| GRASP 已成功，但 MOVE_TO 失败 | replan 使用 context 中已持物 ID，不能重复 GRASP |

**测试方法：**

- 规则准备：`.venv/bin/python -m pytest -q tests/test_validation.py tests/test_llm_client.py tests/test_student_handoff.py`。
- 独立评测直接调用 B，用手工标注的 `SceneDescription` 和 `ExecutionContext` fixtures；无需渲染和 C。
  对每个 case 先检查 schema/validate_plan，再检查语义：选对 object/region、步骤依赖正确、状态正确、遵守已持物上下文。
  不要求所有正确计划逐 token 或逐 action 完全相同。
- 至少 **20 条不同自然语言指令**；可分 10 个同义改写/有效请求、4 个非法或不可行请求、
  3 个需搜索、3 个需澄清。另测 GRASP 失败、运输时丢失目标、已持物状态下的 replan。
- 完成后跑 `.venv/bin/python -m eval.runner --mode planning --trials eval/trials/smoke`。

建议输出 `planning.jsonl`，每条包含
`case_id, instruction, input_scene, input_context, raw_response, parsed_plan,
validation_errors, expected_semantics, semantic_correct, latency_s, model_calls`。
Action Planning Accuracy = 同时结构有效且语义符合预先标注要求的条数 / 全部评测指令数。
解析失败、遗漏必要步骤、选错实例都计失败；单独汇报拒绝/搜索/澄清子集表现。

## 6. Student C — Task 4：真实仿真执行、反馈与恢复

**修改入口：** `executor/student_c.py`，保留类名 `StudentCExecutor`。

`execute(action, env, perception) -> ExecutionResult`：input 是**一个 Action**、公共 RobotEnv 和 A 接口；
output 是该动作的成功与否、`ErrorCode`、`recovery_attempted`、动作后的 `post_frame_id` 和诊断 info。
`execute` 可以内部按小步控制和观察，但需要有模拟时间上限；返回成功应基于该动作的后置条件。

建议实现顺序：

1. 用 fresh `perception.describe(env.get_obs())` 和 `resolve_action_position()` 解析 ID；
   丢失/歧义目标时返回 TARGET_LOST，让上层更新场景或重规划。
2. 复用 `core.skills.approach/reach/grasp/move_to/place/search` 建立第一版；
   这些是公开接口上的参考控制器，不要求重写 IK。**TeleportExecutor 是评测替身，不能作为真实 C。**
3. GRASP 后检查 `env.is_attached()` 和本体/视觉反馈；必要时做小幅 lift check。
   MOVE_TO 后检查到位且仍持物，避免把运输中掉落计成成功。
4. PLACE 需要 C 自己完成安全下降、释放和反馈。`resolve_action_position(PLACE, scene)` 给的是支撑点；
   **不要直接把 TCP 压到桌面 Z。** 根据物体尺寸、抓取偏置选择 TCP 释放高度/下降距离。
   `core.skills.place(env)` 只负责 detach + settle，并不接受目标位置，也不保证物体落在区域里。
5. VERIFY object_in_region 可以复用 `verify_placement(env, perception, object_id, region_id)`；
   STOP 调用 `env.stop_motion()` 并在需要时 step/检查减速。动作后的观察 frame_id 要写入结果。
6. 加至少一种有界恢复/失败报告，例如重新观察后重试 GRASP 一次，仍失败就报告 GRASP_MISSED；
   或 SEARCH_NOT_FOUND 后由 Orchestrator 终止。把本地重试与上层重试次数一起计算，避免无限循环。

`SkillResult` 不能直接返回给 Orchestrator，需转换。例如 primitive 返回成功后：

```python
post = env.get_obs()
result = ExecutionResult(
    action=action,
    success=primitive_result.success,
    error_code=primitive_result.error_code,
    post_frame_id=post.frame_id,
    info=primitive_result.info,
)
```

**具体 input/output：** 输入 `Action(Skill.GRASP, target="p0")`；在确认 attachment 后，
输出 `ExecutionResult(action=action, success=True, error_code=ErrorCode.NONE,
post_frame_id=post.frame_id, info={"attachment": opaque_handle})`。
如果目标未进入夹持范围，则 success=False、error_code=GRASP_MISSED；不能因为发出了 close 命令就返回成功。

**测试方法：**

- 仿真控制准备：`.venv/bin/python -m pytest -q tests/test_g1_control.py tests/test_gripper.py tests/test_student_handoff.py`。
- 可达空间检查：`.venv/bin/python scripts/ik_reach_test.py`，读取 `runs/ik_reach/` 下 CSV 和图；
  先在已验证工作空间完成单物体循环，再扩展底盘重定位和视角变化。
- 单动作测试可在 `tests/` 中注入 GTPerception 和确定 Action，真实控制由 StudentCExecutor 完成；
  oracle 只放在测试脚本，用于检查实际被抓取的对象和最终位置。
- 至少 **10 个不同初始 robot/object 配置**。逐阶段记录 approach、grasp、transport、place、stop。
  另加入至少一个恢复/失败 case：不可达点、空抓、目标丢失，或执行途中 STOP。
- 完成后跑 `.venv/bin/python -m eval.runner --mode manipulation --trials eval/trials/smoke`。
  扩展 trial YAML 目录后，把 `--trials` 指向该目录即可。

建议输出 `manipulation.csv`，每次尝试一行，包含
`trial_id, seed, robot_init, target_gt_id, grasp_mode, skill, attempt, claimed_success,
actual_grasp, actual_place, placement_xy_error_m, support_height_error_m, failure_code,
recovery_attempted, sim_duration_s, wall_duration_s`。

- Grasp Success Rate：正确物体实际抓取成功次数 / 实际 GRASP 尝试次数；错误物体不算成功。
- Place Success Rate：正确释放并通过区域/稳定检查的次数 / 实际 PLACE 尝试次数；
  同时报告完整 manipulation trials 成功率，避免因前面抓取失败而没有 PLACE 尝试导致分母失真。
- 位置误差可报告物体中心到 region 支撑中心的 XY 欧氏距离，并另列高度偏差；
  也可采用“是否进入区域”的等价 placement metric。说明失败样本如何计入，零分母用 null/N/A。

当前默认 weld 模式足以完成课程基线。physical 模式的旧测量为 stone 0/10、cube 7/10、bottle 1/10，
仅是夹爪专项试验；本次没有把它改成合格物理夹持，也不要混入上述 C 的 weld 结果。

## 7. A/B 的模型配置与可复现性

公共 `LLMClient` 已提供 `call_vlm(image_png_bytes, prompt, system, json_schema)` 和
`call_llm(prompt, system, json_schema)`，返回 `LLMResponse`，包含 parsed/raw、缓存标记、tokens 和 latency。
RGB ndarray 要先用 Pillow 编码成 PNG bytes 才能传给 call_vlm。
配置必须显式给 `LLMConfig(model, base_url, api_key, ...)`；当前 runner 没有 `--model` 或 `--base-url` 参数。

runner 当前**无参数构造**学生类。A/B 可设计 `__init__(client=None)` 供单元测试注入，
默认分支通过自己明确记录的配置构造 client；或者由 ALL 在后续统一扩展 runner 配置工厂。
不要提交 API key。配置缺失应有明确错误，不能静默退回 GT/规则实现。

用 `FakeTransport` 写解析、错误恢复和缓存单元测试；它不验证模型质量。
真实 20-trial 评测要记录模型版本、prompt/schema、图像预处理、生成参数、seed、请求数、
latency 和缓存使用情况。缓存命中的耗时不能冒充 live API latency。

## 8. 联调输出、正式评测与提交

先分别完成 A/B/C 的组件评测和对应混合模式，再运行真实 `e2e`，不要给正式命令加 `--mock-all`。
每个类真实实现后才置 `IMPLEMENTED = True`。

当前 runner 输出到 `runs/<timestamp>_<mode>/`，包括：

| 文件 | Output 内容 |
| --- | --- |
| `run_meta.json` | mode、contract_version、mock 标记、trial 来源；具体替换组合以各 trial_record 的 module_config / infrastructure_check 为准 |
| `<trial_id>/trial_record.json` | module_config、instruction、outcome、claimed_success、actual_success、动作/验证事件、澄清、真值抓取记录、timings |
| `<trial_id>/frames/` | 已标记观察的 RGB、depth.npy、相机标定/时间戳元数据 |
| `metrics.json` | 当前已有的实际/宣称成功率、false claims、wrong object、refusal/clarification 指标 |

可用 `.venv/bin/python -m eval.metrics runs/实际运行目录名` 重新查看聚合指标。
`CLAIMED_SUCCESS` 是系统的结论；`actual_success` 是 evaluator 使用真值独立判断的结论。
当前实际成功要求抓对物体、已释放、中心处于区域 XY 内、高度相对支撑面处于
`[-0.005, 0.12]` m，并通过 2 秒稳定性检查（位移 ≤ 2 cm、速度 ≤ 0.05 m/s）。
系统侧的 0.2 秒感知检查受 A 的准确性和可见性限制，不能代替这个较长的真值评测。

**正式 Task 5 还需要 ALL 一起补齐以下评测工作；它们不是本次三项修复已经交付的功能：**

1. 至少 **20 个 randomized E2E trials**，变化 target object、object position、robot init、target area 和 instruction phrasing。
   现有 `eval/trials/smoke` 只有 5 个。`SceneConfig` 当前只配置 objects、robot_init、seed，
   尚不能通过 YAML 改 target region；需要在 ALL 的场景层增加区域配置并贯通 reset / oracle / 感知定义，
   或提供明确选择的不同场景文件。不要只改 expected.region 标签而不改变实际场景。
2. 实现/保存本指南给出的独立 A/B/C 评分记录，并在 eval 侧补充 Grounding Accuracy、
   Planning Accuracy、Grasp/Place/Manipulation Success、平均完成时间和各变化子集统计。
   **当前 eval.metrics 不会自动产生全部课程指标**；三个 mode 的 E2E success 不能分别充当 A/B/C 准确率。
   计时应区分模拟秒与真实墙钟，完成时间说明是否仅统计成功任务；timeout 单独列出。
3. 共同提交源代码/模型配置/场景/README/实验记录，标记学生贡献，记录固定依赖与 commit。
   写 Task 1 的文献综述（≤2 页）、环境和接口说明；准备 3–5 分钟不剪辑演示，
   至少包含成功任务和一次搜索、重试、澄清或失败报告。

A/B/C 可以立即开始 Task 2/3/4；区域随机化和完整评分器应在准备正式 Task 5 数据集时补齐。
学生的独立模型/执行工作、真实 API 评测和上述课程实验仍需实际完成并报告结果。
