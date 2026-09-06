# EE4705 Project 1.3 — Backbone 设计方案(High-Level Plan)

> 目标:先冻结模块间契约,让 Student A/B/C 各自对接口开发,集成只是"换实现"。
> 平台假设:MuJoCo(Track 1)+ 云端 VLM/LLM API。其它平台只需替换 `core/env.py`。

---

## 0. 总览:backbone 包含什么、不包含什么

| 层 | 内容 | 谁维护 | 何时冻结 |
|---|---|---|---|
| **契约层** | `types.py` 数据类型、`interfaces.py` 三个抽象接口 | 全员(改动需三人同意) | Week 1 末 |
| **环境层** | `env.py` MuJoCo 场景封装、`skills.py` 底层运动原语签名 | Task 1 全员;后续由 C 主要维护 | Week 1 末 |
| **控制层** | `orchestrator.py` 闭环主循环 | 全员(Task 5) | Week 1 骨架,Week 4 定稿 |
| **评估层** | `eval/` trial 格式、runner、指标、日志 | 全员(建议一人牵头) | Week 1 骨架 |
| **Mock 层** | `mocks.py` 三个接口的真值实现 | Task 1 全员 | Week 1 末 |
| 三个实现模块 | `perception/`、`planner/`、`executor/` | A / B / C 各自 | 不属于 backbone |

**设计原则**:模块之间只通过 `types.py` 里的 dataclass 传数据,不直接 import 对方的代码;所有模块都能被 mock 替换;所有 trial 都通过同一个 runner 跑、同一个 logger 记。

---

## 1. 契约层 `core/types.py` + `core/interfaces.py`

### 1.1 数据类型(按数据流方向)

```
env ──Observation──▶ Perception ──SceneDescription──▶ Planner ──Plan──▶ Executor ──ExecutionResult──▶ Orchestrator
```

| 类型 | 生产者 → 消费者 | 关键字段 | 设计要点 |
|---|---|---|---|
| `Observation` | env → Perception / Executor | `rgb`(HxWx3)、可选 `depth`、`cam_intrinsics`、`cam_extrinsics`、`sim_time` | 带相机位姿,A 才能把 2D bbox 反投影成 3D |
| `GroundedObject` | Perception 输出 | `name`(归一化名)、`bbox_xyxy`、`pos_world`(可 None)、`confidence`、`source`("vlm"/"gt") | `source` 字段让 mock 和真实现共用同一类型 |
| `SceneDescription` | Perception → Planner | `objects: list[GroundedObject]`、`regions: list[GroundedObject]`、`caption: str`、`ambiguities: list[str]` | `ambiguities` 承载"多个相似物体"这类失败信号 |
| `Action` | Planner 输出单元 | `skill`(枚举)、`target: str|None`、`params: dict` | skill 枚举固定 8 个:SEARCH / APPROACH / REACH / GRASP / MOVE_TO / PLACE / VERIFY / STOP |
| `Plan` | Planner → Executor | `actions: list[Action]`、`feasible: bool`、`reason: str|None`、`raw_llm_output: str` | `feasible=False` 用于拒绝不可行指令;保留原始输出便于失败分析 |
| `ExecutionResult` | Executor → Orchestrator | `action`、`success`、`error_code`(枚举)、`recovery_attempted`、`post_obs: Observation` | `error_code` 枚举化:GRASP_MISSED / TARGET_LOST / UNREACHABLE / TIMEOUT / … |
| `TrialRecord` | logger 输出 | 各阶段输入输出、每阶段耗时、API 调用次数、最终判定 | 一个 trial 一个 JSON 文件 |

### 1.2 三个接口

```python
class Perception(ABC):
    def describe(self, obs: Observation, query: str | None = None) -> SceneDescription
    def ground(self, obs: Observation, target: str) -> GroundedObject | None
    # ground 返回 None = 目标不可见,这是 A 的失败处理点

class Planner(ABC):
    def plan(self, instruction: str, scene: SceneDescription) -> Plan
    def replan(self, instruction, scene, history: list[ExecutionResult]) -> Plan
    # replan 是可选进阶;基础版可以直接调 plan

class Executor(ABC):
    def execute(self, action: Action, env: SimEnv, perception: Perception) -> ExecutionResult
    # 注入 perception 是为了 VERIFY / SEARCH 能用视觉;C 也可以只用 env 真值(需在报告里说明)
```

**冻结规则**:Week 1 结束时三人签字;之后改字段只能"加可选字段",不能删或改已有字段。

---

## 2. 环境层 `core/env.py` + `core/skills.py`

### 2.1 `SimEnv`(MuJoCo 封装)

| 方法 | 作用 | 备注 |
|---|---|---|
| `reset(config: SceneConfig) -> Observation` | 按 trial 配置摆放物体、机器人初始位姿、目标区域 | 随机化在 `SceneConfig` 里,不在 env 内 |
| `get_obs() -> Observation` | 渲染当前相机帧 | 固定分辨率(建议 640×480),避免 VLM 输入不一致 |
| `get_state() -> dict` | 所有物体/机器人真值位姿 | 只给 mock 和 eval 用,**Perception 实现不允许调用** |
| `step(ctrl, n)` | 推进物理 | 供 skills 调用 |
| `object_in_region(obj, region) -> bool` | 真值判定 | eval 用,也是 VERIFY 的 ground truth |
| `attach(obj)` / `detach()` | 简化抓取(weld constraint) | 基础版允许"吸附式"抓取;真物理抓取算加分 |

### 2.2 场景资产 `assets/`

- `scene.xml`:仿人机器人(建议 MuJoCo Menagerie 里的 Unitree G1/H1 或简化的上半身+底座)、桌面/地面、相机。
- 至少 3 个物体(stone / cube / bottle 等,颜色形状可区分)+ 至少 1 个目标区域(红色贴图平面)。建议再放 1 个"干扰物"(第二块相似石头),给 A 的多目标歧义、B 的澄清逻辑提供素材。
- 相机:头部相机(第一人称,随机器人动)+ 一个固定俯视相机(调试用)。

### 2.3 `skills.py`:运动原语签名

backbone 只定签名和返回类型;C 负责实现。签名统一 `-> SkillResult(success, error_code, info)`。

| 原语 | 签名 | 最低实现 |
|---|---|---|
| `approach(env, pos_world)` | 底座/身体移到目标附近 | 位置插值即可 |
| `reach(env, pos_world)` | 末端到达目标上方 | IK 或直接设关节 |
| `grasp(env, obj_name)` | 抓取 | weld attach + 距离校验 |
| `move_to(env, pos_world)` | 持物搬运 | 同 approach |
| `place(env, region_name)` | 放置 | detach + 落地 |
| `search(env, perception, target)` | 转头/转身扫视直到 `ground` 非 None | 这是 Task 4.iii 恢复行为之一 |

---

## 3. 控制层 `core/orchestrator.py`

主循环(全员共有),逻辑固定,但每一步的"判断"都委托给模块:

```
run(instruction):
  obs   = env.get_obs()
  scene = perception.describe(obs)                       # 阶段 1:感知
  plan  = planner.plan(instruction, scene)               # 阶段 2:规划
  if not plan.feasible: return report(plan.reason)       # B 的拒绝逻辑

  for action in plan.actions:                            # 阶段 3:逐动作执行
      result = executor.execute(action, env, perception)
      log(result)
      if not result.success:
          result = recover(action, result)               # 见下
          if not result.success: return report(fail)
  return verify_final(instruction)                       # 阶段 4:最终验证
```

**Recovery 策略分派**(保证每人的失败处理留在自己模块):

| 失败信号 | 处理者 | 动作 |
|---|---|---|
| `ground()` 返回 None | Perception (A) 产生,Executor (C) 消费 | 触发 `search` |
| `scene.ambiguities` 非空 | Planner (B) | 返回 `feasible=False, reason="clarify: which stone?"` |
| `GRASP_MISSED` | Executor (C) | 重试 ≤ N 次 |
| 重试耗尽 | Orchestrator | STOP + 报告 |

配置项:`max_retries`、`max_replans`、`verify_mode`("vision" / "gt" / "both")、超时。

---

## 4. 评估层 `eval/`

### 4.1 Trial 定义(YAML,一个 trial 一个文件或一个列表)

```yaml
id: e2e_017
category: instruction_variation        # standard / scene_variation / instruction_variation / robustness
instruction: "could you move that rock onto the red marker?"
scene:
  seed: 17
  objects: {stone: [0.4, 0.1, 0.02], cube: [0.2, -0.3, 0.02], bottle: [-0.1, 0.2, 0.05]}
  regions: {red_area: [0.6, 0.4]}
  robot_init: {x: 0, y: 0, yaw: 0}
  distractors: []                       # robustness 用:如再放一块 stone
expected:
  target: stone
  region: red_area
  feasible: true
  plan_skeleton: [SEARCH?, APPROACH, GRASP, MOVE_TO, PLACE, VERIFY]   # "?" 表示可选
```

### 4.2 Runner 与分项评估

同一个 `run_trials(trials, system, mode)`,`mode` 决定哪些模块是 mock:

| 评估目标 | 配置 | 判定依据 | 对应任务 |
|---|---|---|---|
| Target Grounding Accuracy | 真 A + mock B/C | `ground()` 输出 vs `env.get_state()` 真值(bbox IoU ≥ 0.5 或 3D 误差 ≤ 阈值) | Task 2.iv(≥20) |
| Action Planning Accuracy | mock A + 真 B + mock C | plan 与 `expected.plan_skeleton` 语义匹配 + `feasible` 一致 | Task 3.iv(≥20) |
| Manipulation Success Rate | mock A/B + 真 C | `env.object_in_region` + 位置误差 | Task 4.iv(≥10) |
| End-to-End Success | 全真 | 最终真值判定 + 各阶段判定 | Task 5.ii(≥20) |

**成功判据写死在 `eval/criteria.py`**,报告里直接引用——满足文档"state clearly how success is defined"的要求。

### 4.3 指标与日志

- `logger.py`:每个 trial 写一个 `TrialRecord` JSON + 每阶段的相机截图;自动累计 API 调用数、token、延迟。
- `metrics.py`:从 JSON 目录直接生成文档要求的六项指标表 + 按 category 分组的鲁棒性表 + 失败类型直方图(按 `error_code`)。报告里的表格直接由脚本输出,避免手填。

---

## 5. Mock 层 `core/mocks.py`

| Mock | 行为 | 用途 |
|---|---|---|
| `GTPerception` | 从 `env.get_state()` 读位姿,投影成 bbox,`confidence=1.0`,`source="gt"` | B/C 开发时用;同时是 A 的评估 ground truth |
| `RulePlanner` | 正则/关键词提取 target + region,输出固定动作骨架 | A/C 开发时用,不烧 API |
| `TeleportExecutor` | 直接把物体 set 到目标位置,所有动作 success | A/B 开发时用;也可注入 `fail_prob` 测 orchestrator 的 recovery 分支 |

Week 1 交付标准:`GTPerception + RulePlanner + TeleportExecutor` 跑通参考指令,录屏。

---

## 6. 时间线与责任

| 周 | Backbone 里程碑 | A | B | C |
|---|---|---|---|---|
| W1 | 场景 XML、`SimEnv`、`types/interfaces` 冻结、三个 mock、orchestrator 骨架、eval runner 骨架、mock pipeline 跑通 | 共同 | 共同 | 共同(主导 env/skills) |
| W2 | eval 的分项模式可用 | VLM 接入、`describe/ground` | prompt + JSON schema、`plan` | `approach/reach/grasp` 原语 |
| W3 | logger 全量记录、metrics 脚本 | 反投影 3D、歧义/不可见处理、20 trial | 改述鲁棒、拒绝逻辑、20 指令 | `place/search`、闭环校验、重试 |
| W4 | 全真集成、recovery 分派定稿 | 联调 | 联调 | 10 trial 执行评估 |
| W5 | 20 次 e2e、指标表、失败分析、视频、报告 | 共同 | 共同 | 共同 |

**贡献标注**:每人一个顶层目录;文件头写 `# Author: <name> (Student X)`;backbone 文件按 git blame 认定;报告各章节署名。

---

## 7. 几个提前定下来会省很多事的决定

1. **坐标系**:统一世界系(z 向上,单位米);Perception 返回 `pos_world`,不返回相机系坐标。
2. **物体命名**:`assets/objects.yaml` 维护"规范名 ↔ 同义词"表(stone/rock/pebble),A 的 grounding 和 B 的 planner 共用,避免两边各写一份。
3. **VLM/LLM 调用统一封装**:`core/llm_client.py` 提供 `call_vlm(image, prompt, json_schema)` / `call_llm(prompt, json_schema)`,自动计数、重试、缓存(相同输入直接返回,省钱且 trial 可复现)。A 和 B 都只调这层。
4. **随机种子**:所有随机化走 `SceneConfig.seed`,保证 20 次 trial 可复现、三人评估用同一批种子。
5. **抓取真实度**:基础版允许 weld 吸附;报告里如实说明,时间富余再换真物理抓取(可作为进阶亮点)。
