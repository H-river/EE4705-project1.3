"""Optional student modules, with observers around the unchanged contracts."""
from core.interfaces import Executor, Perception, Planner
from core.types import SceneDescription
from executor.demo_skills import DemoExecutor
from perception.demo_rgbd import RGBDPerception
from planner.demo_rules import DemoPlanner


def load_components(students=(), retry=False):
    if retry and "C" in students:
        raise ValueError("The injected retry scenario requires DemoExecutor; do not combine it with --student C")
    from executor.student_c import StudentCExecutor
    from perception.student_a import StudentAPerception
    from planner.student_b import StudentBPlanner
    student_classes = {"A": StudentAPerception, "B": StudentBPlanner, "C": StudentCExecutor}
    for role in students:
        if not student_classes[role].IMPLEMENTED:
            raise ValueError(f"Student {role} is still a stub. Implement its methods and set IMPLEMENTED = True first.")
    components = {"A": RGBDPerception(), "B": DemoPlanner(), "C": DemoExecutor(fail_first_grasp=retry)}
    for role in students:
        components[role] = student_classes[role]()
    labels = {role: (f"{role}: {type(module).__name__} (student implementation, model use not audited)"
                     if role in students else module.LABEL) for role, module in components.items()}
    return components, labels


class ObservedPerception(Perception):
    def __init__(self, inner, recorder, store):
        self.inner, self.recorder, self.store = inner, recorder, store

    def reset(self):
        self.inner.reset()

    def describe(self, obs, query=None):
        result = self.inner.describe(obs, query)
        self.store.mark_persist(obs.frame_id)
        self.recorder.on_scene(obs, result)
        return result

    def ground(self, obs, target):
        result = self.inner.ground(obs, target)
        self.store.mark_persist(obs.frame_id)
        self.recorder.event("A.ground", {"frame_id": obs.frame_id, "target": target, "result": result})
        self.recorder.obs = obs
        self.recorder.scene = SceneDescription(
            objects=[result] if result and result.kind != "region" else [],
            regions=[result] if result and result.kind == "region" else [],
            frame_id=obs.frame_id, sim_time=obs.sim_time, caption="Single-target grounding result")
        return result


class ObservedPlanner(Planner):
    def __init__(self, inner, recorder):
        self.inner, self.recorder = inner, recorder

    def reset(self):
        self.inner.reset()

    def plan(self, instruction, scene):
        result = self.inner.plan(instruction, scene)
        self.recorder.on_plan(instruction, scene, result)
        return result

    def replan(self, instruction, scene, history, context, clarification=None):
        result = self.inner.replan(instruction, scene, history, context, clarification)
        self.recorder.on_plan(instruction, scene, result)
        return result


class ObservedExecutor(Executor):
    def __init__(self, inner, recorder):
        self.inner, self.recorder = inner, recorder

    def reset(self):
        self.inner.reset()

    def execute(self, action, env, perception):
        self.recorder.on_action("start", action, None)
        result = self.inner.execute(action, env, perception)
        self.recorder.on_action("end", action, result)
        return result
