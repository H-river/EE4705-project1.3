# Owner: backbone demo (ALL), replaceable by Student B
"""Small, explicitly labeled rule planner for an offline workflow demo."""
import re

from core.interfaces import Planner
from core.types import Action, GroundStatus, Plan, PlanStatus, Skill
from core.vocab import Vocab


class DemoPlanner(Planner):
    LABEL = "B: finite instruction rules (offline demo, no LLM)"

    def __init__(self, on_plan=None):
        self.vocab, self.on_plan = Vocab(), on_plan

    def plan(self, instruction, scene):
        return self._make(instruction, scene, None, None)

    def replan(self, instruction, scene, history, context, clarification=None):
        return self._make(instruction, scene, context.held_instance_id, clarification)

    def _make(self, instruction, scene, held, clarification):
        obj = self.vocab.find_phrase(instruction, "object")
        dest = self.vocab.find_phrase(instruction, "region")
        text = instruction.lower()
        if (not obj or not dest or not re.search(r"\b(move|put|place|bring|pick|take)\b", text)
                or re.search(r"\b(do not|don't|never|avoid)\b", text)):
            plan = Plan(status=PlanStatus.INFEASIBLE, reason="Demo supports moving a known object to the red area.")
        else:
            color = self.vocab.color_in_text(clarification) if clarification else obj[1]
            candidates = [g for g in scene.objects if g.name == obj[0]
                          and (not color or g.attributes.get("color") == color)]
            region = next((g for g in scene.regions if g.name == dest[0]), None)
            if not candidates and not held:
                plan = Plan([Action(Skill.SEARCH, obj[0])], PlanStatus.NEEDS_SEARCH)
            elif len(candidates) > 1 and not held:
                plan = Plan(status=PlanStatus.NEEDS_CLARIFICATION,
                            clarification_question="Which stone: the gray one or the dark red one?")
            elif region is None or region.status is not GroundStatus.LOCALIZED:
                plan = Plan([Action(Skill.SEARCH, dest[0])], PlanStatus.NEEDS_SEARCH)
            elif not held and candidates[0].status is not GroundStatus.LOCALIZED:
                plan = Plan(status=PlanStatus.INFEASIBLE, reason="Visible target has no reliable 3D position.")
            else:
                target = held or candidates[0].instance_id
                actions = []
                if not held:
                    pos = list(candidates[0].pos_world)
                    actions += [Action(Skill.APPROACH, target, {"pos": pos}),
                                Action(Skill.GRASP, target, {"pos": pos})]
                rp = list(region.pos_world)
                actions += [Action(Skill.MOVE_TO, region.instance_id, {"pos": [*rp[:2], rp[2]+.18]}),
                            Action(Skill.PLACE, region.instance_id, {"object": target, "pos": rp}),
                            Action(Skill.VERIFY, params={"condition": "object_in_region", "object": target,
                                                       "region": region.instance_id}), Action(Skill.STOP)]
                plan = Plan(actions)
        if self.on_plan:
            self.on_plan(instruction, scene, plan)
        return plan
