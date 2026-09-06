#!/usr/bin/env python
# Owner: backbone (ALL)
"""Generate assets/g1_2f85_ee4705.xml from the pinned upstream Menagerie files.

Composition (deterministic, run `python scripts/build_g1_model.py`):

* `unitree_g1/g1.xml` (29-DoF G1 WITHOUT hands) with the EE4705 platform
  adaptations: mocap base body + base weld, head camera, foot/world contact
  exclusion, upstream light/keyframe/rubber-hand stubs removed.
* `robotiq_2f85/2f85.xml` instantiated TWICE (prefixes `rg_` right, `lg_`
  left) on the wrist links, with every default class, asset, body, joint,
  geom, site, tendon, equality and actuator name prefixed so nothing
  collides between the two grippers or with the G1 file.  The 2F-85
  linkage constraints (connect ×2, joint coupling) and the upstream contact
  exclusions are kept; gripper-vs-own-forearm pairs are excluded in
  addition.  Wrist cameras are mounted on the gripper bases.

The flange transform, camera poses and the pad contact parameters are the
documented local decisions (assets/README.md).  The generated file is
committed so the model is usable without running this script.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
MENAGERIE = ROOT / "assets" / "menagerie"
G1_XML = MENAGERIE / "unitree_g1" / "g1.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"
OUT = ROOT / "assets" / "g1_2f85_ee4705.xml"

# ---------------------------------------------------------------- documented decisions
# Flange: gripper base_mount origin in the wrist_yaw link frame.  The G1
# wrist_yaw mesh ends at x = +0.042 (the upstream rubber-hand stub sat at
# x = 0.0415); the 7 mm mount plate starts at x = 0.045.  Rotation = cyclic
# axis permutation (quat 0.5 0.5 0.5 0.5): gripper +z (approach) -> link +x
# (forearm axis), gripper +x (pad closing axis) -> link +y, gripper +y ->
# link +z.  Same transform on both wrists (the G1 arm links are mirrored,
# so the grippers end up mirrored too).
FLANGE_POS = "0.045 0 0"
FLANGE_QUAT = "0.5 0.5 0.5 0.5"

# Wrist camera on the gripper `base` body (frame: +z approach, pads close
# along +-y of `base`).  Lens on the -x side of the base (base mesh half
# extent 0.0375 in x), 5.8 cm from the axis, near the mount plate, looking
# along +z tilted only 5 deg toward the axis: the open pads then sit in the
# bottom quarter of the frame and the CLOSED finger tips (on the axis,
# 0.13 m ahead) stay out of the central region; image "up" (camera +y)
# points away from the gripper.
WRIST_CAM_POS = "-0.058 0 0.02"
WRIST_CAM_XYAXES = "0 -1 0  -0.9962 0 0.0872"  # x right, y up (away from axis), looks along +z tilted 5 deg toward +x
WRIST_CAM_FOVY = "90"

# Pad contact tuning for physical grasping (upstream: friction 0.7/0.6,
# condim 3, solref 0.004, solimp 0.95 0.99 0.001, no margin).
PAD_FRICTION = "1.0 0.02 0.002"
PAD_CONDIM = "4"
PAD_SOLREF = "0.004 1"
PAD_SOLIMP = "0.95 0.99 0.001"
PAD_MARGIN = "0.0003"

HEAD_CAMERA = dict(name="head", pos="0.09 0 0.40", xyaxes="0 -1 0  0.5 0 0.86603", fovy="60")

GRIPPER_CLASS_MAP = {
    "2f85": "2f85", "driver": "2f85_driver", "follower": "2f85_follower", "spring_link": "2f85_spring_link",
    "coupler": "2f85_coupler", "visual": "2f85_visual", "collision": "2f85_collision",
    "pad_box1": "2f85_pad_box1", "pad_box2": "2f85_pad_box2",
}
GRIPPER_MESHES = ["base_mount", "base", "driver", "coupler", "follower", "pad", "silicone_pad", "spring_link"]
GRIPPER_MATERIALS = {"metal": "2f85_metal", "silicone": "2f85_silicone", "gray": "2f85_gray", "black": "2f85_black"}
NAMED_ATTRS = ("name", "joint", "joint1", "joint2", "body1", "body2", "tendon", "site")


def comment(text: str) -> ET.Element:
    return ET.Comment(f" EE4705: {text} ")


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def build_gripper(prefix: str, grip_root: ET.Element) -> tuple[ET.Element, list[ET.Element], list[ET.Element], ET.Element, ET.Element]:
    """Return (body tree, contact excludes, equalities, tendon, actuator) for one prefixed gripper."""
    body = copy.deepcopy(grip_root.find("worldbody").find("body"))  # base_mount

    def fix(el: ET.Element) -> None:
        for attr in NAMED_ATTRS:
            if attr in el.attrib:
                el.set(attr, prefix + el.get(attr))
        if "class" in el.attrib:
            el.set("class", GRIPPER_CLASS_MAP[el.get("class")])
        if "childclass" in el.attrib:
            el.set("childclass", GRIPPER_CLASS_MAP[el.get("childclass")])
        if el.tag == "geom" and "mesh" in el.attrib:
            el.set("mesh", "2f85_" + el.get("mesh"))
        if el.tag == "geom" and "material" in el.attrib:
            el.set("material", GRIPPER_MATERIALS[el.get("material")])
        for child in list(el):
            fix(child)

    fix(body)
    body.set("pos", FLANGE_POS)
    body.set("quat", FLANGE_QUAT)
    # unnamed collision/visual geoms get names so contact diagnostics are readable
    counter = [0]

    def name_geoms(el: ET.Element) -> None:
        for g in el.findall("geom"):
            if "name" not in g.attrib:
                g.set("name", f"{el.get('name')}_geom{counter[0]}")
                counter[0] += 1
        for child in el.findall("body"):
            name_geoms(child)

    name_geoms(body)
    # wrist camera on the gripper base body
    base = body.find("body")  # `base`
    assert base.get("name") == prefix + "base"
    cam = ET.SubElement(base, "camera", name=f"{'right' if prefix == 'rg_' else 'left'}_wrist",
                        pos=WRIST_CAM_POS, xyaxes=WRIST_CAM_XYAXES, fovy=WRIST_CAM_FOVY)
    base.insert(list(base).index(cam), comment(
        "wrist RGB-D camera on the gripper base: lens 5.8 cm off the approach axis (base mesh |x| <= 0.0375), "
        "looking along the approach axis tilted 5 deg toward it so the finger pads appear at the bottom of the frame"))
    base.remove(cam)
    base.append(cam)

    excludes = []
    for ex in grip_root.find("contact").findall("exclude"):
        e = copy.deepcopy(ex)
        fix(e)
        excludes.append(e)
    equalities = []
    for eq in grip_root.find("equality"):
        e = copy.deepcopy(eq)
        fix(e)
        e.set("name", f"{prefix}{eq.tag}_{len(equalities)}")
        equalities.append(e)
    tendon = copy.deepcopy(grip_root.find("tendon").find("fixed"))
    fix(tendon)
    actuator = copy.deepcopy(grip_root.find("actuator").find("general"))
    fix(actuator)
    return body, excludes, equalities, tendon, actuator


def main() -> int:
    g1 = ET.parse(G1_XML).getroot()
    grip = ET.parse(GRIPPER_XML).getroot()
    g1.set("model", "g1_29dof_2f85_ee4705")

    # ---- compiler / assets
    g1.find("compiler").set("meshdir", "menagerie/unitree_g1/assets")
    asset = g1.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("file") in ("left_rubber_hand.STL", "right_rubber_hand.STL"):
            asset.remove(mesh)  # hand stubs are replaced by the grippers
    asset.append(comment("Robotiq 2F-85 assets (menagerie/robotiq_2f85), names prefixed, scale applied explicitly"))
    for old, new in GRIPPER_MATERIALS.items():
        rgba = next(m.get("rgba") for m in grip.find("asset").findall("material") if m.get("name") == old)
        ET.SubElement(asset, "material", name=new, rgba=rgba)
    for mesh in GRIPPER_MESHES:
        ET.SubElement(asset, "mesh", name="2f85_" + mesh, file=f"../../robotiq_2f85/assets/{mesh}.stl",
                      scale="0.001 0.001 0.001")

    # ---- defaults: gripper classes (renamed) appended after the g1 class
    default = g1.find("default")
    ET.SubElement(default, "geom", conaffinity="3")
    gd = copy.deepcopy(grip.find("default").find("default"))  # class 2f85

    def rename_classes(el: ET.Element) -> None:
        if el.tag == "default" and "class" in el.attrib:
            el.set("class", GRIPPER_CLASS_MAP[el.get("class")])
        for child in list(el):
            if child.tag == "mesh":
                el.remove(child)  # scale is applied on the mesh assets themselves
            else:
                rename_classes(child)

    rename_classes(gd)
    # Bit 2 identifies grippers and permits upstream internal contacts.
    # Other physical geoms use bit 1, affinity 3. Welded objects temporarily
    # use affinity 1, retaining table/object contacts but excluding grippers.
    coll = next(d for d in gd.iter("default") if d.get("class") == "2f85_collision")
    coll.find("geom").set("contype", "2")
    coll.find("geom").set("conaffinity", "2")
    for pad_cls in ("2f85_pad_box1", "2f85_pad_box2"):
        node = next(d for d in gd.iter("default") if d.get("class") == pad_cls)
        geom = node.find("geom")
        geom.set("friction", PAD_FRICTION)
        geom.set("condim", PAD_CONDIM)
        geom.set("solref", PAD_SOLREF)
        geom.set("solimp", PAD_SOLIMP)
        geom.set("margin", PAD_MARGIN)
        geom.set("contype", "2")
        geom.set("conaffinity", "2")
    default.append(comment("Robotiq 2F-85 default classes (prefixed 2f85_*); pad contact parameters tuned for physical grasping"))
    default.append(gd)

    # ---- worldbody: mocap base, head camera, grippers
    wb = g1.find("worldbody")
    for light in wb.findall("light"):
        wb.remove(light)
    pelvis = wb.find("body")
    assert pelvis.get("name") == "pelvis"
    mocap = ET.Element("body", name="base_target", mocap="true", pos="0 0 0.793")
    ET.SubElement(mocap, "geom", type="box", size="0.02 0.02 0.02", contype="0", conaffinity="0", group="4", rgba="0 0.6 1 0.3")
    wb.insert(0, comment("world-child mocap body driving the free pelvis through base_weld (sliding-base approximation, no walking)"))
    wb.insert(1, mocap)
    torso = next(b for b in g1.iter("body") if b.get("name") == "torso_link")
    torso.append(comment("head RGB-D camera on the torso (the G1 head is a fixed mesh, no neck joint); 30 deg down, ~47 deg with the waist lean"))
    ET.SubElement(torso, "camera", **HEAD_CAMERA)
    all_excludes: list[ET.Element] = []
    all_eq: list[ET.Element] = []
    tendons: list[ET.Element] = []
    actuators: list[ET.Element] = []
    for side, prefix in (("right", "rg_"), ("left", "lg_")):
        wrist = next(b for b in g1.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
        for geom in list(wrist.findall("geom")):
            if geom.get("mesh", "").endswith("rubber_hand"):
                wrist.remove(geom)
        body, excludes, eqs, tendon, act = build_gripper(prefix, grip)
        wrist.append(comment(f"Robotiq 2F-85 ({prefix}) on the flange: pos {FLANGE_POS}, quat {FLANGE_QUAT} "
                             "(gripper +z approach -> link +x, pad closing axis -> link +y)"))
        wrist.append(body)
        # gripper vs own forearm/wrist (non-adjacent pairs are not auto-excluded)
        for gb in (f"{prefix}base_mount", f"{prefix}base", f"{prefix}right_driver", f"{prefix}left_driver",
                   f"{prefix}right_spring_link", f"{prefix}left_spring_link"):
            for arm in (f"{side}_wrist_yaw_link", f"{side}_wrist_pitch_link", f"{side}_wrist_roll_link", f"{side}_elbow_link"):
                excludes.append(ET.Element("exclude", body1=gb, body2=arm))
        all_excludes += excludes
        all_eq += eqs
        tendons.append(tendon)
        actuators.append(act)

    # ---- contact / equality / tendon / actuator sections
    contact = g1.find("contact")
    if contact is None:
        contact = ET.SubElement(g1, "contact")
    contact.append(comment("sliding base: ONLY the two foot bodies are excluded from world (floor) contacts"))
    ET.SubElement(contact, "exclude", body1="left_ankle_roll_link", body2="world")
    ET.SubElement(contact, "exclude", body1="right_ankle_roll_link", body2="world")
    contact.append(comment("2F-85 upstream linkage exclusions (prefixed) + gripper-vs-own-forearm pairs"))
    for e in all_excludes:
        contact.append(e)
    equality = g1.find("equality")
    if equality is None:
        equality = ET.SubElement(g1, "equality")
    equality.append(comment("base weld (mocap -> pelvis); solref/solimp tuned so the pelvis tracks within 2 mm"))
    ET.SubElement(equality, "weld", name="base_weld", body1="base_target", body2="pelvis", solref="0.01 1",
                  solimp="0.99 0.999 0.001 0.5 2")
    equality.append(comment("2F-85 linkage constraints (connect x2 + driver joint coupling per gripper), kept active"))
    for e in all_eq:
        equality.append(e)
    tendon_el = g1.find("tendon")
    if tendon_el is None:
        tendon_el = ET.SubElement(g1, "tendon")
    for t in tendons:
        tendon_el.append(t)
    actuator = g1.find("actuator")
    actuator.append(comment("2F-85 finger actuators (upstream general/affine position actuator, ctrlrange 0..255, forcerange +-5 N)"))
    for a in actuators:
        actuator.append(a)
    # keyframe: nq changed (gripper joints) -> drop; SimWorld sets the posture explicitly
    kf = g1.find("keyframe")
    if kf is not None:
        g1.remove(kf)

    header = ET.Comment(
        " Owner: backbone (ALL)\n"
        "     GENERATED by scripts/build_g1_model.py from the pinned MuJoCo Menagerie files\n"
        "     unitree_g1/g1.xml (BSD-3-Clause, Unitree) and robotiq_2f85/2f85.xml (BSD-2-Clause,\n"
        "     ROS-Industrial), revision assets/menagerie_revision.txt.  Do not edit by hand; edit the\n"
        "     generator and re-run it.  See assets/README.md for the documented decisions. ")
    indent(g1)
    xml = ET.tostring(g1, encoding="unicode")
    OUT.write_text("<!--" + header.text + "-->\n" + xml + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
