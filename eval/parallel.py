# Owner: backbone (ALL)
"""``eval.runner --jobs N``: run trials in N parallel worker lanes.

Trials are split round-robin into N lanes.  Each lane runs its trials one at
a time, each in a fresh ``eval.runner --only <id>`` subprocess (so a trial can
be killed at its wall-clock limit without losing the others), with its own
output dir, A/B audit dirs and log under ``<out>/job_<k>/``.  After all lanes
finish, every trial's final record is linked into ``<out>/merged/<id>/`` and
``<out>/E2E_TABLE.md`` is written (eval.final_table).

A trial that ends in ERROR because of the provider (HTTP 5xx/429, DNS,
connection, content filter) is rerun once; both attempts are kept in the lane
dir and the rerun counts.  A trial still running after ``timeout_s`` is killed
and recorded as outcome TIMEOUT.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

from core.types import TrialRecord
from eval.logger import write_trial_record

PROVIDER_ERROR = re.compile(
    r"HTTP 5\d\d|HTTP 429|NameResolution|name resolution|Temporary failure|ConnectionError|"
    r"Connection (?:reset|refused|aborted)|Read timed out|content filter|ContentFiltered|TransportError",
    re.I)


def split_round_robin(ids: list[str], jobs: int) -> list[list[str]]:
    return [ids[k::jobs] for k in range(jobs)]


def _latest_record(lane: pathlib.Path, trial_id: str):
    paths = sorted(lane.glob(f"runs/*/{trial_id}/trial_record.json"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def is_provider_error(record: dict, log_text: str = "") -> bool:
    if record.get("outcome") != "ERROR":
        return False
    return bool(PROVIDER_ERROR.search(str(record.get("error") or "")))


def _run_one(trial_id, args, lane, env, log, timeout_s):
    cmd = [sys.executable, "-m", "eval.runner", "--mode", args.mode, "--trials", str(args.trials),
           "--only", trial_id, "--out", str(lane / "runs")]
    if args.mock_all:
        cmd.append("--mock-all")
    if args.no_cache:
        cmd.append("--no-cache")
    if args.video:
        cmd.append("--video")
    trial_log = lane / "logs" / f"{trial_id}.log"
    trial_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(trial_log, "a") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
    wall = time.monotonic() - start
    if timed_out:
        rec = TrialRecord(trial_id=trial_id, outcome="TIMEOUT",
                          error=f"killed after {timeout_s:.0f} s wall clock (eval.parallel)")
        rec.timings["wall_s"] = wall
        path = lane / "runs" / f"timeout_{int(time.time())}" / trial_id / "trial_record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_trial_record(rec, path)
    path = _latest_record(lane, trial_id)
    record = json.loads(path.read_text()) if path else {"outcome": "ERROR", "error": "no trial record"}
    with open(log, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {trial_id} outcome={record.get('outcome')} "
                f"claimed={record.get('claimed_success')} actual={record.get('actual_success')} "
                f"wall={wall:.0f}s\n")
    return record, trial_log.read_text(errors="replace")


def _lane(k, ids, args, root, timeout_s, rerun_log):
    lane = root / f"job_{k}"
    lane.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["EE4705_QWEN_AUDIT_DIR"] = str((lane / "audit_b").resolve())
    env["EE4705_VLM_AUDIT_DIR"] = str((lane / "audit_a").resolve())
    log = lane / "job.log"
    for trial_id in ids:
        record, text = _run_one(trial_id, args, lane, env, log, timeout_s)
        if is_provider_error(record, text):
            rerun_log.append({"trial_id": trial_id, "lane": k, "error": str(record.get("error"))[-300:]})
            _run_one(trial_id, args, lane, env, log, timeout_s)


def run_parallel(args, trials: list[dict]) -> int:
    from eval.final_table import merge_and_table

    root = pathlib.Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    ids = [str(t.get("id", pathlib.Path(t["_path"]).stem)) for t in trials]
    lanes = split_round_robin(ids, args.jobs)
    reruns: list[dict] = []
    (root / "parallel_meta.json").write_text(json.dumps(
        {"jobs": args.jobs, "lanes": lanes, "mode": args.mode, "trials": str(args.trials),
         "no_cache": args.no_cache, "trial_timeout_s": args.trial_timeout,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    threads = [threading.Thread(target=_lane, args=(k, lane, args, root, args.trial_timeout, reruns))
               for k, lane in enumerate(lanes) if lane]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (root / "provider_reruns.json").write_text(json.dumps(reruns, indent=2))
    score = merge_and_table(root, args.trials, ids)
    print(score["line"])
    return 0
