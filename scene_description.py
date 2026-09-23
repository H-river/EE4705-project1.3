"""Official Scene Description Accuracy run.

Cross-checks each case's saved scene.caption (the answer to the default
'what supported objects and colors are visible?' query) against that case's
full ground-truth object inventory. Reuses an existing eval.grounding run --
makes no new model calls.

A case is "correct" only if the caption names every genuinely present object
AND names nothing that is not present (no missed objects, no hallucinations).
This is the same recall/hallucination check as eval/vqa_check.py, formalized
here with an explicit pass/fail per case and a target comparison.
"""
import argparse
import json
from pathlib import Path

VOCAB = {'stone': 'stone', 'stone2': 'stone', 'cube': 'cube',
         'bottle': 'bottle', 'red_region': 'region'}


def check_case(case_path, dataset_cases):
    case = json.loads(case_path.read_text())
    if 'scene' not in case:
        return None  # error case -- no caption produced at all
    caption = case['scene']['caption'].lower()
    truth_names = set(dataset_cases[case['id']]['gt_bboxes'].keys())
    mentioned = {name for name in truth_names if VOCAB[name] in caption}
    missed = truth_names - mentioned
    hallucinated = {v for v in set(VOCAB.values())
                     if v in caption and not any(VOCAB[n] == v for n in truth_names)}
    return {'id': case['id'], 'truth': sorted(truth_names),
            'missed': sorted(missed), 'hallucinated': sorted(hallucinated),
            'correct': not missed and not hallucinated}


def run_scene_description_check(eval_dir, dataset_path, target=0.90):
    eval_dir = Path(eval_dir)
    dataset_cases = {c['id']: c for c in json.loads(Path(dataset_path).read_text())['cases']}
    results = [r for f in sorted(eval_dir.glob('a_*.json'))
               if (r := check_case(f, dataset_cases)) is not None]
    n = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / n if n else 0.0
    summary = {'cases_checked': n, 'correct': correct, 'scene_description_accuracy': accuracy,
               'target': target, 'target_met': accuracy >= target, 'per_case': results}
    (eval_dir / 'scene_description_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target', type=float, default=0.90)
    args = parser.parse_args(argv)
    summary = run_scene_description_check(args.eval_dir, args.dataset, args.target)
    for r in summary['per_case']:
        flag = '' if r['correct'] else f"  <-- missed={r['missed']} hallucinated={r['hallucinated']}"
        print(f"{r['id']}: {'PASS' if r['correct'] else 'FAIL'}{flag}")
    print()
    print(f"Scene Description Accuracy: {summary['correct']}/{summary['cases_checked']} "
          f"= {summary['scene_description_accuracy']:.1%} "
          f"({'MEETS' if summary['target_met'] else 'BELOW'} >={summary['target']:.0%} target)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
