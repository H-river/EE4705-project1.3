"""Offline VQA object-mention accuracy check, reusing an existing eval.grounding run.

Cross-checks each case's saved scene.caption against that case's ground-truth
object inventory from dataset.json. Makes no new model calls.
"""
import argparse
import json
from pathlib import Path

VOCAB = {'stone': 'stone', 'stone2': 'stone', 'cube': 'cube',
         'bottle': 'bottle', 'red_region': 'region'}


def check_case(case_path, dataset_cases):
    case = json.loads(case_path.read_text())
    if 'scene' not in case:
        return None  # error case (e.g. an exhausted repair attempt) -- no caption to check
    caption = case['scene']['caption'].lower()
    ds_case = next(c for c in dataset_cases if c['id'] == case['id'])
    truth_names = set(ds_case['gt_bboxes'].keys())
    mentioned = {name for name in truth_names if VOCAB[name] in caption}
    missed = truth_names - mentioned
    hallucinated = {v for v in set(VOCAB.values())
                     if v in caption and not any(VOCAB[n] == v for n in truth_names)}
    recall = len(mentioned) / len(truth_names) if truth_names else 1.0
    return {'id': case['id'], 'truth': sorted(truth_names), 'recall': recall,
            'missed': sorted(missed), 'hallucinated': sorted(hallucinated)}


def check_captions(eval_dir, dataset_path):
    eval_dir = Path(eval_dir)
    dataset = json.loads(Path(dataset_path).read_text())
    results = [r for f in sorted(eval_dir.glob('a_*.json'))
               if (r := check_case(f, dataset['cases'])) is not None]
    n = len(results)
    mean_recall = sum(r['recall'] for r in results) / n if n else 0.0
    perfect = sum(1 for r in results if r['recall'] == 1.0 and not r['hallucinated'])
    summary = {'cases_checked': n, 'mean_object_mention_recall': mean_recall,
               'fully_correct_captions': perfect, 'per_case': results}
    (eval_dir / 'vqa_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True, help='an existing eval.grounding run output folder')
    parser.add_argument('--dataset', required=True, help='the matching dataset.json from eval.grounding capture')
    args = parser.parse_args(argv)
    summary = check_captions(args.eval_dir, args.dataset)
    for r in summary['per_case']:
        flag = '  <-- MISSED or HALLUCINATED' if r['missed'] or r['hallucinated'] else ''
        print(f"{r['id']}: recall={r['recall']:.2f} missed={r['missed']} "
              f"hallucinated={r['hallucinated']}{flag}")
    print()
    print(f"Mean object-mention recall: {summary['mean_object_mention_recall']:.3f}")
    print(f"Fully correct captions: {summary['fully_correct_captions']}/{summary['cases_checked']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
