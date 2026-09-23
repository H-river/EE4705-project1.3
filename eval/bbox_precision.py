"""Bounding Box Precision: how tightly A's predicted box matches ground truth,
for cases where target selection was already correct. Reuses an existing
eval.grounding run + its dataset.json -- makes no new model calls.

A case counts toward "correct" if IoU(predicted, ground_truth) >= --iou-threshold
(default 0.5, the common "tight box" convention -- stricter than the 0.30 used
elsewhere in this project only to decide *which* object a box belongs to).
"""
import argparse
import json
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_case(case_path, dataset_cases):
    case = json.loads(case_path.read_text())
    if 'scene' not in case or not case.get('target_selection_correct'):
        return None  # error case, or selection itself was wrong -- box precision not meaningful here
    ds_case = dataset_cases[case['id']]
    if ds_case['expected_kind'] != 'unique' or not case.get('grounded'):
        return None  # only single, unambiguous, present targets have one ground-truth box to compare against
    matched_name = ds_case['candidate_gt_ids'][0]
    gt_box = ds_case['gt_bboxes'].get(matched_name)
    pred_box = case['grounded'].get('bbox_xyxy')
    if gt_box is None or pred_box is None:
        return None
    return {'id': case['id'], 'target': case['target'], 'matched_name': matched_name,
            'predicted_bbox': pred_box, 'ground_truth_bbox': gt_box, 'iou': iou(pred_box, gt_box)}


def check_precision(eval_dir, dataset_path, iou_threshold=0.5):
    eval_dir = Path(eval_dir)
    dataset_cases = {c['id']: c for c in json.loads(Path(dataset_path).read_text())['cases']}
    results = [r for f in sorted(eval_dir.glob('a_*.json'))
               if (r := check_case(f, dataset_cases)) is not None]
    n = len(results)
    mean_iou = sum(r['iou'] for r in results) / n if n else 0.0
    tight = sum(1 for r in results if r['iou'] >= iou_threshold)
    precision = tight / n if n else 0.0
    summary = {'cases_checked': n, 'iou_threshold': iou_threshold, 'mean_iou': mean_iou,
               'cases_at_or_above_threshold': tight, 'bounding_box_precision': precision,
               'target_met_ge_0.70': precision >= 0.70, 'per_case': results}
    (eval_dir / 'bbox_precision_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args(argv)
    summary = check_precision(args.eval_dir, args.dataset, args.iou_threshold)
    for r in summary['per_case']:
        flag = '  <-- LOOSE' if r['iou'] < summary['iou_threshold'] else ''
        print(f"{r['id']}: iou={r['iou']:.3f} ({r['matched_name']}){flag}")
    print()
    print(f"Cases checked: {summary['cases_checked']}")
    print(f"Mean IoU: {summary['mean_iou']:.3f}")
    print(f"Bounding Box Precision (IoU>={summary['iou_threshold']}): "
          f"{summary['bounding_box_precision']:.1%} "
          f"({'MEETS' if summary['target_met_ge_0.70'] else 'BELOW'} >=70% target)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
