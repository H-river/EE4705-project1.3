"""Summarize C action attempts and independent final placement measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(records):
    actions = defaultdict(lambda: {'attempts': 0, 'passed': 0, 'recovery_attempts': 0})
    cases, errors = [], []
    for record in records:
        sequence = [x['result'] for x in record.get('extra', {}).get('execution_records', [])]
        failed = [x for x in sequence if not x['success']]
        recovered = [x for x in sequence if x['recovery_attempted']]
        for result in sequence:
            stat = actions[result['action']['skill']]
            stat['attempts'] += 1
            stat['passed'] += int(result['success'])
            stat['recovery_attempts'] += int(result['recovery_attempted'])
        geometry = record.get('extra', {}).get('placement_geometry')
        if geometry:
            errors.append(geometry['center_xy_error_m'])
        cases.append({'trial_id': record['trial_id'], 'actual_success': record['actual_success'],
                      'telemetry_present': bool(sequence),
                      'completed_without_retry_or_recovery': bool(sequence) and record['actual_success'] is True
                                                           and not failed and not recovered,
                      'failed_actions': [{'skill': x['action']['skill'], 'error': x['error_code']} for x in failed],
                      'recovery_actions': [x['action']['skill'] for x in recovered],
                      'center_xy_error_m': geometry['center_xy_error_m'] if geometry else None})
    for value in actions.values():
        value['success_rate'] = value['passed']/value['attempts']
    return {'trials': len(records), 'actual_successes': sum(r['actual_success'] is True for r in records),
            'trials_with_execution_telemetry': sum(c['telemetry_present'] for c in cases),
            'completed_without_retry_or_recovery': sum(c['completed_without_retry_or_recovery'] for c in cases),
            'actions': dict(actions), 'placement_geometry_samples': len(errors),
            'mean_center_xy_error_m': statistics.mean(errors) if errors else None,
            'max_center_xy_error_m': max(errors) if errors else None, 'cases': cases,
            'note': 'Action rates include every attempt, including retries. Final XY distance is measured by the evaluator; height is not counted as XY error. Missing telemetry is not a pass.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.run_dir.glob('*/trial_record.json'))
    if not paths: parser.error('No trial_record.json files in this run directory')
    summary = summarize([json.loads(p.read_text()) for p in paths])
    (args.run_dir/'skill_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
