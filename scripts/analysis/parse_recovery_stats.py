#!/usr/bin/env python3
"""
Parse evaluation result JSONs and print recovery statistics for paper rebuttal.

Usage:
    # Single file
    python scripts/analysis/parse_recovery_stats.py

    # Custom path
    python scripts/analysis/parse_recovery_stats.py \
        --json_path outputs/eval_results/eval_organize_table_v1_n10_above_atomic_20260322_194917.json

    # Multiple files (per-task breakdown + aggregate)
    python scripts/analysis/parse_recovery_stats.py \
        --json_path file1.json file2.json file3.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_single_file(json_path: str) -> dict:
    """Parse one eval result JSON and return raw per-skill records."""
    with open(json_path) as f:
        data = json.load(f)

    task_name = data['task_name']
    records = []

    for trial in data['trials']:
        trial_num = trial['trial_num']
        total_skills = trial['total_skills']

        for sd in trial.get('skill_details', []):
            retries = sd['retries']
            pick_rec = sd.get('pick_recovery', {})
            total_retries = retries['pose'] + retries['mp'] + retries['vla']
            pick_recovery_attempts = pick_rec.get('attempts', 0)
            triggered = total_retries > 0 or pick_recovery_attempts > 0

            records.append({
                'task_name': task_name,
                'trial_num': trial_num,
                'total_skills_in_trial': total_skills,
                'skill_name': sd['skill_name'],
                'skill_type': sd['skill_type'],
                'success': sd['success'],
                'pose_retries': retries['pose'],
                'mp_retries': retries['mp'],
                'vla_retries': retries['vla'],
                'pick_recovery_attempts': pick_recovery_attempts,
                'pick_recovery_success': pick_rec.get('success', False),
                'recovery_triggered': triggered,
                'resolved': sd['success'] and triggered,
            })

    return {'task_name': task_name, 'records': records, 'raw': data}


def compute_stats(records: list, label: str = '') -> dict:
    """Compute aggregate recovery statistics from a list of skill records."""
    if not records:
        return None

    total_executions = len(records)
    triggered = [r for r in records if r['recovery_triggered']]
    resolved = [r for r in records if r['resolved']]
    first_attempt_success = [r for r in records if r['success'] and not r['recovery_triggered']]

    # Failure type distribution (skills that ultimately failed)
    failed = [r for r in records if not r['success']]
    fail_by_type = defaultdict(int)
    for r in failed:
        fail_by_type[r['skill_type']] += 1

    # Recovery trigger breakdown by stage
    pose_triggered = sum(1 for r in records if r['pose_retries'] > 0)
    mp_triggered = sum(1 for r in records if r['mp_retries'] > 0)
    vla_triggered = sum(1 for r in records if r['vla_retries'] > 0)
    pick_rec_triggered = sum(1 for r in records if r['pick_recovery_attempts'] > 0)

    # Total retry counts
    total_pose = sum(r['pose_retries'] for r in records)
    total_mp = sum(r['mp_retries'] for r in records)
    total_vla = sum(r['vla_retries'] for r in records)
    total_pick_rec = sum(r['pick_recovery_attempts'] for r in records)
    pick_rec_successes = sum(1 for r in records if r['pick_recovery_success'])

    # Recovery trigger by skill type
    trigger_by_type = defaultdict(lambda: {'total': 0, 'triggered': 0, 'resolved': 0, 'failed': 0})
    for r in records:
        t = r['skill_type']
        trigger_by_type[t]['total'] += 1
        if r['recovery_triggered']:
            trigger_by_type[t]['triggered'] += 1
            if r['resolved']:
                trigger_by_type[t]['resolved'] += 1
        if not r['success']:
            trigger_by_type[t]['failed'] += 1

    return {
        'label': label,
        'total_executions': total_executions,
        'first_attempt_success': len(first_attempt_success),
        'recovery_triggered': len(triggered),
        'recovery_resolved': len(resolved),
        'recovery_unresolved': len(triggered) - len(resolved),
        'total_failed': len(failed),
        'trigger_rate': len(triggered) / total_executions if total_executions > 0 else 0,
        'resolution_rate': len(resolved) / len(triggered) if len(triggered) > 0 else 0,
        'fail_by_type': dict(fail_by_type),
        'trigger_by_type': dict(trigger_by_type),
        'stage_breakdown': {
            'pose': {'skills_triggered': pose_triggered, 'total_retries': total_pose},
            'mp': {'skills_triggered': mp_triggered, 'total_retries': total_mp},
            'vla': {'skills_triggered': vla_triggered, 'total_retries': total_vla},
            'pick_recovery': {'skills_triggered': pick_rec_triggered, 'total_attempts': total_pick_rec, 'successes': pick_rec_successes},
        },
    }


def print_stats(stats: dict):
    """Pretty-print recovery statistics."""
    if stats is None:
        print("  (no skill_details data available)\n")
        return

    label = stats['label']
    total = stats['total_executions']
    trig = stats['recovery_triggered']
    res = stats['recovery_resolved']
    unres = stats['recovery_unresolved']

    print(f"{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  Total skill executions:       {total}")
    print(f"  First-attempt successes:      {stats['first_attempt_success']}  ({100*stats['first_attempt_success']/total:.1f}%)" if total > 0 else "")
    print(f"  Total failures (unrecovered): {stats['total_failed']}")

    print(f"\n  --- Recovery Trigger Rate ---")
    print(f"  Triggered:  {trig}/{total}  ({100*stats['trigger_rate']:.1f}%)")

    print(f"\n  --- Recovery Resolution Rate ---")
    print(f"  Resolved:   {res}/{trig}  ({100*stats['resolution_rate']:.1f}%)" if trig > 0 else f"  Resolved:   0/0  (N/A)")
    print(f"  Unresolved: {unres}/{trig}" if trig > 0 else "")

    print(f"\n  --- Recovery Stage Breakdown ---")
    sb = stats['stage_breakdown']
    print(f"  Above-Pose retries:  {sb['pose']['skills_triggered']} skills, {sb['pose']['total_retries']} total retries")
    print(f"  Motion-Plan retries: {sb['mp']['skills_triggered']} skills, {sb['mp']['total_retries']} total retries")
    print(f"  VLA retries:         {sb['vla']['skills_triggered']} skills, {sb['vla']['total_retries']} total retries")
    pr = sb['pick_recovery']
    print(f"  Pick-Recovery:       {pr['skills_triggered']} skills, {pr['total_attempts']} attempts, {pr['successes']} succeeded")

    print(f"\n  --- Failure Type Distribution ---")
    fail_by = stats['fail_by_type']
    if fail_by:
        for stype in ['pick', 'place', 'other']:
            if stype in fail_by:
                print(f"  {stype:>8s}: {fail_by[stype]} failures")
    else:
        print(f"  (no unrecovered failures)")

    print(f"\n  --- Per Skill-Type Breakdown ---")
    print(f"  {'Type':>8s}  {'Total':>6s}  {'Triggered':>10s}  {'Resolved':>9s}  {'Failed':>7s}")
    for stype in ['pick', 'place', 'other']:
        tb = stats['trigger_by_type'].get(stype)
        if tb:
            print(f"  {stype:>8s}  {tb['total']:>6d}  {tb['triggered']:>10d}  {tb['resolved']:>9d}  {tb['failed']:>7d}")

    print()


def main():
    default_path = "outputs/eval_results/eval_organize_table_v1_n10_above_atomic_20260322_194917.json"

    parser = argparse.ArgumentParser(description='Parse eval result JSONs and print recovery statistics')
    parser.add_argument('--json_path', type=str, nargs='+', default=[default_path],
                        help='Path(s) to evaluation result JSON file(s)')
    args = parser.parse_args()

    all_records = []

    for jp in args.json_path:
        path = Path(jp)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.\n")
            continue

        parsed = parse_single_file(path)
        records = parsed['records']
        task_name = parsed['task_name']

        if not records:
            print(f"\n{'='*70}")
            print(f"  {task_name}  ({path.name})")
            print(f"{'='*70}")
            print("  (no skill_details data in any trial — likely an older result format)\n")
            continue

        # Per-task stats
        stats = compute_stats(records, label=f"{task_name}  ({path.name})")
        print_stats(stats)

        # Also print per-trial mini-summary
        trials_by_num = defaultdict(list)
        for r in records:
            trials_by_num[r['trial_num']].append(r)

        print(f"  --- Per-Trial Summary ---")
        print(f"  {'Trial':>6s}  {'Skills':>6s}  {'1st-Atpt':>8s}  {'Triggered':>10s}  {'Resolved':>9s}  {'Failed':>7s}")
        for trial_num in sorted(trials_by_num.keys()):
            tr = trials_by_num[trial_num]
            n = len(tr)
            first = sum(1 for r in tr if r['success'] and not r['recovery_triggered'])
            trig = sum(1 for r in tr if r['recovery_triggered'])
            resol = sum(1 for r in tr if r['resolved'])
            fail = sum(1 for r in tr if not r['success'])
            print(f"  {trial_num:>6d}  {n:>6d}  {first:>8d}  {trig:>10d}  {resol:>9d}  {fail:>7d}")
        print()

        all_records.extend(records)

    # Aggregate across all files if multiple
    if len(args.json_path) > 1 and all_records:
        agg_stats = compute_stats(all_records, label="AGGREGATE (all tasks)")
        print_stats(agg_stats)


if __name__ == "__main__":
    main()
