#!/usr/bin/env python3
"""Recompute SR/AP from LiLo-VLA eval result JSONs.

The `success_rate` field written into the result JSONs is WRONG and must never be
read. It is the environment's *instantaneous* BDDL goal conjunction, sampled at a
single point inside the VLA rollout (`lilo_vla/core/pipeline.py`, `_execute_vla`).
Every other `env.step` on the path -- post-actions, stabilisation, place recovery and
all motion-planner steps -- discards `done`, and `_execute_vla` returns the moment the
*skill* predicate fires, so the instant the task actually completes is never sampled.
(`ignore_done=True` is a red herring: LIBERO's `BDDLBaseDomain.step()` overwrites
`done` with `_check_success()`, and the field is non-zero in several result files.)

Correct definitions (paper Sec IV-A.3):
  SR = fraction of trials where the ENTIRE skill sequence completed in order
     = mean(skills_completed == total_skills)
  AP = mean(skills_completed / total_skills)   -- longest in-order prefix

Usage:
    python scripts/analysis/score_results.py 'outputs/eval_results/*.json'

    # the "w/o Recovery" ablation row: the same runs, scored from the synthesised
    # no-recovery block (see docs/REPRODUCTION.md section 8.2)
    python scripts/analysis/score_results.py --bucket no_recovery 'outputs/eval_results/*.json'
"""
import json, sys, glob, os


def score(path, bucket='with_recovery'):
    d = json.load(open(path))
    trials = d.get('trials', [])
    if not trials:
        return None
    n = len(trials)
    srs, aps = [], []
    for t in trials:
        blk = t.get(bucket, t)
        total = t.get('total_skills') or blk.get('total_skills')
        done = blk.get('skills_completed')
        if total in (None, 0) or done is None:
            continue
        srs.append(1.0 if done == total else 0.0)
        aps.append(done / total)
    if not srs:
        return None
    return {
        'file': os.path.basename(path),
        'task': d.get('task_name') or d.get('pipeline_config', {}).get('task_name'),
        'benchmark': d.get('pipeline_config', {}).get('benchmark_name'),
        'recovery_mode': d.get('pipeline_config', {}).get('recovery_mode'),
        'n_trials': n,
        'SR': round(100 * sum(srs) / len(srs), 1),
        'AP': round(100 * sum(aps) / len(aps), 2),
        # The wrong field, shown so the difference is visible. It lives at
        # metrics.<bucket>.success_rate, NOT at the top level.
        'reported_success_rate_field_BOGUS': d.get('metrics', {}).get(bucket, {}).get('success_rate'),
        'timestamp': d.get('timestamp'),
    }


if __name__ == '__main__':
    argv = sys.argv[1:]
    bucket = 'with_recovery'
    if '--bucket' in argv:
        i = argv.index('--bucket')
        bucket = argv[i + 1]
        del argv[i:i + 2]
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)) if any(c in a for c in '*?[') else [a])
    rows = [r for r in (score(p, bucket) for p in paths) if r]
    rows.sort(key=lambda r: (r['task'] or '', r['timestamp'] or ''))
    if not rows:
        print('no scorable result files'); sys.exit(1)
    w = max(len(str(r['task'])) for r in rows)
    print(f"# bucket: {bucket}")
    print(f"{'task'.ljust(w)}  {'n':>3}  {'SR%':>6}  {'AP%':>7}  {'recov':<12} {'bogus_field':>11}  file")
    for r in rows:
        print(f"{str(r['task']).ljust(w)}  {r['n_trials']:>3}  {r['SR']:>6}  {r['AP']:>7}  "
              f"{str(r['recovery_mode'] or 'full'):<12} {str(r['reported_success_rate_field_BOGUS']):>11}  {r['file']}")
    if len(rows) > 1:
        print(f"\nMEAN over {len(rows)} configs:  SR {sum(r['SR'] for r in rows)/len(rows):.1f}%   "
              f"AP {sum(r['AP'] for r in rows)/len(rows):.2f}%")
