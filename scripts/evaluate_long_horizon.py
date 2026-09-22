#!/usr/bin/env python3
"""
LiLo-VLA long-horizon task evaluation entry point.

Runs multiple trials of a long-horizon task and computes:
1. Success Rate: Percentage of trials where the entire task succeeds
2. Average Success Progress: Average ratio of successfully completed skills across all trials
3. Recovery Statistics: Detailed retry and recovery metrics

Run from the lilo-vla/ project root:

    # Suite 2 -- Ultra-Long  (3 tasks x Original / Reorder 1 / Reorder 2)
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
        --vla_checkpoint "$CKPT_MAIN" \
        --task_name 'Complete Kitchen Organization (Original)' \
        --num_trials 10 \
        --benchmark long_horizon_tasks_v1 \
        --apply_distractor_masking

    # Suite 1 -- LIBERO-Long++  (6 tasks x Original / Reorder 1)
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_long_horizon.py \
        --vla_checkpoint "$CKPT_MAIN" \
        --task_name 'Turn On The Stove And Put The Moka Pot On It (Original)' \
        --num_trials 10 \
        --benchmark long_horizon_tasks_libero_long \
        --apply_distractor_masking

    # Naive-retry control: add  --recovery_mode naive_strict
    # No-masking ablation:   drop --apply_distractor_masking

Notes:
- $CKPT_MAIN must point at the released checkpoint directory. The directory NAME is parsed
  is no longer parsed for `unnorm_key`: that now comes from the checkpoint's
  dataset_statistics.json, so a checkpoint downloaded from the Hub works as-is.
- The historical --benchmark names (long_horizon_tasks_v1, long_horizon_tasks_libero_long)
  are aliased onto the published suite names (ultra_long, libero_long_plus_plus). Both work.
- Metrics: metrics.<bucket>.success_rate is the fraction of trials that completed the
  ENTIRE skill sequence in order; metrics.<bucket>.avg_progress is the mean fraction of
  the sequence completed. Both are the metrics the paper reports.
  (Result JSONs written before 2026-09 stored the environment's instantaneous done flag
  in success_rate instead; recompute those with scripts/analysis/score_results.py.)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Run-from-anywhere: put this checkout's project root at the FRONT of sys.path, so
# `lilo_vla` and this repo's `prismatic` resolve here even when the interpreter was
# started from another directory, and even when some other openvla-oft checkout is
# pip-installed in the same environment (its editable finder is appended to
# sys.meta_path, so sys.path[0] wins).  A no-op under `pip install -e .`.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Bootstrap: writes the LIBERO config and applies the libero_compat patches.
# MUST be imported before anything that imports `libero`.
import lilo_vla  # noqa: F401

from lilo_vla.core.pipeline import LongHorizonPipeline
from lilo_vla.utils.task_names import resolve_task_name

# lilo-vla/  (this file lives at lilo-vla/scripts/evaluate_long_horizon.py)
_LILO_ROOT = Path(__file__).resolve().parents[1]

# Historical CLI --benchmark names -> published lilo_vla.benchmark suite names.
# New suite names pass through unchanged, so every historical command still works verbatim.
_BENCHMARK_ALIASES = {
    "long_horizon_tasks_v1": "ultra_long",
    "long_horizon_tasks_libero_long": "libero_long_plus_plus",
}


def run_evaluation(num_trials: int, eval_mode: str = 'above', **pipeline_kwargs):
    """Run multiple trials and compute aggregate metrics."""
    task_name = pipeline_kwargs.get('task_name', 'Unknown')

    results = {
        'task_name': task_name,
        'num_trials': num_trials,
        'eval_mode': eval_mode,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'pipeline_config': {
            'benchmark_name': pipeline_kwargs.get('benchmark_name', 'long_horizon_tasks_v1'),
            'max_vla_retries': pipeline_kwargs.get('max_vla_retries', 3),
            'vla_horizon': pipeline_kwargs.get('vla_horizon', 200),
            'recovery_mode': pipeline_kwargs.get('recovery_mode', 'full'),
        },
        'trials': []
    }

    # Add mode-specific config
    if eval_mode == 'above':
        results['pipeline_config'].update({
            'above_height': pipeline_kwargs.get('above_height', 0.03),
            'motion_planner_type': pipeline_kwargs.get('motion_planner_type', 'mplib'),
            'collision_aware': pipeline_kwargs.get('collision_aware', True)
        })

    task_successes = 0
    no_recovery_successes = 0
    progress_ratios = []
    no_recovery_progress_ratios = []

    # Create pipeline ONCE to avoid OOM from loading multiple VLA models
    print(f"\n{'='*80}")
    print(f"🔧 Initializing LiLo-VLA Pipeline (Above-Pose Strategy)...")
    print(f"{'='*80}\n")

    try:
        if eval_mode == 'above':
            pipeline = LongHorizonPipeline(**pipeline_kwargs)
        else:
            raise ValueError(f"Invalid eval_mode: {eval_mode}. Must be 'above'.")
        print(f"✅ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return results

    for trial_num in range(1, num_trials + 1):
        print(f"\n{'='*80}")
        print(f"🔄 TRIAL {trial_num}/{num_trials}")
        print(f"{'='*80}\n")

        try:
            # Set trial number for video naming (reuses same subfolder)
            pipeline.trial_num = trial_num
            # Run pipeline (reuses same VLA model, creates new env each time)
            task_success = pipeline.execute()

            # Get skill progress from pipeline
            skills_completed = getattr(pipeline, 'success_count', 0)
            total_skills = getattr(pipeline, 'total_skills', 0)
            recovery_stats = getattr(pipeline, 'recovery_stats', None)

            # Calculate WITH RECOVERY metrics (actual results)
            with_recovery_progress = skills_completed / total_skills if total_skills > 0 else 0.0
            # Task success == the ENTIRE skill sequence completed, in order. This is the
            # metric the paper reports. `task_success` (the env's done flag) is only an
            # instantaneous BDDL goal conjunction, sampled at a single point inside the
            # rollout, so it under-reports badly; it is kept below as `env_done` only.
            with_recovery_success = (total_skills > 0 and skills_completed == total_skills)

            # Calculate NO RECOVERY metrics (hypothetical: what if no retries allowed?)
            no_recovery_skills_completed = 0
            no_recovery_success = False
            stopped_at_skill = None

            if recovery_stats and 'skills' in recovery_stats:
                for idx, skill_stat in enumerate(recovery_stats['skills'], 1):
                    # Check if this skill needed ANY retries
                    needed_retry = (
                        skill_stat['pose_retries'] > 0 or
                        skill_stat['mp_retries'] > 0 or
                        skill_stat['vla_retries'] > 0 or
                        skill_stat['pick_recovery_attempts'] > 0
                    )

                    if needed_retry:
                        # This skill failed on first attempt - no recovery would stop here
                        no_recovery_skills_completed = idx - 1
                        stopped_at_skill = skill_stat['skill_name']
                        break
                    else:
                        # This skill succeeded on first attempt
                        if skill_stat['success']:
                            no_recovery_skills_completed = idx
                        else:
                            # Skill failed even on first attempt (shouldn't happen, but handle it)
                            stopped_at_skill = skill_stat['skill_name']
                            break
                else:
                    # All skills completed without retries
                    no_recovery_success = (total_skills > 0
                                           and no_recovery_skills_completed == total_skills)

            no_recovery_progress = no_recovery_skills_completed / total_skills if total_skills > 0 else 0.0

            # Track progress ratios for both scenarios
            progress_ratios.append(with_recovery_progress)
            no_recovery_progress_ratios.append(no_recovery_progress)

            env_done = bool(task_success)

            if with_recovery_success:
                task_successes += 1
            if no_recovery_success:
                no_recovery_successes += 1

            # Extract detailed skill information
            skill_details = []
            if recovery_stats and 'skills' in recovery_stats:
                for skill_stat in recovery_stats['skills']:
                    skill_detail = {
                        'skill_name': skill_stat['skill_name'],
                        'skill_type': skill_stat['skill_type'],
                        'success': skill_stat['success'],
                        'retries': {
                            'pose': skill_stat['pose_retries'],
                            'mp': skill_stat['mp_retries'],
                            'vla': skill_stat['vla_retries']
                        }
                    }
                    # Add pick recovery info for place skills
                    if skill_stat['pick_recovery_attempts'] > 0:
                        skill_detail['pick_recovery'] = {
                            'attempts': skill_stat['pick_recovery_attempts'],
                            'success': skill_stat['pick_recovery_success']
                        }
                    skill_details.append(skill_detail)

            # Extract failure information
            failure_info = None
            if recovery_stats and recovery_stats.get('final_failure'):
                ff = recovery_stats['final_failure']
                failure_info = {
                    'failed_skill': ff['skill'],
                    'failed_skill_idx': ff['skill_idx'],
                    'failed_skill_type': ff['skill_type'],
                    'failed_stage': ff['stage'],  # 'above_pose', 'mp', or 'vla'
                    'retry_attempts': ff['attempted_recoveries']
                }

            # Store trial result with both scenarios
            trial_result = {
                'trial_num': trial_num,
                'total_skills': total_skills,
                'with_recovery': {
                    'task_success': with_recovery_success,
                    'skills_completed': skills_completed,
                    'progress_ratio': with_recovery_progress,
                    # The environment's instantaneous BDDL goal conjunction. Recorded for
                    # transparency only -- it is NOT the reported success metric.
                    'env_done': env_done
                },
                'no_recovery': {
                    'task_success': no_recovery_success,
                    'skills_completed': no_recovery_skills_completed,
                    'progress_ratio': no_recovery_progress,
                    'stopped_at_skill': stopped_at_skill
                },
                'skill_details': skill_details,
                'failure_info': failure_info
            }
            results['trials'].append(trial_result)

            print(f"\n📊 Trial {trial_num} Results:")
            print(f"   With Recovery: {'✅' if with_recovery_success else '❌'} ({skills_completed}/{total_skills} skills, {100*with_recovery_progress:.1f}%)")
            print(f"   No Recovery:   {'✅' if no_recovery_success else '❌'} ({no_recovery_skills_completed}/{total_skills} skills, {100*no_recovery_progress:.1f}%)")
            if stopped_at_skill:
                print(f"                  └─ Would have stopped at '{stopped_at_skill}'")
            print(f"\n{'='*80}")
            print(f"📈 Progress After Trial {trial_num}/{num_trials}:")
            print(f"   Success Rate (with recovery): {task_successes}/{trial_num} ({100*task_successes/trial_num:.1f}%)")
            print(f"   Success Rate (no recovery):   {no_recovery_successes}/{trial_num} ({100*no_recovery_successes/trial_num:.1f}%)")
            print(f"   Avg Progress (with recovery): {100*sum(progress_ratios)/len(progress_ratios):.1f}%")
            print(f"   Avg Progress (no recovery):   {100*sum(no_recovery_progress_ratios)/len(no_recovery_progress_ratios):.1f}%")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ Trial {trial_num} encountered error: {e}")
            import traceback
            traceback.print_exc()

            # Try to get partial progress even if exception occurred
            skills_completed = getattr(pipeline, 'success_count', 0)
            total_skills = getattr(pipeline, 'total_skills', 0)
            recovery_stats = getattr(pipeline, 'recovery_stats', None)

            # Calculate WITH RECOVERY metrics
            with_recovery_progress = skills_completed / total_skills if total_skills > 0 else 0.0
            with_recovery_success = False
            env_done = False          # the trial raised; the env flag was never read

            # Calculate NO RECOVERY metrics
            no_recovery_skills_completed = 0
            no_recovery_success = False
            stopped_at_skill = None

            if recovery_stats and 'skills' in recovery_stats:
                for idx, skill_stat in enumerate(recovery_stats['skills'], 1):
                    needed_retry = (
                        skill_stat['pose_retries'] > 0 or
                        skill_stat['mp_retries'] > 0 or
                        skill_stat['vla_retries'] > 0 or
                        skill_stat['pick_recovery_attempts'] > 0
                    )
                    if needed_retry:
                        no_recovery_skills_completed = idx - 1
                        stopped_at_skill = skill_stat['skill_name']
                        break
                    else:
                        if skill_stat['success']:
                            no_recovery_skills_completed = idx
                        else:
                            stopped_at_skill = skill_stat['skill_name']
                            break

            no_recovery_progress = no_recovery_skills_completed / total_skills if total_skills > 0 else 0.0

            progress_ratios.append(with_recovery_progress)
            no_recovery_progress_ratios.append(no_recovery_progress)

            # Extract skill details if available
            skill_details = []
            if recovery_stats and 'skills' in recovery_stats:
                for skill_stat in recovery_stats['skills']:
                    skill_detail = {
                        'skill_name': skill_stat['skill_name'],
                        'skill_type': skill_stat['skill_type'],
                        'success': skill_stat['success'],
                        'retries': {
                            'pose': skill_stat['pose_retries'],
                            'mp': skill_stat['mp_retries'],
                            'vla': skill_stat['vla_retries']
                        }
                    }
                    if skill_stat['pick_recovery_attempts'] > 0:
                        skill_detail['pick_recovery'] = {
                            'attempts': skill_stat['pick_recovery_attempts'],
                            'success': skill_stat['pick_recovery_success']
                        }
                    skill_details.append(skill_detail)

            # Record failed trial with both scenarios
            trial_result = {
                'trial_num': trial_num,
                'total_skills': total_skills,
                'with_recovery': {
                    'task_success': with_recovery_success,
                    'skills_completed': skills_completed,
                    'progress_ratio': with_recovery_progress,
                    # The environment's instantaneous BDDL goal conjunction. Recorded for
                    # transparency only -- it is NOT the reported success metric.
                    'env_done': env_done
                },
                'no_recovery': {
                    'task_success': no_recovery_success,
                    'skills_completed': no_recovery_skills_completed,
                    'progress_ratio': no_recovery_progress,
                    'stopped_at_skill': stopped_at_skill
                },
                'skill_details': skill_details,
                'failure_info': None,
                'error': str(e)
            }
            results['trials'].append(trial_result)

    # Calculate aggregate metrics for both scenarios
    with_recovery_success_rate = task_successes / num_trials if num_trials > 0 else 0.0
    with_recovery_avg_progress = sum(progress_ratios) / len(progress_ratios) if progress_ratios else 0.0
    no_recovery_success_rate = no_recovery_successes / num_trials if num_trials > 0 else 0.0
    no_recovery_avg_progress = sum(no_recovery_progress_ratios) / len(no_recovery_progress_ratios) if no_recovery_progress_ratios else 0.0

    # Calculate retry statistics across all trials
    total_pose_retries = 0
    total_mp_retries = 0
    total_vla_retries = 0
    total_pick_recoveries = 0
    successful_pick_recoveries = 0

    for trial in results['trials']:
        if trial.get('skill_details'):
            for skill in trial['skill_details']:
                total_pose_retries += skill['retries']['pose']
                total_mp_retries += skill['retries']['mp']
                total_vla_retries += skill['retries']['vla']
                if 'pick_recovery' in skill:
                    total_pick_recoveries += skill['pick_recovery']['attempts']
                    if skill['pick_recovery']['success']:
                        successful_pick_recoveries += 1

    results['metrics'] = {
        'with_recovery': {
            'success_rate': with_recovery_success_rate,
            'successful_trials': task_successes,
            'avg_progress': with_recovery_avg_progress
        },
        'no_recovery': {
            'success_rate': no_recovery_success_rate,
            'successful_trials': no_recovery_successes,
            'avg_progress': no_recovery_avg_progress
        },
        'retry_statistics': {
            'total_pose_retries': total_pose_retries,
            'total_mp_retries': total_mp_retries,
            'total_vla_retries': total_vla_retries,
            'total_pick_recoveries': total_pick_recoveries,
            'successful_pick_recoveries': successful_pick_recoveries
        }
    }

    return results


def save_results(results: dict, output_dir: Path, checkpoint: str = ''):
    """Save evaluation results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with key parameters
    task_name = results['task_name'].replace(' ', '_').lower()
    timestamp = results['timestamp']
    num_trials = results['num_trials']

    # Add checkpoint info if available
    checkpoint_suffix = ''
    if 'above_atomic' in checkpoint:
        checkpoint_suffix = '_above_atomic'
    elif 'above_atomic_long_id1' in checkpoint:
        checkpoint_suffix = '_id1'
    elif 'above_atomic_long_id2' in checkpoint:
        checkpoint_suffix = '_id2'
    elif 'above_atomic_long_id3' in checkpoint:
        checkpoint_suffix = '_id3'
    elif 'both_view' in checkpoint:
        checkpoint_suffix = '_both_view'

    filename = f"eval_{task_name}_n{num_trials}{checkpoint_suffix}_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def print_summary(results: dict):
    """Print concise evaluation summary."""
    metrics = results['metrics']
    retry_stats = metrics['retry_statistics']
    with_recovery = metrics['with_recovery']
    no_recovery = metrics['no_recovery']

    eval_mode = results.get('eval_mode', 'above')
    mode_str = "Above Pose"

    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY ({mode_str})")
    print(f"{'='*80}")
    print(f"Task: {results['task_name']}")
    print(f"Trials: {results['num_trials']}")
    print(f"\n🎯 Main Metrics:")
    print(f"                          WITH RECOVERY      NO RECOVERY")
    print(f"   Success Rate:          {with_recovery['successful_trials']}/{results['num_trials']} ({100*with_recovery['success_rate']:5.1f}%)      {no_recovery['successful_trials']}/{results['num_trials']} ({100*no_recovery['success_rate']:5.1f}%)")
    print(f"   Avg Progress:          {100*with_recovery['avg_progress']:5.1f}%            {100*no_recovery['avg_progress']:5.1f}%")

    if eval_mode == 'above':
        print(f"\n🔄 Recovery Statistics:")
        print(f"   Total Above Pose Retries: {retry_stats['total_pose_retries']}")
        print(f"   Total Motion Planning Retries: {retry_stats['total_mp_retries']}")
        print(f"   Total VLA Retries: {retry_stats['total_vla_retries']}")
        print(f"   Pick Recovery Attempts: {retry_stats['total_pick_recoveries']}")
        print(f"   Successful Pick Recoveries: {retry_stats['successful_pick_recoveries']}")

    print(f"\n📝 Per-Trial Breakdown:")
    for trial in results['trials']:
        wr = trial['with_recovery']
        nr = trial['no_recovery']

        wr_status = '✅' if wr['task_success'] else '❌'
        nr_status = '✅' if nr['task_success'] else '❌'

        error_msg = f" [Error: {trial.get('error', 'N/A')}]" if 'error' in trial else ''

        print(f"   Trial {trial['trial_num']}:")
        print(f"     {wr_status} ({wr['skills_completed']}/{trial['total_skills']} skills, {100*wr['progress_ratio']:.1f}%) [With recovery]{error_msg}")

        # Add "no recovery" line with stop info if applicable
        nr_line = f"     {nr_status} ({nr['skills_completed']}/{trial['total_skills']} skills, {100*nr['progress_ratio']:.1f}%) [No recovery"
        if nr['stopped_at_skill']:
            nr_line += f" - stopped at '{nr['stopped_at_skill']}'"
        elif nr['task_success']:
            nr_line += " - all first attempts succeeded"
        nr_line += "]"
        print(nr_line)
        print()

    print(f"⚙️  Pipeline Configuration:")
    config = results.get('pipeline_config', {})
    print(f"   Benchmark: {config.get('benchmark_name', 'N/A')}")
    print(f"   Max VLA Retries: {config.get('max_vla_retries', 'N/A')}")
    print(f"   VLA Horizon: {config.get('vla_horizon', 'N/A')} steps")
    if eval_mode == 'above':
        print(f"   Above Height: {config.get('above_height', 'N/A')}m")
        print(f"   Motion Planner: {config.get('motion_planner_type', 'N/A')}")
        print(f"   Collision Aware: {config.get('collision_aware', 'N/A')}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 3 long horizon pipeline over multiple trials')

    # Evaluation parameters
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials to run (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory for result JSONs (default: <lilo-vla>/outputs/eval_results)')

    # Pipeline parameters (see lilo_vla/core/pipeline.py)
    parser.add_argument('--task_name', type=resolve_task_name,
                       default='Complete Kitchen Organization (Original)',
                       # Legacy names (…V1/V2/V3) are still accepted: `type` runs
                       # before `choices`, so resolve_task_name maps them first.
                       choices=[
                           'Complete Kitchen Organization (Original)',
                           'Organize Table (Original)',
                           'Cooking Preparation Setup (Original)',
                           'Turn On The Stove And Put The Moka Pot On It (Original)',
                           'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Original)',
                           'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Original)',
                           'Put Both The Cream Cheese Box And The Butter In The Basket (Original)',
                           'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Original)',
                           'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Original)',
                           'Complete Kitchen Organization (Reorder 1)',
                           'Organize Table (Reorder 1)',
                           'Cooking Preparation Setup (Reorder 1)',
                           'Complete Kitchen Organization (Reorder 2)',
                           'Organize Table (Reorder 2)',
                           'Cooking Preparation Setup (Reorder 2)',
                           'Turn On The Stove And Put The Moka Pot On It (Reorder 1)',
                           'Put Both The Alphabet Soup And The Cream Cheese Box In The Basket (Reorder 1)',
                           'Put Both The Alphabet Soup And The Tomato Sauce In The Basket (Reorder 1)',
                           'Put Both The Cream Cheese Box And The Butter In The Basket (Reorder 1)',
                           'Put The White Mug On The Left Plate And Put The Yellow And White Mug On The Right Plate (Reorder 1)',
                           'Put The White Mug On The Plate And Put The Chocolate Pudding To The Right Of The Plate (Reorder 1)',
                       ],
                       help='Task configuration to evaluate (21 total).')
    parser.add_argument('--unnorm_key', type=str, default=None,
                       help='Override the action de-normalisation key. Normally resolved from the checkpoint\'s dataset_statistics.json; set this only if that file is missing.')
    parser.add_argument('--vla_checkpoint', type=str, required=True,
                       help='Path to the trained LiLo-VLA checkpoint directory. NOTE: the directory '
                            'may be a local path or a Hugging Face Hub repo id.')
    parser.add_argument('--wrist_only', action='store_true', default=True)
    parser.add_argument('--visualize_poses', action='store_true')
    parser.add_argument('--ee_backward_offset', type=float, default=0.0)
    parser.add_argument('--motion_planner_method', type=str, default='cartesian_linear',
                       choices=['ik_setjoint', 'cartesian_linear'])
    parser.add_argument('--motion_planner_type', type=str, default='mplib', choices=['standard', 'mplib'],
                       help='Motion planner type: standard or mplib (default: mplib)')
    parser.add_argument('--continue_on_failure', action='store_true', default=False)
    parser.add_argument('--motion_planner_steps', type=int, default=400)
    parser.add_argument('--motion_planner_pos_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_ori_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_pos_threshold', type=float, default=0.02,
                       help='Position threshold for motion planning (default: 0.02m = 2cm)')
    parser.add_argument('--motion_planner_ori_threshold', type=float, default=0.524,
                       help='Orientation threshold for motion planning (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_pos_success_threshold', type=float, default=0.025,
                       help='Position success threshold (default: 0.025m = 2.5cm)')
    parser.add_argument('--motion_planner_ori_success_threshold', type=float, default=0.524,
                       help='Orientation success threshold (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_velocity_factor', type=float, default=0.9,
                       help='Velocity factor for motion planner (default: 0.9)')
    parser.add_argument('--max_vla_retries', type=int, default=10,
                       help='Maximum number of retry attempts for VLA execution (default: 3)')
    parser.add_argument('--vla_horizon', type=int, default=200,
                       help='Maximum steps per skill VLA execution (default: 200)')
    parser.add_argument('--verbose_vla_failure', action='store_true', default=False,
                       help='Print detailed diagnostic when VLA fails')
    parser.add_argument('--above_height', type=float, default=0.1,
                       help='Default height above object (meters, default: 0.03m = 3cm)')
    parser.add_argument('--benchmark', type=str, default='long_horizon_tasks_v1',
                       choices=['long_horizon_tasks_v1', 'long_horizon_tasks_libero_long',
                                'ultra_long', 'libero_long_plus_plus'],
                       help='Benchmark suite (default: long_horizon_tasks_v1). The two historical '
                            'names are aliased onto ultra_long / libero_long_plus_plus.')
    parser.add_argument('--recovery_mode', type=str, default='full', choices=['full', 'naive', 'naive_strict'],
                       help="Recovery strategy: 'full' = pose re-estimation + approach-pose reset + pick backtracking (default); 'naive' = in-place policy re-invocation only, same retry budget")

    # MPlib-specific parameters
    parser.add_argument('--mplib_collision_aware', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MPlib collision avoidance (default: True)')
    parser.add_argument('--mplib_joint_vel_limits', type=float, default=2.0,
                       help='MPlib joint velocity limits in rad/s (default: 2.0)')
    parser.add_argument('--mplib_joint_acc_limits', type=float, default=4.0,
                       help='MPlib joint acceleration limits in rad/s^2 (default: 4.0)')
    parser.add_argument('--mplib_waypoint_target_ratio', type=int, default=50,
                       help='MPlib target number of waypoints (default: 50)')
    parser.add_argument('--mplib_planning_time', type=float, default=1.0,
                       help='MPlib planning time budget in seconds (default: 1.0)')
    parser.add_argument('--mplib_safety_margin', type=float, default=1.1,
                       help='MPlib AABB safety margin multiplier (default: 1.1)')
    parser.add_argument('--mplib_erode_kernel_size', type=int, default=3,
                       help='MPlib mask erosion kernel size (default: 3)')
    parser.add_argument('--mplib_outlier_std_ratio', type=float, default=0.0,
                       help='MPlib 3D outlier removal std ratio (default: 0.0)')
    parser.add_argument('--mplib_save_pointcloud', action='store_true', default=False,
                       help='Save scene pointcloud and camera images for each MPlib planning call (default: False)')
    parser.add_argument('--apply_background_erasing', action='store_true', default=False,
                       help='Apply random erasing to wrist camera background (default: False)')
    parser.add_argument('--apply_distractor_masking', action='store_true', default=False,
                       help='Mask out distractor objects in wrist camera by blacking out their bounding boxes (default: False)')

    args = parser.parse_args()

    # Historical --benchmark names -> published suite names. New names pass through unchanged.
    args.benchmark = _BENCHMARK_ALIASES.get(args.benchmark, args.benchmark)

    # Prepare pipeline kwargs
    pipeline_kwargs = {
        'vla_checkpoint': args.vla_checkpoint,
        'task_name': args.task_name,
        'wrist_only': args.wrist_only,
        'visualize_poses': args.visualize_poses,
        'ee_offset': args.ee_backward_offset,
        'motion_planner_method': args.motion_planner_method,
        'motion_planner_type': args.motion_planner_type,
        'continue_on_failure': args.continue_on_failure,
        'mp_steps': args.motion_planner_steps,
        'mp_pos_gain': args.motion_planner_pos_gain,
        'mp_ori_gain': args.motion_planner_ori_gain,
        'mp_pos_threshold': args.motion_planner_pos_threshold,
        'mp_ori_threshold': args.motion_planner_ori_threshold,
        'mp_pos_success_threshold': args.motion_planner_pos_success_threshold,
        'mp_ori_success_threshold': args.motion_planner_ori_success_threshold,
        'mp_velocity_factor': args.motion_planner_velocity_factor,
        'max_vla_retries': args.max_vla_retries,
        'collision_aware': args.mplib_collision_aware,
        'mplib_joint_vel_limits': args.mplib_joint_vel_limits,
        'mplib_joint_acc_limits': args.mplib_joint_acc_limits,
        'mplib_waypoint_target_ratio': args.mplib_waypoint_target_ratio,
        'mplib_planning_time': args.mplib_planning_time,
        'mplib_safety_margin': args.mplib_safety_margin,
        'mplib_erode_kernel_size': args.mplib_erode_kernel_size,
        'mplib_outlier_std_ratio': args.mplib_outlier_std_ratio,
        'mplib_save_pointcloud': args.mplib_save_pointcloud,
        'vla_horizon': args.vla_horizon,
        'verbose_vla_failure': args.verbose_vla_failure,
        'above_height': args.above_height,
        'benchmark_name': args.benchmark,
        'unnorm_key': args.unnorm_key,
        'recovery_mode': args.recovery_mode,
        'apply_background_erasing': args.apply_background_erasing,
        'apply_distractor_masking': args.apply_distractor_masking,
    }

    print(f"\n🚀 Starting Long Horizon Evaluation")
    print(f"{'='*80}")
    print(f"Task: {args.task_name}")
    print(f"Trials: {args.num_trials}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Checkpoint: {Path(args.vla_checkpoint).name}")
    print(f"Max VLA Retries: {args.max_vla_retries}")
    print(f"Above Height: {args.above_height}m")
    print(f"Motion Planner: {args.motion_planner_type}")
    print(f"{'='*80}\n")

    # Run evaluation
    results = run_evaluation(args.num_trials, **pipeline_kwargs)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else _LILO_ROOT / "outputs" / "eval_results"
    save_results(results, output_dir, checkpoint=args.vla_checkpoint)

    # Print summary
    print_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
