#!/usr/bin/env python3
"""
Task Sequence Planner for Long Horizon Manipulation Tasks.

Simplified version that uses ONLY tasks_and_skills.json.
All legacy CSV/JSON loading and fuzzy matching has been removed.
"""

from typing import List, Optional

# Import config loader
from lilo_vla.utils.config import get_cached_config


def plan_task_sequence(task_name: str) -> Optional[List[str]]:
    """
    Get skill sequence for a long horizon task.

    This is the main function called by execute_long_horizon_pipeline.py.

    Args:
        task_name: Name of the long horizon task (e.g., "Pick Black Bowl")

    Returns:
        List of skill names in execution order, or None if task not found

    Example:
        >>> plan_task_sequence("Pick Black Bowl")
        ["pick black bowl"]
    """
    config = get_cached_config()

    # Find matching task (case-insensitive)
    task_name_lower = task_name.lower().strip()

    for task in config['long_horizon_tasks']:
        if task['name'].lower().strip() == task_name_lower:
            skills = task['skills']

            print(f"✅ Found task: {task['name']}")
            print(f"   Task ID: {task['task_id']}")
            print(f"   Skills: {len(skills)}")
            for i, skill in enumerate(skills, 1):
                print(f"     {i}. {skill}")

            return skills

    # Task not found
    available_tasks = [task['name'] for task in config['long_horizon_tasks']]
    print(f"❌ Task '{task_name}' not found")
    print(f"📋 Available tasks ({len(available_tasks)}):")
    for task in available_tasks:
        print(f"   - {task}")

    return None


def get_task_info(task_name: str) -> Optional[dict]:
    """
    Get complete task information from config.

    Args:
        task_name: Name of the long horizon task

    Returns:
        Task dictionary or None if not found
    """
    config = get_cached_config()
    task_name_lower = task_name.lower().strip()

    for task in config['long_horizon_tasks']:
        if task['name'].lower().strip() == task_name_lower:
            return task

    return None


def get_all_tasks() -> List[str]:
    """Get list of all available task names."""
    config = get_cached_config()
    return [task['name'] for task in config['long_horizon_tasks']]


def main():
    """Test function."""
    import argparse

    parser = argparse.ArgumentParser(description='Plan task sequences')
    parser.add_argument('--task', type=str, required=True,
                       help='Name of the long horizon task to plan')
    args = parser.parse_args()

    print("🎯 Task Sequence Planner (Simplified)")
    print(f"📁 Using: tasks_and_skills.json")
    print()

    skills = plan_task_sequence(args.task)

    if skills:
        print("\n✅ Task planning completed!")
    else:
        print("\n❌ Task planning failed")


if __name__ == "__main__":
    main()
