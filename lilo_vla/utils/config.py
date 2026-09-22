#!/usr/bin/env python3
"""
Config loader for unified task configuration.

This module provides utilities to load and access the unified task configuration
that combines all annotation data from long_horizon_tasks.json, skills_objects_mapping_updated.csv,
and object mappings.

Usage:
    from lilo_vla.utils.config import load_config, get_task_skills, get_skill_mapping

    config = load_config()
    skills = get_task_skills(config, "Complete Kitchen Organization")
    mapping = get_skill_mapping(config, "pick black bowl 1")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# lilo-vla/  (this file lives at lilo-vla/lilo_vla/utils/config.py)
_LILO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _LILO_ROOT / "configs" / "tasks_and_skills.json"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load the unified task configuration.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary containing the complete configuration
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def get_task_skills(config: Dict[str, Any], task_name: str) -> Optional[List[str]]:
    """
    Get the skills for a specific task.

    Args:
        config: Loaded configuration dictionary
        task_name: Name of the task

    Returns:
        List of skill names for the task, or None if task not found
    """
    for task in config.get("long_horizon_tasks", []):
        if task["name"] == task_name:
            return task["skills"]
    return None


def get_skill_mapping(config: Dict[str, Any], skill_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the mapping information for a specific skill.

    Args:
        config: Loaded configuration dictionary
        skill_name: Name of the skill

    Returns:
        Dictionary containing skill mapping info, or None if skill not found
    """
    return config.get("skill_mappings", {}).get(skill_name)


def get_object_mapping(config: Dict[str, Any], logical_object: str) -> Optional[str]:
    """
    Get the physical object name for a logical object.

    Args:
        config: Loaded configuration dictionary
        logical_object: Logical object name (e.g., "bowl_1")

    Returns:
        Physical object name (e.g., "akita_black_bowl_1"), or None if not found
    """
    return config.get("object_mappings", {}).get(logical_object)


def get_target_object(config: Dict[str, Any], skill_name: str) -> Optional[str]:
    """
    Get the target object for a specific skill.

    Args:
        config: Loaded configuration dictionary
        skill_name: Name of the skill

    Returns:
        Target object name, or None if skill not found
    """
    skill_mapping = get_skill_mapping(config, skill_name)
    if skill_mapping:
        return skill_mapping.get("target_object")
    return None




def get_task_by_name(config: Dict[str, Any], task_name: str) -> Optional[Dict[str, Any]]:
    """
    Get complete task information by name.

    Args:
        config: Loaded configuration dictionary
        task_name: Name of the task

    Returns:
        Complete task dictionary, or None if task not found
    """
    for task in config.get("long_horizon_tasks", []):
        if task["name"] == task_name:
            return task
    return None


# Global config instance for caching
_cached_config = None


def get_cached_config() -> Dict[str, Any]:
    """
    Get cached config instance to avoid repeated file reads.

    Returns:
        Cached configuration dictionary
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def refresh_config():
    """Refresh the cached config by reloading from file."""
    global _cached_config
    _cached_config = None
    return get_cached_config()