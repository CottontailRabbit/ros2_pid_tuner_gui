"""Read / write ros2_control controller gain YAML files.

Works with any controller that stores per-joint PID gains under a
``gains.<joint>.<field>`` mapping and lists its joints via ``dof_names``
(pid_controller) or ``joints`` (joint_trajectory_controller).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


# Parameters that hold the controller's DOF list, in detection order.
DOF_PARAMS = ('dof_names', 'joints')


def _find_controller_section(doc: dict) -> tuple[dict, str, str]:
    """
    Locate a controller section that carries a ``gains`` mapping.

    Supports wildcard ('/**/<controller>') and explicit ('/<ns>/<controller>')
    keys, for both pid_controller and joint_trajectory_controller.

    Returns (params_dict, full_key, dof_param).
    """
    if not isinstance(doc, dict):
        raise ValueError('YAML root is not a mapping.')
    for key, val in doc.items():
        if not isinstance(val, dict):
            continue
        params = val.get('ros__parameters', val)
        if not isinstance(params, dict) or 'gains' not in params:
            continue
        for dof_param in DOF_PARAMS:
            if dof_param in params:
                return params, key, dof_param
    raise KeyError(
        "No controller section with a 'gains' block and a "
        "'dof_names'/'joints' list found in YAML.")


def load_gains(yaml_path: str | Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Load controller gains from a yaml file.

    Returns:
        (joint_names, {joint: {<field>: value, ...}})
    Every numeric gain field present for a joint is preserved.
    """
    path = Path(yaml_path)
    with path.open('r', encoding='utf-8') as f:
        doc = yaml.safe_load(f)

    params, _, dof_param = _find_controller_section(doc)
    joints = list(params.get(dof_param, []) or [])
    raw_gains = params.get('gains', {}) or {}

    gains: dict[str, dict[str, float]] = {}
    for j in joints:
        g = raw_gains.get(j, {}) or {}
        parsed: dict[str, float] = {}
        for field_name, value in g.items():
            try:
                parsed[field_name] = float(value)
            except (TypeError, ValueError):
                continue
        gains[j] = parsed
    return joints, gains


def save_gains(
    yaml_path: str | Path,
    joints: Iterable[str],
    gains: dict[str, dict[str, float]],
) -> None:
    """
    Write gains back into the yaml. Preserves other keys; only updates
    each joint's entries inside the controller `gains` block, leaving gain
    fields not present in `gains[joint]` untouched.
    """
    path = Path(yaml_path)
    with path.open('r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}

    params, key, _ = _find_controller_section(doc)
    existing = params.get('gains', {}) or {}
    for j in joints:
        g = gains.get(j, {})
        merged = dict(existing.get(j, {}) or {})
        for field_name, value in g.items():
            merged[field_name] = float(value)
        existing[j] = merged
    params['gains'] = existing
    if isinstance(doc.get(key), dict) and 'ros__parameters' in doc[key]:
        doc[key]['ros__parameters'] = params
    else:
        doc[key] = params

    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
