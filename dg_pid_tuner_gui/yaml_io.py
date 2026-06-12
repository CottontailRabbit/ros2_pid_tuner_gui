"""Read / write ros2_control PidController gain YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


GAIN_KEYS = ('p', 'i', 'd', 'i_clamp_max', 'i_clamp_min', 'antiwindup', 'feedforward_gain')


def _find_pid_controller_section(doc: dict) -> tuple[dict, str]:
    """
    Locate the 'pid_controller' top-level key in a parameter YAML.

    Supports both '/**/pid_controller' (wildcard) and '/<ns>/pid_controller'.
    Returns (params_dict, full_key).
    """
    if not isinstance(doc, dict):
        raise ValueError('YAML root is not a mapping.')
    for key, val in doc.items():
        if not isinstance(val, dict):
            continue
        if key.endswith('/pid_controller') or key == 'pid_controller':
            params = val.get('ros__parameters', val)
            return params, key
    raise KeyError("No '*/pid_controller' section found in YAML.")


def load_gains(yaml_path: str | Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Load pid_controller gains from a yaml file.

    Returns:
        (joint_names, {joint: {p, i, d, i_clamp_max, i_clamp_min, ...}})
    """
    path = Path(yaml_path)
    with path.open('r', encoding='utf-8') as f:
        doc = yaml.safe_load(f)

    params, _ = _find_pid_controller_section(doc)
    joints = list(params.get('dof_names', []) or [])
    raw_gains = params.get('gains', {}) or {}

    gains: dict[str, dict[str, float]] = {}
    for j in joints:
        g = raw_gains.get(j, {}) or {}
        gains[j] = {
            'p': float(g.get('p', 0.0)),
            'i': float(g.get('i', 0.0)),
            'd': float(g.get('d', 0.0)),
            'i_clamp_max': float(g.get('i_clamp_max', 0.0)),
            'i_clamp_min': float(g.get('i_clamp_min', 0.0)),
        }
    return joints, gains


def save_gains(
    yaml_path: str | Path,
    joints: Iterable[str],
    gains: dict[str, dict[str, float]],
) -> None:
    """
    Write gains back into the yaml. Preserves other keys; only updates
    the pid_controller `gains` block.
    """
    path = Path(yaml_path)
    with path.open('r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}

    params, key = _find_pid_controller_section(doc)
    new_gains = {}
    for j in joints:
        g = gains.get(j, {})
        new_gains[j] = {
            'p': float(g.get('p', 0.0)),
            'i': float(g.get('i', 0.0)),
            'd': float(g.get('d', 0.0)),
            'i_clamp_max': float(g.get('i_clamp_max', 0.0)),
            'i_clamp_min': float(g.get('i_clamp_min', 0.0)),
        }
    params['gains'] = new_gains
    doc[key]['ros__parameters'] = params

    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
