"""
cex_config.py -- config loader with local-only enforcement.

Reads config/cex_config.yaml.
Raises RuntimeError if any code tries to call cloud when local_only=true.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("cex_config")

_DEFAULT_CONFIG = {
    "inference": {
        "backend": "local",
        "local_only": True,
        "preferred_local": ["dml", "cuda", "rocm", "npu", "cpu"],
        "lan_discovery": True,
        "lan_subnet": "192.168.1.0/24",
        "npu_port": 7474,
        "gpu_port": 7478,
    },
    "cloud": {
        "enabled": False,
        "provider": None,
        "api_key_env": None,
        "endpoint": None,
        "model": None,
    },
    "ngp": {
        "enabled": False,
        "backend": "auto",
        "memory_fraction": 0.90,
        "scene_scale": 1.0,
        "n_steps": 35000,
        "snapshot_dir": "./snapshots",
    },
    "nvidia": {
        "cuda_device": 0,
        "tcc_mode": False,
        "allow_growth": True,
        "persistent_mode": False,
    },
    "amd": {
        "dml_device": 0,
        "rocm_device": 0,
        "vulkan_device": 0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _find_config() -> Optional[Path]:
    candidates = [
        Path("config/cex_config.yaml"),
        Path(__file__).parent.parent / "config" / "cex_config.yaml",
        Path(os.environ.get("CEX_CONFIG", "")) if os.environ.get("CEX_CONFIG") else None,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def load() -> dict:
    """Load config from YAML file merged over defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    path = _find_config()

    if path:
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, user_cfg)
            log.debug("Config loaded from %s", path)
        except Exception as exc:
            log.warning("Could not load %s: %s -- using defaults", path, exc)
    else:
        log.info("No cex_config.yaml found -- using defaults (local_only=true)")

    return cfg


# Module-level singleton -- loaded once per process
_cfg: Optional[dict] = None


def get() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load()
    return _cfg


def is_local_only() -> bool:
    return bool(get()["inference"].get("local_only", True))


def is_cloud_allowed() -> bool:
    cfg = get()
    if is_local_only():
        return False
    return bool(cfg["cloud"].get("enabled", False))


def assert_local(context: str = "") -> None:
    """
    Call this before any cloud API call.
    Raises RuntimeError if local_only is true (default).
    """
    if not is_cloud_allowed():
        msg = (
            f"Cloud inference blocked: local_only=true. "
            f"Context: {context}. "
            "To enable cloud: set inference.local_only=false AND cloud.enabled=true "
            "in config/cex_config.yaml."
        )
        raise RuntimeError(msg)


def get_preferred_local() -> list:
    return get()["inference"].get("preferred_local", ["dml", "cuda", "cpu"])


def get_ngp_config() -> dict:
    return get().get("ngp", {})


def get_nvidia_config() -> dict:
    return get().get("nvidia", {})


def get_amd_config() -> dict:
    return get().get("amd", {})


def get_lan_config() -> dict:
    inf = get()["inference"]
    return {
        "subnet":   inf.get("lan_subnet", "192.168.1.0/24"),
        "npu_port": inf.get("npu_port", 7474),
        "gpu_port": inf.get("gpu_port", 7478),
        "enabled":  inf.get("lan_discovery", True),
    }


def summary() -> str:
    cfg = get()
    inf = cfg["inference"]
    cloud = cfg["cloud"]
    lines = [
        f"backend       : {inf['backend']}",
        f"local_only    : {inf.get('local_only', True)}",
        f"cloud.enabled : {cloud.get('enabled', False)}",
        f"cloud.provider: {cloud.get('provider', 'none')}",
        f"ngp.enabled   : {cfg['ngp'].get('enabled', False)}",
        f"ngp.backend   : {cfg['ngp'].get('backend', 'auto')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(summary())
