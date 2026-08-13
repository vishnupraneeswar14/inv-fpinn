import argparse
import json

import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def coerce(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def apply_overrides(cfg, overrides):
    for override in overrides:
        key, _, raw = override.partition("=")
        set_nested(cfg, key, coerce(raw))
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Fractional SDOF PINN inverse identification")
    parser.add_argument("--config", default="config.yaml", help="path to yaml config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set training.iters=1000")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    return cfg