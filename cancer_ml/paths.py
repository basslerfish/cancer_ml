"""
Handle paths.
"""
import argparse
from pathlib import Path


def get_arg_paths() -> dict:
    """Use argparse to get directories."""
    parser = argparse.ArgumentParser()
    arg_names = ["data_dir", "output_dir", "wandb_dir", "tb_dir", "config_file"]
    for arg_name in arg_names:
        parser.add_argument(f"--{arg_name}")
    args = parser.parse_args()

    directories = {}
    for arg_name in arg_names:
        this_arg = getattr(args, arg_name)
        if this_arg is not None:
            this_arg = Path(this_arg)
        key = arg_name.split("_")[0]
        directories[key] = this_arg
    return directories