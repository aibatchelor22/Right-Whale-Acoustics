import json
from pathlib import Path
from datetime import datetime


def make_run_dir(base_dir):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path(base_dir)
        / f"run_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir


def save_json(
    data,
    out_path
):

    with open(out_path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )
