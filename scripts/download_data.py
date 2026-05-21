#!/usr/bin/env python3

"""
Download right whale datasets from NOAA.

Usage:

    python scripts/download_data.py

Requires:
    gsutil
"""

import subprocess
from pathlib import Path

# =========================================
# DATASETS
# =========================================

DATASETS = [

    {
        "name": "sbnms_200903_nopp6_ch10",

        "audio": (
            "gs://noaa-passive-bioacoustic/"
            "dclde/2013/"
            "nefsc_sbnms_200903_nopp6_ch10/"
            "source-audio/*.wav"
        ),

        "detections": (
            "gs://noaa-passive-bioacoustic/"
            "dclde/2013/"
            "nefsc_sbnms_200903_nopp6_ch10/"
            "detections/*.csv"
        ),

        "metadata": (
            "gs://noaa-passive-bioacoustic/"
            "dclde/2013/"
            "nefsc_sbnms_200903_nopp6_ch10/"
            "metadata/*.json"
        )
    }

]

# =========================================
# ROOT DIRECTORY
# =========================================

ROOT = Path("data/right_whale_data")

# =========================================
# DOWNLOAD FUNCTION
# =========================================

def gsutil_copy(
    source,
    destination
):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [
        "gsutil",
        "-m",
        "cp",
        source,
        str(destination)
    ]

    print()
    print("RUNNING:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True
    )

# =========================================
# MAIN
# =========================================

def main():

    for ds in DATASETS:

        print()
        print("=" * 60)
        print("DOWNLOADING:", ds["name"])
        print("=" * 60)

        dataset_dir = ROOT / ds["name"]

        audio_dir = dataset_dir / "audio"
        detections_dir = dataset_dir / "detections"
        metadata_dir = dataset_dir / "metadata"

        # ---------------------------------
        # AUDIO
        # ---------------------------------

        gsutil_copy(
            ds["audio"],
            audio_dir
        )

        # ---------------------------------
        # DETECTIONS
        # ---------------------------------

        gsutil_copy(
            ds["detections"],
            detections_dir
        )

        # ---------------------------------
        # METADATA
        # ---------------------------------

        gsutil_copy(
            ds["metadata"],
            metadata_dir
        )

    print()
    print("DONE.")

# =========================================

if __name__ == "__main__":

    main()
