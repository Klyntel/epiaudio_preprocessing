"""Load the TUT 2017 acoustic-scene and sound-event datasets into an AudioDataset.

These are the DCASE 2017 challenge datasets. The development records provide four official
folds; the selected fold becomes ``train`` and ``valid``. The separately published evaluation
record becomes ``eval``. The acoustic-scene task covers 15 scenes; the sound-event task covers
a single ``street`` scene with six event classes.

Run ``python -m audio_preprocessing.dataset.load_tut2017`` from the repo root to download and
build the sound-event task.
"""

from __future__ import annotations

from pathlib import Path

from audio_preprocessing.dataset._tut import TaskConfig, TUTDCASELoader

TASKS = {
    "acoustic_scenes": TaskConfig(
        source_dataset="TUT Acoustic Scenes 2017",
        development_record="400515",
        evaluation_record="1040168",
        release_prefix="TUT-acoustic-scenes-2017",
    ),
    "sound_events": TaskConfig(
        source_dataset="TUT Sound Events 2017",
        development_record="814831",
        evaluation_record="1040179",
        release_prefix="TUT-sound-events-2017",
        scenes=("street",),
    ),
}


class TUT2017Loader(TUTDCASELoader):
    """Download and index either TUT 2017 benchmark task using its official partitions."""

    TASKS = TASKS
    DATASET_LABEL = "TUT 2017"
    DEFAULT_ROOT = Path("data/tut2017")


def main():
    return TUT2017Loader()()


if __name__ == "__main__":
    tut2017 = main()
    print(tut2017.info())
