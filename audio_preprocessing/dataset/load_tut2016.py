"""Load the TUT 2016 acoustic-scene and sound-event datasets into an AudioDataset.

The development records provide four official folds; the selected fold becomes ``train`` and
``valid``. The separately published evaluation record becomes ``eval``.

Run ``python -m audio_preprocessing.dataset.load_tut2016`` from the repo root to download and
build the sound-event task.
"""

from __future__ import annotations

from pathlib import Path

from audio_preprocessing.dataset._tut import TaskConfig, TUTDCASELoader

TASKS = {
    "acoustic_scenes": TaskConfig(
        source_dataset="TUT Acoustic Scenes 2016",
        development_record="45739",
        evaluation_record="165995",
        release_prefix="TUT-acoustic-scenes-2016",
    ),
    "sound_events": TaskConfig(
        source_dataset="TUT Sound Events 2016",
        development_record="45759",
        evaluation_record="996424",
        release_prefix="TUT-sound-events-2016",
        scenes=("home", "residential_area"),
    ),
}


class TUT2016Loader(TUTDCASELoader):
    """Download and index either TUT 2016 benchmark task using its official partitions."""

    TASKS = TASKS
    DATASET_LABEL = "TUT 2016"
    DEFAULT_ROOT = Path("data/tut2016")


def main():
    return TUT2016Loader()()


if __name__ == "__main__":
    tut2016 = main()
    print(tut2016.info())
