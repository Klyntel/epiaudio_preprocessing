"""Load TAU-NIGENS 2021 FOA audio and frame-level spatial metadata.

TAU-NIGENS 2021 is the DCASE 2021 Task 3 sound event localization and
detection corpus. Development clips use the official fold protocol:
folds 1–4 are training, fold 5 is validation, and fold 6 is testing.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Sequence
from pathlib import Path

from audio_preprocessing.dataset._common import (
    archive_stems_for_splits,
    extracted_archive_dir,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset._spatial_common import (
    FOA_DATA_FEATURES,
    FrameRow,
    frames_to_events,
    parse_frame_csv,
    scan_foa_dev,
    scan_foa_eval,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource

ZENODO_RECORD = "4844825"
SOURCE_DATASET = "TAU-NIGENS21"

SPLIT_ARCHIVES = {
    "train": "foa_dev.zip",
    "valid": "foa_dev.zip",
    "test": "foa_dev.zip",
    "eval": "foa_eval.zip",
}
FOLD_SPLIT = {
    1: "train",
    2: "train",
    3: "train",
    4: "train",
    5: "valid",
    6: "test",
}

METADATA_FRAMES_PER_SECOND = 10

TAU_NIGENS21_CLASSES = {
    0: "alarm",
    1: "crying baby",
    2: "crash",
    3: "barking dog",
    4: "female scream",
    5: "female speech",
    6: "footsteps",
    7: "knocking on door",
    8: "male scream",
    9: "male speech",
    10: "ringing phone",
    11: "piano",
}

DEFAULT_ROOT = os.path.join("data", "tau_nigens21_raw")


def group_tau_event_tracks(frame_rows: list[FrameRow]) -> list[list[FrameRow]]:
    """Group frame rows by TAU event ID and class."""
    groups: dict[tuple[int, str], list[FrameRow]] = {}
    for row in frame_rows:
        groups.setdefault((row["source_idx"], row["class"]), []).append(row)
    return list(groups.values())


def _parse_metadata_csv(csv_path: str) -> list[FrameRow]:
    def parse_row(columns: list[str]) -> FrameRow:
        frame, class_index, event_index, azimuth, elevation = columns
        class_index = int(class_index)
        return {
            "frame": int(frame),
            "class": TAU_NIGENS21_CLASSES.get(class_index, str(class_index)),
            # TAU's event index is the stable source identifier this schema calls source_idx.
            "source_idx": int(event_index),
            # TAU azimuth: 0° front, increasing ccw (+90° left).
            "azimuth": float(azimuth),
            "elevation": float(elevation),
        }

    return parse_frame_csv(csv_path, parse_row)


def _dev_context(
    _path: str,
    clip_ids: dict[str, int],
) -> tuple[str, dict] | None:
    split = FOLD_SPLIT.get(clip_ids["fold"])
    if split is None:
        return None
    environment = f"fold{clip_ids['fold']}_room{clip_ids['room_id']}"
    return split, {"environment": environment}


def _metadata_path_from_index(csv_index: dict[str, str]):
    def metadata_path(
        _metadata_dir: str,
        audio_path: str,
        _dev_dir: str,
    ) -> str | None:
        return csv_index.get(os.path.splitext(os.path.basename(audio_path))[0])

    return metadata_path


def _scan_dev(
    dev_dir: str,
    metadata_dir: str,
    keep_splits: Sequence[str] | None = None,
) -> dict[str, list[dict]]:
    csv_index = {
        os.path.splitext(os.path.basename(csv_path))[0]: csv_path
        for csv_path in glob.glob(
            os.path.join(metadata_dir, "**", "*.csv"),
            recursive=True,
        )
    }
    return scan_foa_dev(
        dev_dir,
        metadata_dir,
        source_dataset=SOURCE_DATASET,
        filename_re=re.compile(
            r"fold(\d+)_room(\d+)_mix(\d+)",
            re.IGNORECASE,
        ),
        split_context_fn=_dev_context,
        metadata_path_fn=_metadata_path_from_index(csv_index),
        parse_metadata_csv=_parse_metadata_csv,
        aggregate_fn=frames_to_events(
            identity=group_tau_event_tracks,
            frames_per_sec=METADATA_FRAMES_PER_SECOND,
        ),
        keep_splits=keep_splits,
    )


def _scan_eval(eval_dir: str) -> list[dict]:
    return scan_foa_eval(
        eval_dir,
        source_dataset=SOURCE_DATASET,
        filename_re=re.compile(r"^mix(\d+)$", re.IGNORECASE),
    )


def scan(
    root: str | Path = DEFAULT_ROOT,
    *,
    splits: Sequence[str] = ("train", "valid", "test"),
) -> dict[str, list[dict]]:
    """Walk extracted TAU-NIGENS21 files and return records grouped by split."""
    root = str(root)
    metadata_dir = os.path.join(root, "metadata_dev")
    rows_by_split: dict[str, list[dict]] = {}

    dev_splits = [split for split in splits if split != "eval"]
    if dev_splits:
        dev_dirs = {
            extracted_archive_dir(root, SPLIT_ARCHIVES, split) for split in dev_splits
        }
        if len(dev_dirs) != 1:
            raise ValueError(
                f"TAU-NIGENS21 dev splits map to multiple archives: {sorted(dev_dirs)}"
            )
        dev_rows = _scan_dev(
            dev_dirs.pop(),
            metadata_dir,
            keep_splits=dev_splits,
        )
        for split in dev_splits:
            rows_by_split[split] = dev_rows.get(split, [])

    if "eval" in splits:
        eval_dir = extracted_archive_dir(root, SPLIT_ARCHIVES, "eval")
        rows_by_split["eval"] = _scan_eval(eval_dir)

    return rows_by_split


class TAUNIGENS21Loader(ZenodoLoader):
    """Prepare and build TAU-NIGENS21 with official folds and spatial labels."""

    record_id = ZENODO_RECORD
    label_source = LabelSource.SYNTHETIC

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        splits: Sequence[str] = ("train", "valid", "test"),
        prepare: bool = True,
    ) -> None:
        unknown = [split for split in splits if split not in SPLIT_ARCHIVES]
        if unknown:
            raise ValueError(
                f"Unknown split(s) {unknown}; valid: {list(SPLIT_ARCHIVES)}"
            )

        self.splits = tuple(splits)
        only = archive_stems_for_splits(SPLIT_ARCHIVES, self.splits)
        if any(split != "eval" for split in self.splits):
            only.append("metadata_dev")
        super().__init__(root=root, only=only, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "TAUNIGENS21Loader needs a root path. "
                "Use prepare=False with an existing local root."
            )

        root = str(self.root)
        rows_by_split = scan(root=root, splits=self.splits)
        data = {}
        for split in self.splits:
            rows = rows_by_split.get(split, [])
            if not rows:
                split_dir = extracted_archive_dir(root, SPLIT_ARCHIVES, split)
                raise RuntimeError(
                    f"No TAU-NIGENS21 FOA clips found for split {split!r} "
                    f"under {split_dir!r}"
                )
            data[split] = rows

        return splits_to_audio_dataset(
            data,
            features=FOA_DATA_FEATURES,
            label_source=self.label_source,
        )
