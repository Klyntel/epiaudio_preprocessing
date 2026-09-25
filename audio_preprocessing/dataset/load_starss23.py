# Adapted from: https://github.com/Klyntel/EpiAudio/blob/3da6d8b045f2eddac6115b912cff75033da80e53/epiaudio/dataset/load_starss23.py
"""Load STARSS23 FOA audio and frame-level spatial metadata.

STARSS23 (Sony-TAu Realistic Spatial Soundscapes 2023) is a collection of
first-order Ambisonics (FOA) recordings of real spatial scenes, annotated with
frame-level sound event localization and detection labels (class, direction of
arrival, and distance).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import (
    archive_stems_for_splits,
    extracted_archive_dir,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset._dcase_task3_common import (
    DCASE_TASK3_CLASSES,
    clip_events_to_audio,
    dcase_task3_identity,
    parse_dcase_task3_metadata_csv,
    parse_dcase_task3_metadata_row,
)
from audio_preprocessing.dataset._spatial_common import (
    FOA_DATA_FEATURES,
    frames_to_events,
    scan_foa_dev,
    scan_foa_eval,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource

ZENODO_RECORD = "7880637"
SOURCE_DATASET = "STARSS23"

SPLIT_ARCHIVES = {
    "train": "foa_dev.zip",
    "test": "foa_dev.zip",
    "eval": "foa_eval.zip",
}
VALID_ORIGINS = ("sony", "tau")

FRAME_RATE = 10

STARSS23_CLASSES = DCASE_TASK3_CLASSES

STARSS23_FEATURES = Features({
    **FOA_DATA_FEATURES,
    "sample_id": Value("string"),
    "origin": Value("string"),
})

DEFAULT_ROOT = os.path.join("data", "starss23_raw")

_DEV_NAME = re.compile(r"fold(\d+)_room(\d+)_mix(\d+)", re.IGNORECASE)
_EVAL_NAME = re.compile(r"^mix(\d+)$", re.IGNORECASE)


def _selection(
    name: str,
    values: Sequence[str],
    allowed: Sequence[str],
) -> tuple[str, ...]:
    selected = (values,) if isinstance(values, str) else tuple(values)
    unknown = [value for value in selected if value not in allowed]
    if not selected or unknown:
        raise ValueError(
            f"Invalid STARSS23 {name} {unknown or list(selected)}; valid: {list(allowed)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"STARSS23 {name} must not contain duplicates")
    return selected


def _sample_id(row: dict) -> str:
    """Return a stable STARSS23 source ID independent of the extraction root."""
    parts = ["starss23", str(row["split"])]
    origin = row.get("origin")
    if origin is not None:
        parts.append(str(origin))
    parts.append(Path(row["audio_path"]).stem)
    return ":".join(parts)


_parse_metadata_row = parse_dcase_task3_metadata_row
_parse_metadata_csv = parse_dcase_task3_metadata_csv
starss_identity = dcase_task3_identity


def _metadata_csv_path(metadata_dir: str, audio_path: str, split_dir: str) -> str:
    """Resolve metadata across flat and archive-wrapper directory layouts."""
    relative_parts = os.path.relpath(audio_path, split_dir).split(os.sep)
    dev_index = next(
        (index for index, part in enumerate(relative_parts) if part.startswith("dev-")),
        0,
    )
    relative_stem = os.path.splitext(os.path.join(*relative_parts[dev_index:]))[0]
    candidates = (
        os.path.join(metadata_dir, relative_stem + ".csv"),
        os.path.join(
            metadata_dir,
            os.path.basename(metadata_dir),
            relative_stem + ".csv",
        ),
    )
    existing = [candidate for candidate in candidates if os.path.isfile(candidate)]
    if not existing:
        raise FileNotFoundError(
            f"Missing STARSS23 metadata CSV for {audio_path!r}; expected {candidates[0]!r}"
        )
    if len(existing) > 1:
        raise ValueError(
            f"Multiple STARSS23 metadata CSVs match {audio_path!r}: {existing}"
        )
    return existing[0]


def _dev_context(origins: Sequence[str]):
    allowed_origins = set(origins)

    def context(path: str, _clip_ids: dict[str, int]) -> tuple[str, dict] | None:
        subset = next((part for part in path.split(os.sep) if part.startswith("dev-")), None)
        if subset is None or subset.count("-") != 2:
            return None
        _, split, origin = subset.split("-")
        if origin not in allowed_origins:
            return None
        return split, {"origin": origin}

    return context


_clip_events_to_audio = clip_events_to_audio


def _scan_dev(
    split_dir: str,
    metadata_dir: str,
    origins: Sequence[str],
    keep_splits: Sequence[str] | None = None,
) -> dict[str, list[dict]]:
    rows_by_split = scan_foa_dev(
        split_dir,
        metadata_dir,
        source_dataset=SOURCE_DATASET,
        filename_re=_DEV_NAME,
        split_context_fn=_dev_context(origins),
        metadata_path_fn=_metadata_csv_path,
        parse_metadata_csv=_parse_metadata_csv,
        aggregate_fn=frames_to_events(
            identity=starss_identity,
            frames_per_sec=FRAME_RATE,
            has_distance=True,
        ),
        keep_splits=keep_splits,
    )
    for rows in rows_by_split.values():
        for row in rows:
            row["events"] = _clip_events_to_audio(
                row["events"],
                row["clip_duration"],
            )
    return rows_by_split


def _scan_eval(split_dir: str) -> list[dict]:
    return scan_foa_eval(
        split_dir,
        source_dataset=SOURCE_DATASET,
        filename_re=_EVAL_NAME,
        extra_null_fields=("origin",),
    )


def scan(
    root: str | Path = DEFAULT_ROOT,
    *,
    splits: Sequence[str] = ("train", "test"),
    origins: Sequence[str] = VALID_ORIGINS,
) -> dict[str, list[dict]]:
    """Walk extracted STARSS23 files and return records grouped by split."""
    splits = _selection("splits", splits, tuple(SPLIT_ARCHIVES))
    origins = _selection("origins", origins, VALID_ORIGINS)
    root = str(root)
    metadata_dir = os.path.join(root, "metadata_dev")
    rows_by_split: dict[str, list[dict]] = {}

    dev_splits = [split for split in splits if split != "eval"]
    if dev_splits:
        dev_dirs = {
            extracted_archive_dir(root, SPLIT_ARCHIVES, split)
            for split in dev_splits
        }
        if len(dev_dirs) != 1:
            raise ValueError(
                f"STARSS23 dev splits map to multiple archives: {sorted(dev_dirs)}"
            )
        dev_rows = _scan_dev(
            dev_dirs.pop(),
            metadata_dir,
            origins,
            keep_splits=dev_splits,
        )
        for split in dev_splits:
            rows_by_split[split] = dev_rows.get(split, [])

    if "eval" in splits:
        eval_dir = extracted_archive_dir(root, SPLIT_ARCHIVES, "eval")
        rows_by_split["eval"] = _scan_eval(eval_dir)

    for rows in rows_by_split.values():
        for row in rows:
            row["sample_id"] = _sample_id(row)

    return rows_by_split


class STARSS23Loader(ZenodoLoader):
    """Prepare and build STARSS23 with official splits and spatial metadata."""

    record_id = ZENODO_RECORD
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        splits: Sequence[str] = ("train", "test"),
        origins: Sequence[str] = VALID_ORIGINS,
        prepare: bool = True,
    ) -> None:
        self.splits = _selection("splits", splits, tuple(SPLIT_ARCHIVES))
        self.origins = _selection("origins", origins, VALID_ORIGINS)
        only = archive_stems_for_splits(SPLIT_ARCHIVES, self.splits)
        if any(split != "eval" for split in self.splits):
            only.append("metadata_dev")
        super().__init__(root=root, only=only, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "STARSS23Loader needs a root path. "
                "Use prepare=False with an existing local root."
            )

        root = str(self.root)
        rows_by_split = scan(
            root=root,
            splits=self.splits,
            origins=self.origins,
        )
        data = {}
        for split in self.splits:
            rows = rows_by_split.get(split, [])
            if not rows:
                split_dir = extracted_archive_dir(root, SPLIT_ARCHIVES, split)
                raise RuntimeError(
                    f"No STARSS23 FOA clips found for split {split!r} under {split_dir!r}"
                )
            data[split] = rows

        return splits_to_audio_dataset(
            data,
            features=STARSS23_FEATURES,
            label_source=self.label_source,
        )


def main() -> AudioDataset:
    # foa_dev.zip is the smallest labeled audio archive (~3.4 GB).
    return STARSS23Loader(splits=("train",), origins=("sony",))()


if __name__ == "__main__":
    dataset = main()
    print(dataset.info())
