"""Load MultiVox vocal performances and all bundled microphone captures.

MultiVox is distributed across two Zenodo records whose condition archives bundle
FOA, ORTF, and near-field audio. The loader therefore exposes one combined dataset
and can limit preparation and local scanning to conditions C1--C4.
"""

# Adapted from: https://github.com/Klyntel/EpiAudio/blob/656c2b35965a44bf25309aea2d29f86249ccc4c3/epiaudio/dataset/load_multivox.py

from __future__ import annotations

import csv
import math
import random
import re
from collections.abc import Iterable
from pathlib import Path

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import (
    build_audio_record,
    find_unique,
    split_counts,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

PART1_RECORD_ID = "17058101"
PART2_RECORD_ID = "17065497"
SOURCE_DATASET = "MultiVox"
CONDITIONS = ("C1", "C2", "C3", "C4")
DEFAULT_ROOT = Path("data") / f"zenodo_{PART1_RECORD_ID}+{PART2_RECORD_ID}"

_METADATA_COLUMNS = (
    "Date",
    "Session_ID",
    "Recording_Space",
    "Song",
    "Path",
    "Duration",
    "Key",
    "Singing_Group",
    "Vocal_Content",
    "Condition",
    "Locations_At_Circle",
    "Empty_Circle_Locations",
    "Instructor_Visible",
    "Visible_People",
    "Degree",
    "Facing_Direction",
    "Singer_Roles",
    "Singer_Height",
    "Gender",
    "Nearfield_Files_Captured",
)

MULTIVOX_FEATURES = Features({
    **DATA_FEATURES,
    "date": Value("string"),
    "session_id": Value("string"),
    "recording_space": Value("string"),
    "song": Value("string"),
    "performance_id": Value("string"),
    "metadata_duration": Value("float64"),
    "key": Value("string"),
    "singing_group": Value("string"),
    "vocal_content": Value("string"),
    "condition": Value("string"),
    "locations_at_circle": Value("int64"),
    "empty_circle_locations": Value("int64"),
    "instructor_visible": Value("bool"),
    "visible_people": Value("int64"),
    "degree": Value("string"),
    "facing_direction": Value("string"),
    "singer_roles": Value("string"),
    "singer_height": Value("string"),
    "gender": Value("string"),
    "nearfield_files_captured": Value("string"),
    "audio_capture_type": Value("string"),
})

_CAPTURE_FILENAME = re.compile(r"^(?P<capture>.+)_Song.+[.]wav$", re.IGNORECASE)


def _select_conditions(conditions: str | Iterable[str] | None) -> tuple[str, ...]:
    if conditions is None:
        selected = CONDITIONS
    elif isinstance(conditions, str):
        selected = (conditions,)
    else:
        try:
            selected = tuple(conditions)
        except TypeError:
            raise TypeError("conditions must be 'C1'--'C4' or an iterable of them.") from None

    if not selected:
        raise ValueError("MultiVoxLoader requires at least one condition.")
    unknown = [condition for condition in selected if condition not in CONDITIONS]
    if unknown:
        choices = ", ".join(CONDITIONS)
        raise ValueError(f"Unknown MultiVox condition {unknown[0]!r}. Choose from: {choices}")
    if len(set(selected)) != len(selected):
        raise ValueError("MultiVox conditions must not contain duplicates.")
    return selected


def _validate_split_ratios(split_ratios: Iterable[float]) -> tuple[float, float]:
    try:
        train_ratio, valid_ratio = (float(value) for value in split_ratios)
    except (TypeError, ValueError):
        raise ValueError("split_ratios must contain exactly two numeric proportions.") from None

    eval_ratio = 1.0 - train_ratio - valid_ratio
    if not all(
        math.isfinite(ratio) and ratio > 0
        for ratio in (train_ratio, valid_ratio, eval_ratio)
    ):
        raise ValueError(
            "split_ratios must define positive, finite train, validation, and evaluation "
            "proportions."
        )
    return train_ratio, valid_ratio


def _parse_float(value: str, *, field: str, line_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(
            f"Invalid {field} value {value!r} on metadata row {line_number}; expected a number."
        ) from None
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Invalid {field} value {value!r} on metadata row {line_number}; "
            "expected a finite non-negative number."
        )
    return parsed


def _parse_count(value: str, *, field: str, line_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid {field} value {value!r} on metadata row {line_number}; "
            "expected an integer."
        ) from None
    if parsed < 0:
        raise ValueError(
            f"Invalid {field} value {value!r} on metadata row {line_number}; "
            "expected a non-negative integer."
        )
    return parsed


def _parse_bool(value: str, *, field: str, line_number: int) -> bool:
    normalized = value.casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(
        f"Invalid {field} value {value!r} on metadata row {line_number}; expected a boolean."
    )


def _read_metadata(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # The official CSV includes a trailing space in its "Date " header.
        fieldnames = tuple(field.strip() for field in (reader.fieldnames or ()))
        reader.fieldnames = list(fieldnames)
        missing = [field for field in _METADATA_COLUMNS if field not in fieldnames]
        if missing:
            raise ValueError(f"MultiVox metadata {path} is missing columns: {', '.join(missing)}")

        by_performance: dict[str, dict[str, object]] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in _METADATA_COLUMNS):
                raise ValueError(f"Malformed MultiVox metadata row {line_number} in {path}.")
            values = {field: row[field].strip() for field in _METADATA_COLUMNS}

            performance_id = values["Path"]
            if not performance_id:
                raise ValueError(f"Missing Path on MultiVox metadata row {line_number} in {path}.")
            if performance_id in by_performance:
                raise ValueError(f"Duplicate MultiVox performance ID {performance_id!r} in {path}.")

            song = values["Song"]
            recording_space = values["Recording_Space"]
            condition = values["Condition"]
            if not song:
                raise ValueError(f"Missing Song for MultiVox performance {performance_id!r}.")
            if not recording_space:
                raise ValueError(
                    f"Missing Recording_Space for MultiVox performance {performance_id!r}."
                )
            if condition not in CONDITIONS:
                raise ValueError(
                    f"Unknown Condition {condition!r} for MultiVox performance {performance_id!r}."
                )

            by_performance[performance_id] = {
                "date": values["Date"],
                "session_id": values["Session_ID"],
                "recording_space": recording_space,
                "song": song,
                "performance_id": performance_id,
                "metadata_duration": _parse_float(
                    values["Duration"], field="Duration", line_number=line_number
                ),
                "key": values["Key"],
                "singing_group": values["Singing_Group"],
                "vocal_content": values["Vocal_Content"],
                "condition": condition,
                "locations_at_circle": _parse_count(
                    values["Locations_At_Circle"],
                    field="Locations_At_Circle",
                    line_number=line_number,
                ),
                "empty_circle_locations": _parse_count(
                    values["Empty_Circle_Locations"],
                    field="Empty_Circle_Locations",
                    line_number=line_number,
                ),
                "instructor_visible": _parse_bool(
                    values["Instructor_Visible"],
                    field="Instructor_Visible",
                    line_number=line_number,
                ),
                "visible_people": _parse_count(
                    values["Visible_People"],
                    field="Visible_People",
                    line_number=line_number,
                ),
                "degree": values["Degree"],
                "facing_direction": values["Facing_Direction"],
                "singer_roles": values["Singer_Roles"],
                "singer_height": values["Singer_Height"],
                "gender": values["Gender"],
                "nearfield_files_captured": values["Nearfield_Files_Captured"],
            }

    if not by_performance:
        raise ValueError(f"MultiVox metadata {path} contains no performances.")
    return by_performance


def _match_performance_id(
    audio_path: Path,
    root: Path,
    known_performance_ids: set[str],
) -> str | None:
    parent = audio_path.parent
    while True:
        if parent.name in known_performance_ids:
            return parent.name
        if parent == root:
            return None
        parent = parent.parent


def _capture_type(audio_path: Path) -> str:
    match = _CAPTURE_FILENAME.fullmatch(audio_path.name)
    if match is None:
        raise ValueError(
            f"Invalid MultiVox audio filename {audio_path.name!r}; "
            "expected '<capture>_Song...wav'."
        )
    return match.group("capture")


def _split_performances(
    records: list[dict[str, object]],
    split_ratios: tuple[float, float],
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    performances_by_song: dict[str, set[str]] = {}
    for record in records:
        performance_id = str(record["performance_id"])
        song = str(record["song"])
        performances_by_song.setdefault(song, set()).add(performance_id)

    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for song in sorted(performances_by_song):
        performance_ids = sorted(performances_by_song[song])
        rng.shuffle(performance_ids)
        n_train, n_valid, _ = split_counts(len(performance_ids), split_ratios)
        for performance_id in performance_ids[:n_train]:
            assignment[performance_id] = "train"
        for performance_id in performance_ids[n_train:n_train + n_valid]:
            assignment[performance_id] = "valid"
        for performance_id in performance_ids[n_train + n_valid:]:
            assignment[performance_id] = "eval"

    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "valid": [],
        "eval": [],
    }
    for source_record in records:
        record = dict(source_record)
        split = assignment[str(record["performance_id"])]
        record["split"] = split
        splits[split].append(record)
    return splits


class MultiVoxLoader(ZenodoLoader):
    """Build one canonical dataset from selected MultiVox recording conditions.

    Preparation downloads ``metadata.csv`` plus the selected condition archives from
    both Zenodo records. The full four-condition download is roughly 81.5 GB compressed;
    use ``prepare=False`` to build from an existing local extraction.
    """

    record_ids = (PART1_RECORD_ID, PART2_RECORD_ID)
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        conditions: str | Iterable[str] | None = None,
        split_ratios: Iterable[float] = (0.8, 0.1),
        seed: int = 42,
        prepare: bool = True,
    ) -> None:
        selected_conditions = _select_conditions(conditions)
        validated_ratios = _validate_split_ratios(split_ratios)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        super().__init__(
            root=root,
            only=["metadata.csv", *selected_conditions],
            prepare=prepare,
        )
        self.conditions = selected_conditions
        self.split_ratios = validated_ratios
        self.seed = seed

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "MultiVoxLoader needs a prepared root. Call it with prepare=True or pass "
                "root=... with prepare=False."
            )
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"MultiVox root does not exist or is not a directory: {root}")

        metadata_path = find_unique(root, "metadata.csv", SOURCE_DATASET)
        metadata_by_performance = _read_metadata(metadata_path)
        known_performance_ids = set(metadata_by_performance)

        audio_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".wav"
        )
        selected_conditions = set(self.conditions)
        records: list[dict[str, object]] = []
        for audio_path in audio_paths:
            performance_id = _match_performance_id(
                audio_path,
                root,
                known_performance_ids,
            )
            if performance_id is None:
                raise ValueError(
                    f"Could not match MultiVox audio {audio_path} to a metadata Path ancestor."
                )

            metadata = metadata_by_performance[performance_id]
            if metadata["condition"] not in selected_conditions:
                continue

            record = build_audio_record(
                audio_path,
                [str(metadata["song"])],
                source_dataset=SOURCE_DATASET,
                metadata_path=str(metadata_path),
                environment=metadata["recording_space"],
            )
            record.update(metadata)
            record["audio_capture_type"] = _capture_type(audio_path)
            records.append(record)

        if not records:
            conditions = ", ".join(self.conditions)
            raise ValueError(f"No MultiVox WAV audio found for selected conditions: {conditions}")

        splits = _split_performances(records, self.split_ratios, self.seed)
        return splits_to_audio_dataset(
            splits,
            features=MULTIVOX_FEATURES,
            label_source=self.label_source,
        )


def main() -> None:
    """Build from the conventional prepared root without starting a large download."""
    dataset = MultiVoxLoader(root=DEFAULT_ROOT, prepare=False)()
    print(dataset)


if __name__ == "__main__":
    main()
