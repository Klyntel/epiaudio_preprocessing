"""Load the official NonSpeech7k train and test splits from Zenodo."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]

from audio_preprocessing.dataset._common import (
    audio_by_name,
    find_unique,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


RECORD_ID = "6967442"
SOURCE_DATASET = "NonSpeech7k"
CLASS_NAMES = {
    0: "breath",
    1: "cough",
    2: "crying",
    3: "laugh",
    4: "screaming",
    5: "sneeze",
    6: "yawn",
}
SPLIT_FILES = {
    "train": ("train.zip", "metadata of train set .csv"),
    "eval": ("test.zip", "metadata of test set.csv"),
}
COLUMN_MAP = {
    "Filename": "filename",
    "File ID": "file_id",
    "File_ID": "file_id",
    "Duration in ms": "duration_ms",
    "Durationin ms": "duration_ms",
    "Class ID": "class_id",
    "Class_id": "class_id",
    "Classname": "class_name",
    "augmentation  id": "augmentation_id",
    "Augment Id": "augmentation_id",
    "Augmentation  type": "augmentation_type",
    "Augmentation type": "augmentation_type",
    "source": "source_url",
}

NONSPEECH7K_FEATURES = Features(
    {
        **DATA_FEATURES,
        "file_id": Value("string"),
        "class_id": Value("int64"),
        "reported_duration_ms": Value("int64"),
        "augmentation_id": Value("int64"),
        "augmentation_type": Value("string"),
        "source_url": Value("string"),
    }
)


def _parse_int(value: str, *, field: str, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"NonSpeech7k metadata {path} line {line_number} has invalid {field} "
            f"{value!r}."
        ) from error


def _read_metadata(path: Path, split: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        canonical_columns = [COLUMN_MAP.get(column) for column in columns]
        expected_columns = set(COLUMN_MAP.values())
        if (
            None in canonical_columns
            or len(canonical_columns) != len(set(canonical_columns))
            or set(canonical_columns) != expected_columns
        ):
            raise ValueError(
                f"Unexpected NonSpeech7k columns in {path}: expected "
                f"{sorted(expected_columns)}, got {reader.fieldnames}."
            )

        rows = []
        filenames = set()
        for line_number, source_row in enumerate(reader, start=2):
            if None in source_row or any(
                source_row[column] is None for column in columns
            ):
                raise ValueError(
                    f"Malformed NonSpeech7k metadata row {line_number} in {path}."
                )
            row = {
                COLUMN_MAP[column]: source_row[column].strip() for column in columns
            }
            if any(not value for value in row.values()):
                raise ValueError(
                    f"NonSpeech7k metadata {path} line {line_number} has an empty field."
                )

            filename = row["filename"]
            if (
                Path(filename).name != filename
                or Path(filename).suffix.lower() != ".wav"
            ):
                raise ValueError(
                    f"NonSpeech7k metadata {path} line {line_number} has invalid filename "
                    f"{filename!r}."
                )
            if filename in filenames:
                raise ValueError(
                    f"Duplicate NonSpeech7k filename {filename!r} in {path}."
                )
            filenames.add(filename)

            duration_ms = _parse_int(
                row["duration_ms"],
                field="duration",
                path=path,
                line_number=line_number,
            )
            class_id = _parse_int(
                row["class_id"],
                field="class ID",
                path=path,
                line_number=line_number,
            )
            augmentation_id = _parse_int(
                row["augmentation_id"],
                field="augmentation ID",
                path=path,
                line_number=line_number,
            )
            if duration_ms <= 0:
                raise ValueError(
                    f"NonSpeech7k metadata {path} line {line_number} has non-positive "
                    "duration."
                )
            class_name = row["class_name"].lower().replace("yawm", "yawn")
            if CLASS_NAMES.get(class_id) != class_name:
                raise ValueError(
                    f"NonSpeech7k metadata {path} line {line_number} has conflicting "
                    f"class ID/name {class_id}/{row['class_name']!r}."
                )
            if augmentation_id != 0 or row["augmentation_type"].lower() not in {
                "original",
                "orignal",
            }:
                raise ValueError(
                    f"NonSpeech7k metadata {path} line {line_number} has unsupported "
                    "augmentation metadata."
                )

            rows.append(
                {
                    **row,
                    "duration_ms": duration_ms,
                    "class_id": class_id,
                    "class_name": class_name,
                    "augmentation_id": augmentation_id,
                    "augmentation_type": "original",
                    "split": split,
                    "metadata_path": path,
                }
            )

    if not rows:
        raise ValueError(f"NonSpeech7k metadata {path} contains no rows.")
    return rows


def _build_record(audio_path: Path, spec: dict) -> dict:
    audio = AudioDecoder(str(audio_path)).metadata
    if audio.sample_rate != 32000 or audio.num_channels != 1:
        raise ValueError(
            f"NonSpeech7k audio {audio_path} must be mono 32 kHz; found "
            f"{audio.num_channels} channel(s) at {audio.sample_rate} Hz."
        )
    if audio.duration_seconds is None:
        raise ValueError(f"Could not determine the duration for {audio_path}.")

    return {
        "audio_path": str(audio_path),
        "sample_rate": audio.sample_rate,
        "clip_offset": 0.0,
        "clip_duration": audio.duration_seconds,
        "class_list": [spec["class_name"]],
        "split": spec["split"],
        "source_dataset": SOURCE_DATASET,
        "metadata_path": str(spec["metadata_path"]),
        "events": [],
        "num_channels": 1,
        "channel_format": "mono",
        "environment": spec["class_name"],
        "file_id": spec["file_id"],
        "class_id": spec["class_id"],
        "reported_duration_ms": spec["duration_ms"],
        "augmentation_id": spec["augmentation_id"],
        "augmentation_type": spec["augmentation_type"],
        "source_url": spec["source_url"],
    }


class NonSpeech7kLoader(ZenodoLoader):
    """Download selected official splits and index human non-speech vocal sounds."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        splits: Iterable[str] | str | None = None,
        prepare: bool = True,
    ) -> None:
        requested = (
            tuple(SPLIT_FILES)
            if splits is None
            else ((splits,) if isinstance(splits, str) else tuple(splits))
        )
        if not requested:
            raise ValueError("NonSpeech7kLoader requires at least one split.")
        if len(set(requested)) != len(requested):
            raise ValueError("NonSpeech7kLoader splits must not contain duplicates.")
        invalid = sorted(set(requested) - set(SPLIT_FILES))
        if invalid:
            raise ValueError(
                f"Unknown NonSpeech7k split(s) {invalid}; valid splits are "
                f"{sorted(SPLIT_FILES)}."
            )

        self.splits = tuple(split for split in SPLIT_FILES if split in requested)
        only = [filename for split in self.splits for filename in SPLIT_FILES[split]]
        super().__init__(root=root, only=only, prepare=prepare)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "NonSpeech7kLoader needs a root path. Use prepare=False with an existing "
                "extraction, or prepare=True to download selected official splits."
            )

        specs = {
            split: _read_metadata(
                find_unique(self.root, SPLIT_FILES[split][1], SOURCE_DATASET),
                split,
            )
            for split in self.splits
        }
        filenames = [
            spec["filename"] for split_specs in specs.values() for spec in split_specs
        ]
        if len(filenames) != len(set(filenames)):
            raise ValueError(
                "NonSpeech7k metadata repeats a filename across official splits."
            )

        audio = audio_by_name(self.root, SOURCE_DATASET)
        if not audio:
            raise FileNotFoundError(f"No NonSpeech7k WAV files found under {self.root}.")
        records: dict[str, list[dict]] = {}
        for split, split_specs in specs.items():
            records[split] = []
            for spec in split_specs:
                try:
                    audio_path = audio[spec["filename"]]
                except KeyError:
                    raise FileNotFoundError(
                        f"NonSpeech7k audio file {spec['filename']!r} referenced by "
                        f"{spec['metadata_path']} was not found under {self.root}."
                    ) from None
                records[split].append(_build_record(audio_path, spec))

        dataset = splits_to_audio_dataset(
            records,
            features=NONSPEECH7K_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = next(
            split for split, rows in dataset.data.items() if rows.num_rows
        )
        return dataset


def main() -> AudioDataset:
    return NonSpeech7kLoader(splits=["eval"])()


if __name__ == "__main__":
    nonspeech7k = main()
    print(nonspeech7k.info())
