"""Shared helpers for splitting records, building datasets, and resolving archives."""

import csv
import os
import random
import uuid
import zipfile
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from datasets.features.features import Features
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource


def validate_choices(
    values: Iterable[str],
    *,
    name: str,
    context: str,
    allowed: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Normalize and validate a named set of loader choices."""
    selected = (values,) if isinstance(values, str) else tuple(values)
    if not selected:
        raise ValueError(f"{context} requires at least one {name}.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{context} {name} selection must not contain duplicates.")
    if allowed is not None:
        invalid = sorted(set(selected) - set(allowed))
        if invalid:
            raise ValueError(
                f"Unknown {context} {name}: {invalid}; "
                f"valid values are {sorted(allowed)}."
            )
    return selected


def read_tsv_rows(path: str | Path) -> list[list[str]]:
    """Read a headerless TSV file as stripped, non-empty rows."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            [value.strip() for value in row]
            for row in csv.reader(handle, delimiter="\t")
            if row and any(value.strip() for value in row)
        ]


def read_tsv_dicts(
    path: str | Path,
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    """Read a headered TSV file, requiring exact columns and at least one data row."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(
                f"Unexpected TSV columns in {path}: "
                f"expected {list(fields)}, got {reader.fieldnames}."
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in fields):
                raise ValueError(
                    f"Malformed TSV row {line_number} in {path}: "
                    f"expected {len(fields)} fields."
                )
            stripped = {field: row[field].strip() for field in fields}
            if any(stripped.values()):
                rows.append(stripped)

    if not rows:
        raise ValueError(f"TSV file {path} is empty.")
    return rows


def split_counts(n, split_ratios):
    """Return ``(n_train, n_valid, n_eval)`` for ``n`` items given ``[train, valid]`` ratios.

    The eval ratio is whatever is left over. When there are at least three items, every split
    is guaranteed at least one item (so a small class never produces an empty valid/eval split);
    any rounding remainder goes to train.
    """
    train_ratio, val_ratio = split_ratios
    if n == 0:
        return 0, 0, 0
    if n < 3:
        # Too few to fill three splits; put everything in train.
        return n, 0, 0

    n_valid = max(1, round(val_ratio * n))
    n_eval = max(1, round((1 - train_ratio - val_ratio) * n))
    n_train = n - n_valid - n_eval
    if n_train < 1:
        # Ratios left nothing for train; take one back from the larger of valid/eval.
        if n_valid >= n_eval:
            n_valid -= 1
        else:
            n_eval -= 1
        n_train = 1
    return n_train, n_valid, n_eval


def build_audio_record(
    path,
    label,
    *,
    split="",
    source_dataset="",
    metadata_path="",
    channel_format=None,
    environment=None,
):
    """Build one canonical metadata record from an audio path and optional weak label.

    ``AudioDecoder`` reads only the stream metadata here (no decode), so this stays cheap.
    ``clip_offset`` is zero because file-tree loaders treat each file as one whole clip.
    Pass ``label=None`` for caption-only or otherwise unlabeled audio.
    """
    metadata = AudioDecoder(path).metadata
    duration = metadata.duration_seconds
    if duration is None:
        raise ValueError(f"Could not determine the duration for {path}.")
    num_channels = metadata.num_channels
    if num_channels is None:
        raise ValueError(f"Could not determine the channel count for {path}.")
    if channel_format is None:
        channel_format = {1: "mono", 2: "stereo"}.get(num_channels, f"{num_channels}-channel")

    if label is None:
        class_list = []
    elif isinstance(label, str):
        class_list = [label]
    elif isinstance(label, Iterable):
        class_list = [str(item) for item in label]
    else:
        class_list = [str(label)]

    if environment is None:
        environment = class_list[0] if len(class_list) == 1 else ""

    return {
        "audio_path": str(path),
        "sample_rate": metadata.sample_rate,
        "clip_offset": 0.0,
        "clip_duration": duration,
        "class_list": class_list,
        "split": split,
        "source_dataset": source_dataset,
        "metadata_path": str(metadata_path),
        "events": [],
        "num_channels": num_channels,
        "channel_format": channel_format,
        "environment": str(environment),
    }


def splits_to_audio_dataset(
    splits: Mapping[str, Iterable[dict[str, Any]] | Dataset],
    *,
    features: Features | None = None,
    label_source: LabelSource | str = LabelSource.UNKNOWN,
) -> AudioDataset:
    """Build an ``AudioDataset`` from split record iterables or existing datasets."""
    data = {}
    for split, rows in splits.items():
        if isinstance(rows, Dataset):
            data[split] = rows
            continue

        records = list(rows)
        if records or features is None:
            data[split] = Dataset.from_list(records, features=features)
        else:
            data[split] = Dataset.from_dict({column: [] for column in features}, features=features)
    return AudioDataset(data=DatasetDict(data), label_source=label_source)


def split_records_by_key(
    records: Iterable[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    split_ratios,
    seed=42,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle and split records into train/valid/eval independently within each key group.

    Use this only for datasets without official splits. The key can represent a class label,
    recording environment, or another group that should contribute independently to each split.
    Groups with fewer than three records are placed entirely in train by :func:`split_counts`.
    """
    rng = random.Random(seed)
    by_key = {}
    for record in records:
        by_key.setdefault(key_fn(record), []).append(record)

    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "valid", "eval")}
    for group in by_key.values():
        rng.shuffle(group)
        n_train, n_valid, _ = split_counts(len(group), split_ratios)
        split_records["train"].extend(group[:n_train])
        split_records["valid"].extend(group[n_train:n_train + n_valid])
        split_records["eval"].extend(group[n_train + n_valid:])

    return split_records


def find_unique(root: Path, name: str, dataset_name: str) -> Path:
    """Return the sole file named ``name`` beneath ``root``."""
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"{dataset_name} file {name!r} not found under {root}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {dataset_name} files named {name!r} found under {root}."
        )
    return matches[0]


def audio_by_name(root: Path, dataset_name: str) -> dict[str, Path]:
    """Index WAV files beneath ``root`` by unique basename."""
    audio = {}
    for path in sorted(root.rglob("*.wav")):
        if path.name in audio:
            raise ValueError(
                f"Multiple {dataset_name} audio files named {path.name!r} found under "
                f"{root}."
            )
        audio[path.name] = path
    return audio


def validate_zip_members(
    archive: zipfile.ZipFile,
    destination: str | Path,
    members: Iterable[zipfile.ZipInfo] | None = None,
) -> tuple[zipfile.ZipInfo, ...]:
    """Return ZIP members after checking that each one stays inside the destination."""
    selected = tuple(archive.infolist() if members is None else members)
    root = Path(destination).resolve()
    for member in selected:
        if not (root / member.filename).resolve().is_relative_to(root):
            raise ValueError(f"Unsafe ZIP member path: {member.filename!r}.")
    return selected


def extracted_archive_dir(root: str, split_archives: Mapping[str, str], split: str) -> str:
    """Return the directory where ``download_zenodo`` extracts the archive for a split."""
    return os.path.join(root, os.path.splitext(split_archives[split])[0])


def archive_stems_for_splits(split_archives: Mapping[str, str], splits: Iterable[str]) -> list[str]:
    """Return sorted archive stems for requested splits present in ``split_archives``."""
    return sorted({
        os.path.splitext(split_archives[split])[0]
        for split in splits
        if split in split_archives
    })


def build_audio_dataset(
    labelled_files,
    split_ratios,
    seed=42,
    *,
    source_dataset=None,
    channel_format=None,
    label_source: LabelSource | str = LabelSource.UNKNOWN,
):
    """Build an ``AudioDataset`` from ``(path, label)`` pairs, split per label.

    Only use this for datasets that have **no official train/test splits**. If the source
    defines splits (e.g. FSD50K, BirdSet, LOCATA), the loader must honour them directly
    instead of calling this function, otherwise official test-set files leak into training.

    Each label's records are shuffled deterministically and split independently into
    train/valid/eval.

    Args:
        labelled_files: Iterable of ``(audio_path, label)`` tuples.
        split_ratios: ``[train_ratio, valid_ratio]``; eval gets the remainder.
        seed: Seed for the per-label shuffle.
        label_source: Dataset-level provenance for the labels.

    Returns:
        AudioDataset: Canonical records with a plain string ``audio_path``.
    """

    if source_dataset is None:
        source_dataset = str(uuid.uuid4())

    records = (
        build_audio_record(
            path,
            label,
            source_dataset=source_dataset,
            channel_format=channel_format,
        )
        for path, label in labelled_files
    )
    split_records = split_records_by_key(
        records,
        key_fn=lambda record: tuple(record["class_list"]),
        split_ratios=split_ratios,
        seed=seed,
    )
    for split, rows in split_records.items():
        for record in rows:
            record["split"] = split

    return splits_to_audio_dataset(
        split_records,
        features=DATA_FEATURES,
        label_source=label_source,
    )
