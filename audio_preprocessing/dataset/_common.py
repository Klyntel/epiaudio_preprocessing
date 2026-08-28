"""Record-building helpers shared by the retained loaders."""

import random
from collections.abc import Iterable, Mapping

from datasets import Dataset, DatasetDict

from audio_preprocessing.datasets import AudioDataset, LabelSource


def build_audio_record(
    path,
    label,
    *,
    split="",
    source_dataset="",
    metadata_path="",
    channel_format=None,
):
    from torchcodec.decoders import AudioDecoder

    metadata = AudioDecoder(path).metadata
    if metadata.duration_seconds is None or metadata.num_channels is None:
        raise ValueError(f"Incomplete audio metadata for {path}")
    classes = [] if label is None else [str(label)]
    return {
        "audio_path": str(path),
        "sample_rate": metadata.sample_rate,
        "clip_offset": 0.0,
        "clip_duration": metadata.duration_seconds,
        "class_list": classes,
        "split": split,
        "source_dataset": source_dataset,
        "metadata_path": str(metadata_path),
        "events": [],
        "num_channels": metadata.num_channels,
        "channel_format": channel_format
        or {1: "mono", 2: "stereo"}.get(metadata.num_channels, f"{metadata.num_channels}-channel"),
        "environment": classes[0] if classes else "",
    }


def splits_to_audio_dataset(
    splits: Mapping[str, Iterable[dict] | Dataset],
    *,
    label_source: LabelSource | str = LabelSource.UNKNOWN,
) -> AudioDataset:
    data = {
        name: rows if isinstance(rows, Dataset) else Dataset.from_list(list(rows))
        for name, rows in splits.items()
    }
    return AudioDataset(data=DatasetDict(data), label_source=label_source)


def build_audio_dataset(
    labelled_files,
    split_ratios,
    seed=42,
    *,
    source_dataset="",
    channel_format=None,
    label_source: LabelSource | str = LabelSource.UNKNOWN,
):
    by_label: dict[str, list[dict]] = {}
    for path, label in labelled_files:
        row = build_audio_record(
            path,
            label,
            source_dataset=source_dataset,
            channel_format=channel_format,
        )
        by_label.setdefault(str(label), []).append(row)

    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "eval": []}
    train_ratio, valid_ratio = split_ratios
    for rows in by_label.values():
        rng.shuffle(rows)
        n = len(rows)
        n_valid = max(1, round(valid_ratio * n)) if n >= 3 else 0
        n_eval = max(1, round((1 - train_ratio - valid_ratio) * n)) if n >= 3 else 0
        n_train = n - n_valid - n_eval
        for name, selected in (
            ("train", rows[:n_train]),
            ("valid", rows[n_train : n_train + n_valid]),
            ("eval", rows[n_train + n_valid :]),
        ):
            for row in selected:
                row["split"] = name
            splits[name].extend(selected)
    return splits_to_audio_dataset(splits, label_source=label_source)

