"""Load DataSED environmental recordings and strong sound-event annotations."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset._common import (
    audio_by_name,
    find_unique,
    split_records_by_key,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, Event, LabelSource

RECORD_ID = "15346092"
ANNOTATION_FILES = {
    "monophonic": "Monophonic_sound_detection.csv",
    "polyphonic": "Polyphonic_sound_detection.csv",
}
ANNOTATION_COLUMNS = {
    "sound_name",
    "class_name",
    "start_perc",
    "end_perc",
    "start_time",
    "end_time",
    "event_length",
}


def _read_annotations(metadata_path: Path) -> dict[str, list[dict]]:
    annotations: dict[str, list[dict]] = defaultdict(list)
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = ANNOTATION_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"DataSED metadata {metadata_path} is missing columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            sound_name = (row["sound_name"] or "").strip()
            label = (row["class_name"] or "").strip()
            if not sound_name or not label:
                raise ValueError(
                    f"DataSED metadata {metadata_path} line {line_number} has an empty "
                    "sound_name or class_name."
                )
            try:
                start = float(row["start_time"])
                end = float(row["end_time"])
                event_length = float(row["event_length"])
                start_fraction = float(row["start_perc"])
                end_fraction = float(row["end_perc"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"DataSED metadata {metadata_path} line {line_number} contains "
                    "non-numeric event bounds."
                ) from error

            values = (start, end, event_length, start_fraction, end_fraction)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"DataSED metadata {metadata_path} line {line_number} contains "
                    "non-finite event bounds."
                )
            if not 0 <= start_fraction <= end_fraction <= 1:
                raise ValueError(
                    f"DataSED metadata {metadata_path} line {line_number} contains "
                    "invalid fractional event bounds."
                )
            if abs((end - start) - event_length) > 0.011:
                raise ValueError(
                    f"DataSED metadata {metadata_path} line {line_number} has an "
                    "inconsistent event_length."
                )

            annotations[sound_name].append(
                Event.build_start_end(
                    start,
                    end,
                    label,
                    additional_metadata={
                        "start_fraction": start_fraction,
                        "end_fraction": end_fraction,
                    },
                ).to_dict()
            )

    if not annotations:
        raise ValueError(f"DataSED metadata {metadata_path} contains no annotations.")
    return dict(annotations)


def collect(root: Path, annotation_version: str) -> list[dict]:
    """Return one canonical record per recording in the selected annotation CSV."""
    metadata_path = find_unique(
        root,
        ANNOTATION_FILES[annotation_version],
        "DataSED",
    )
    annotations = _read_annotations(metadata_path)
    audio_files = audio_by_name(root, "DataSED")

    records = []
    for sound_name, events in annotations.items():
        try:
            audio_path = audio_files[sound_name]
        except KeyError:
            raise FileNotFoundError(
                f"DataSED audio file {sound_name!r} referenced by {metadata_path} "
                f"was not found under {root}."
            ) from None

        audio = AudioDecoder(str(audio_path)).metadata
        if audio.sample_rate != 44100 or audio.num_channels not in (1, 2):
            raise ValueError(
                f"DataSED audio {audio_path} must be mono or stereo 44.1 kHz; "
                f"found {audio.num_channels} channel(s) at {audio.sample_rate} Hz."
            )
        if audio.duration_seconds is None:
            raise ValueError(f"Could not determine the duration for {audio_path}.")
        latest_event_end = max(event["offset"] + event["duration"] for event in events)
        if latest_event_end > audio.duration_seconds + 0.011:
            raise ValueError(
                f"DataSED annotations for {audio_path} extend past the audio duration."
            )

        records.append(
            {
                "audio_path": str(audio_path),
                "sample_rate": audio.sample_rate,
                "clip_offset": 0.0,
                "clip_duration": audio.duration_seconds,
                "class_list": sorted({event["label"] for event in events}),
                "split": "",
                "source_dataset": "DataSED",
                "metadata_path": str(metadata_path),
                "events": events,
                "num_channels": audio.num_channels,
                "channel_format": "mono" if audio.num_channels == 1 else "stereo",
                "environment": "outdoor_environmental_noise",
            }
        )
    return records


class DataSEDLoader(ZenodoLoader):
    """Download and index DataSED with monophonic or polyphonic strong labels."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        annotation_version: str = "polyphonic",
        split_ratios: list[float] | tuple[float, float] = (0.8, 0.1),
        seed: int = 42,
        prepare: bool = True,
    ) -> None:
        if annotation_version not in ANNOTATION_FILES:
            raise ValueError(
                f"Unknown DataSED annotation_version {annotation_version!r}; valid versions: "
                f"{sorted(ANNOTATION_FILES)}"
            )
        super().__init__(root=root, prepare=prepare)
        self.annotation_version = annotation_version
        self.split_ratios = split_ratios
        self.seed = seed

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "DataSEDLoader needs a root path. Use prepare=False with an existing local "
                "extraction, or prepare=True to download DataSED."
            )

        records = collect(self.root, self.annotation_version)
        splits = split_records_by_key(
            records,
            key_fn=lambda _: "all_recordings",
            split_ratios=self.split_ratios,
            seed=self.seed,
        )
        for split, rows in splits.items():
            for record in rows:
                record["split"] = split
        return splits_to_audio_dataset(
            splits,
            features=DATA_FEATURES,
            label_source=self.label_source,
        )
