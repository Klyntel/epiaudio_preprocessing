"""Shared metadata and scan helpers for FOA spatial dataset loaders.

See ``audio_preprocessing/dataset/README.md#spatial-conventions`` for the
canonical spatial metadata and FOA audio conventions.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import warnings
from collections.abc import Callable, Container, Hashable
from re import Pattern
from typing import Any

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import build_audio_record
from audio_preprocessing.datasets import DATA_FEATURES, ContinuousData, ContinuousEvent


FrameRow = dict[str, Any]
# Dataset policy: turn one clip's frame rows into track groups. Each group is a
# single source trajectory (one row per frame) for contiguous-run splitting.
TrackIdentity = Callable[[list[FrameRow]], list[list[FrameRow]]]

FOA_DATA_FEATURES = Features({
    **DATA_FEATURES,
    "fold": Value("int64"),
    "room_id": Value("int64"),
    "mix": Value("int64"),
})


def normalize_azimuth(value: str | float) -> float:
    """Wrap a finite azimuth to the canonical ``[-180, 180]`` degree range."""
    azimuth = float(value)
    if not math.isfinite(azimuth):
        raise ValueError(f"Azimuth must be finite, got {azimuth}.")
    normalized = (azimuth + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 and azimuth > 0 else normalized


def parse_frame_csv(
    csv_path: str,
    row_parser: Callable[[list[str]], FrameRow],
) -> list[FrameRow]:
    """Parse a headerless frame metadata CSV with a dataset-specific row parser."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for line_number, columns in enumerate(csv.reader(handle), start=1):
            if not columns or all(not column.strip() for column in columns):
                continue
            try:
                rows.append(row_parser([column.strip() for column in columns]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid spatial metadata row {line_number} in {csv_path!r}: {error}"
                ) from error
    return rows


def _contiguous_frame_runs(rows: list[FrameRow]) -> list[list[FrameRow]]:
    """Split one source track into runs separated by missing frame indices."""
    runs = []
    current = []
    for row in sorted(rows, key=lambda item: item["frame"]):
        if current and row["frame"] == current[-1]["frame"]:
            raise ValueError(
                "A spatial event has duplicate metadata rows for "
                f"frame {row['frame']}."
            )
        if current and row["frame"] > current[-1]["frame"] + 1:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def frames_to_events(
    identity: TrackIdentity,
    frames_per_sec: int,
    *,
    has_distance: bool = False,
) -> Callable[[list[FrameRow]], list[dict[str, Any]]]:
    """Build an aggregator that converts frame rows into continuous events.

    ``identity`` is dataset-specific: it returns track groups (each one row per
    frame). A gap in a track starts a new event so every event's frame array
    remains contiguous. Input rows must already follow the spatial conventions
    documented in ``audio_preprocessing/dataset/README.md#spatial-conventions``.
    """

    def aggregate(frame_rows: list[FrameRow]) -> list[dict[str, Any]]:
        runs_by_key: dict[Hashable, list[list[FrameRow]]] = {}
        for track in identity(frame_rows):
            for run in _contiguous_frame_runs(track):
                key = (run[0].get("source_idx"), run[0]["class"])
                runs_by_key.setdefault(key, []).append(run)

        events = []
        for key, runs in runs_by_key.items():
            for run_index, run in enumerate(runs):
                event_id = str(key) if len(runs) == 1 else f"{key}:{run_index}"
                frames = [
                    ContinuousData(
                        azimuth=row["azimuth"],
                        elevation=row["elevation"],
                        distance=row.get("distance") if has_distance else None,
                    ).to_dict()
                    for row in run
                ]
                event = ContinuousEvent.build_start_end(
                    start=run[0]["frame"] / frames_per_sec,
                    end=(run[-1]["frame"] + 1) / frames_per_sec,
                    label=run[0]["class"],
                    frame_array=frames,
                    frames_per_sec=frames_per_sec,
                    additional_metadata={"event_id": event_id},
                )
                events.append(event.to_dict())

        return sorted(
            events,
            key=lambda event: (
                event["offset"],
                event["label"],
                event["additional_metadata"]["event_id"],
            ),
        )

    return aggregate


def _build_spatial_record(
    path: str,
    *,
    split: str,
    source_dataset: str,
    environment: str,
    metadata_path: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    record = build_audio_record(
        path,
        "",
        split=split,
        source_dataset=source_dataset,
        metadata_path=metadata_path,
        # "foa" means AmbiX/ACN [W, Y, Z, X]/SN3D; see the dataset README.
        channel_format="foa",
        environment=environment,
    )
    # Spatial CSV rows are strong labels; class_list is reserved for separate weak labels.
    record["class_list"] = []
    record["events"] = events
    return record


def scan_foa_dev(
    dev_dir: str,
    metadata_dir: str,
    *,
    source_dataset: str,
    filename_re: Pattern[str],
    split_context_fn: Callable[[str, dict[str, int]], tuple[str, dict[str, Any]] | None],
    metadata_path_fn: Callable[[str, str, str], str | None],
    parse_metadata_csv: Callable[[str], list[FrameRow]],
    aggregate_fn: Callable[[list[FrameRow]], list[dict[str, Any]]],
    keep_splits: Container[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Walk a labeled FOA tree and group canonical clip records by split."""
    by_split: dict[str, list[dict[str, Any]]] = {}
    pattern = os.path.join(dev_dir, "**", "*.wav")
    for path in sorted(glob.glob(pattern, recursive=True)):
        match = filename_re.search(os.path.basename(path))
        if match is None:
            continue

        fold, room_id, mix = (int(group) for group in match.groups())
        clip_ids = {"fold": fold, "room_id": room_id, "mix": mix}
        context = split_context_fn(path, clip_ids)
        if context is None:
            continue
        split, extra_fields = context
        if keep_splits is not None and split not in keep_splits:
            continue

        resolved_metadata_path = metadata_path_fn(metadata_dir, path, dev_dir)
        metadata_path = ""
        events = []
        if resolved_metadata_path is not None and os.path.isfile(resolved_metadata_path):
            metadata_path = os.path.abspath(resolved_metadata_path)
            events = aggregate_fn(parse_metadata_csv(resolved_metadata_path))
        elif os.path.isdir(metadata_dir):
            warnings.warn(
                f"metadata_dev exists but no CSV found for {os.path.basename(path)}",
                stacklevel=2,
            )

        record = _build_spatial_record(
            os.path.abspath(path),
            split=split,
            source_dataset=source_dataset,
            environment=f"room{room_id}",
            metadata_path=metadata_path,
            events=events,
        )
        record.update(extra_fields)
        record.update(clip_ids)
        by_split.setdefault(split, []).append(record)
    return by_split


def scan_foa_eval(
    eval_dir: str,
    *,
    source_dataset: str,
    filename_re: Pattern[str],
    extra_null_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Walk an unlabeled FOA evaluation tree and build canonical clip records."""
    rows = []
    pattern = os.path.join(eval_dir, "**", "*.wav")
    for path in sorted(glob.glob(pattern, recursive=True)):
        match = filename_re.match(os.path.splitext(os.path.basename(path))[0])
        if match is None:
            continue

        record = _build_spatial_record(
            os.path.abspath(path),
            split="eval",
            source_dataset=source_dataset,
            environment="",
            metadata_path="",
            events=[],
        )
        record.update({field: None for field in extra_null_fields})
        record.update({"fold": None, "room_id": None, "mix": int(match.group(1))})
        rows.append(record)
    return rows
