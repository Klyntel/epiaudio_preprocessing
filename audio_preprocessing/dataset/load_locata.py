"""Load LOCATA array recordings and 120 Hz source trajectories."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]

from audio_preprocessing.dataset._common import (
    archive_stems_for_splits,
    build_audio_record,
    extracted_archive_dir,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import (
    DATA_FEATURES,
    AudioDataset,
    ContinuousData,
    ContinuousEvent,
    LabelSource,
)

ZENODO_RECORD = "3630471"
SOURCE_DATASET = "LOCATA"
SPLIT_ARCHIVES = {"train": "dev.zip", "eval": "eval.zip"}
VALID_ARRAYS = ("benchmark2", "eigenmike", "dicit", "dummy")
VALID_TASKS = tuple(range(1, 7))
EXPECTED_CHANNELS = {"benchmark2": 12, "eigenmike": 32, "dicit": 15, "dummy": 4}
FRAME_RATE = 120
AUDIO_SAMPLE_RATE = 48_000
NS_PER_SEC = 1_000_000_000
DEFAULT_ROOT = Path("data/locata_raw")

TIME_FIELDS = ("year", "month", "day", "hour", "minute", "second")
ARRAY_FIELDS = (
    "x",
    "y",
    "z",
    "rotation_11",
    "rotation_12",
    "rotation_13",
    "rotation_21",
    "rotation_22",
    "rotation_23",
    "rotation_31",
    "rotation_32",
    "rotation_33",
)
SOURCE_FIELDS = ("x", "y", "z")

LOCATA_FEATURES = Features({
    **DATA_FEATURES,
    "task": Value("int64"),
    "recording": Value("int64"),
    "array": Value("string"),
})

Row = dict[str, str]


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: int
    valid: bool


def _table(path: Path) -> Iterator[tuple[int, Row]]:
    """Yield rows from LOCATA's comma-, tab-, or space-delimited text files."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing LOCATA metadata file: {path}")
    with path.open(encoding="utf-8") as handle:
        lines = (
            (line_number, line.strip())
            for line_number, line in enumerate(handle, start=1)
            if line.strip()
        )
        try:
            _, header = next(lines)
        except StopIteration as error:
            raise ValueError(f"Empty LOCATA metadata file: {path}") from error
        fields = re.split(r"[,\t ]+", header)
        for line_number, line in lines:
            values = re.split(r"[,\t ]+", line)
            if len(values) != len(fields):
                raise ValueError(f"Malformed LOCATA row {line_number} in {path}")
            yield line_number, dict(zip(fields, values, strict=True))


def _rows(path: Path, required: Sequence[str]) -> list[Row]:
    rows = [row for _, row in _table(path)]
    if not rows:
        raise ValueError(f"LOCATA metadata file has no rows: {path}")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise ValueError(f"LOCATA metadata {path} is missing columns {missing}")
    return rows


def _integer(value: str, name: str) -> int:
    number = Decimal(value)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"LOCATA {name} must be an integer, got {value!r}")
    return int(number)


def _timestamp(row: Row) -> int:
    year, month, day_of_month, hour, minute = (
        _integer(row[field], f"timestamp {field}") for field in TIME_FIELDS[:-1]
    )
    whole_seconds = (
        date(year, month, day_of_month).toordinal() * 86_400 + hour * 3_600 + minute * 60
    )
    return whole_seconds * NS_PER_SEC + int(Decimal(row["second"]) * NS_PER_SEC)


def _binary_flag(value: str, name: str, path: Path) -> int:
    flag = Decimal(value)
    if not flag.is_finite() or flag not in (0, 1):
        raise ValueError(f"LOCATA {name} must be 0 or 1 in {path}")
    return int(flag)


def _finite_float(value: str, field: str, path: Path) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"LOCATA spatial field {field} must be finite in {path}, got {value!r}"
        )
    return number


def _required_frames(path: Path) -> list[Frame]:
    rows = _rows(path, (*TIME_FIELDS, "valid_flag"))
    timestamps = [_timestamp(row) for row in rows]
    frames = []
    # LOCATA captures at 120 fps, but its released system-clock timestamps contain jitter
    # and occasional repeats; row order defines the fixed-rate tracking frame index.
    for index, (row, timestamp) in enumerate(zip(rows, timestamps, strict=True)):
        valid = _binary_flag(row["valid_flag"], "valid_flag", path)
        frames.append(Frame(index, timestamp, bool(valid)))
    return frames


def _positions(path: Path, frames: Sequence[Frame], fields: Sequence[str]) -> list[Row]:
    rows = _rows(path, (*TIME_FIELDS, *fields))
    if len(rows) != len(frames):
        raise ValueError(f"LOCATA position row count mismatch in {path}")
    if any(
        abs(_timestamp(row) - frame.timestamp) > 1_000
        for row, frame in zip(rows, frames, strict=True)
    ):
        raise ValueError(f"LOCATA position timestamp mismatch in {path}")
    for row in rows:
        for field in fields:
            _finite_float(row[field], field, path)
    return rows


def _audio_alignment(
    path: Path,
    frames: Sequence[Frame],
    sample_count: int,
) -> tuple[list[int | None], int]:
    """Map tracking times to the preceding audio sample, matching LOCATA evaluation."""
    sample_indices: list[int | None] = [None] * len(frames)
    target = 0
    first_timestamp: int | None = None
    previous_timestamp: int | None = None
    count = 0
    for _, row in _table(path):
        timestamp = _timestamp(row)
        if first_timestamp is None:
            first_timestamp = timestamp
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"LOCATA audio timestamps are not increasing in {path}")
        while target < len(frames) and frames[target].timestamp <= timestamp:
            if frames[target].timestamp >= first_timestamp:
                sample_indices[target] = max(0, count - 1)
            target += 1
        previous_timestamp = timestamp
        count += 1
    if first_timestamp is None or count != sample_count:
        raise ValueError(f"LOCATA audio-timestamp length mismatch in {path}")
    return sample_indices, first_timestamp


def _vad_frames(
    path: Path,
    sample_indices: Sequence[int | None],
    sample_count: int,
) -> list[int]:
    targets: dict[int, list[int]] = {}
    for frame_index, sample_index in enumerate(sample_indices):
        if sample_index is not None:
            targets.setdefault(sample_index, []).append(frame_index)

    sampled = [0] * len(sample_indices)
    count = 0
    for _, row in _table(path):
        if len(row) != 1:
            raise ValueError(f"LOCATA VAD must contain one column: {path}")
        value = _binary_flag(next(iter(row.values())), "VAD", path)
        for frame_index in targets.get(count, ()):
            sampled[frame_index] = value
        count += 1
    if count != sample_count:
        raise ValueError(f"LOCATA VAD length mismatch in {path}")
    return sampled


def _spatial_frame(array_row: Row, source_row: Row) -> dict[str, Any]:
    """Convert LOCATA global Cartesian data to canonical receiver-centered angles."""
    array_xyz = [float(array_row[field]) for field in SOURCE_FIELDS]
    source_xyz = [float(source_row[field]) for field in SOURCE_FIELDS]
    dx, dy, dz = (
        source - array for source, array in zip(source_xyz, array_xyz, strict=True)
    )
    rotation = [float(array_row[field]) for field in ARRAY_FIELDS[3:]]
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rotation
    # LOCATA publishes R; its official receiver-relative conversion is R.T @ (source-array).
    x = r11 * dx + r21 * dy + r31 * dz
    y = r12 * dx + r22 * dy + r32 * dz
    z = r13 * dx + r23 * dy + r33 * dz
    distance = math.sqrt(x * x + y * y + z * z)
    if distance == 0:
        raise ValueError("LOCATA source and array positions coincide")
    return ContinuousData(
        azimuth=(math.degrees(math.atan2(-x, y)) + 180) % 360 - 180,
        elevation=math.degrees(math.atan2(z, math.hypot(x, y))),
        distance=distance,
    ).to_dict()


def _source_events(
    source_id: str,
    frames: Sequence[Frame],
    array_rows: Sequence[Row],
    source_rows: Sequence[Row],
    sample_indices: Sequence[int | None],
    audio_start: int,
    vad: Sequence[int],
    clip_duration: float,
) -> list[dict[str, Any]]:
    active = []
    for frame, array_row, source_row, sample_index, is_active in zip(
        frames, array_rows, source_rows, sample_indices, vad, strict=True
    ):
        offset = (frame.timestamp - audio_start) / NS_PER_SEC
        if frame.valid and sample_index is not None and is_active and offset < clip_duration:
            active.append((frame.index, offset, _spatial_frame(array_row, source_row)))

    runs: list[list[tuple[int, float, dict[str, Any]]]] = []
    for item in active:
        if not runs or item[0] != runs[-1][-1][0] + 1:
            runs.append([])
        runs[-1].append(item)

    events = []
    for run in runs:
        start = run[0][1]
        end = min(clip_duration, start + len(run) / FRAME_RATE)
        frame_count = math.ceil((end - start) * FRAME_RATE - 1e-9)
        event = ContinuousEvent.build_start_end(
            start=start,
            end=end,
            label="speech",
            frame_array=[item[2] for item in run[:frame_count]],
            frames_per_sec=FRAME_RATE,
            additional_metadata={"source_id": source_id},
        )
        events.append(event.to_dict())
    return events


def _source_pairs(array_dir: Path, array: str) -> list[tuple[str, Path, Path]]:
    pairs = []
    source_ids = set()
    for position_path in sorted(array_dir.glob("position_source_*.txt")):
        source_id = position_path.stem.removeprefix("position_source_")
        source_ids.add(source_id)
        vad_path = array_dir / f"VAD_{array}_{source_id}.txt"
        if not vad_path.is_file():
            raise FileNotFoundError(f"Missing LOCATA VAD file: {vad_path}")
        pairs.append((source_id, position_path, vad_path))
    for vad_path in sorted(array_dir.glob(f"VAD_{array}_*.txt")):
        source_id = vad_path.stem.removeprefix(f"VAD_{array}_")
        if source_id not in source_ids:
            position_path = array_dir / f"position_source_{source_id}.txt"
            raise FileNotFoundError(f"Missing LOCATA source position file: {position_path}")
    if not pairs:
        raise FileNotFoundError(f"No LOCATA source positions found under {array_dir}")
    return pairs


def _record(
    audio_path: Path,
    split: str,
    task: int,
    recording: int,
    array: str,
) -> dict[str, Any]:
    array_dir = audio_path.parent
    array_path = array_dir / f"position_array_{array}.txt"
    frames = _required_frames(array_dir / "required_time.txt")
    array_rows = _positions(array_path, frames, ARRAY_FIELDS)

    record = build_audio_record(
        str(audio_path.resolve()),
        None,
        split=split,
        source_dataset=SOURCE_DATASET,
        metadata_path=str(array_path.resolve()),
        channel_format="microphone_array",
        environment="reverberant_computing_lab",
    )
    if record["sample_rate"] != AUDIO_SAMPLE_RATE:
        raise ValueError(f"LOCATA audio must be {AUDIO_SAMPLE_RATE} Hz: {audio_path}")
    if record["num_channels"] != EXPECTED_CHANNELS[array]:
        raise ValueError(f"Unexpected LOCATA channel count for {audio_path}")
    clip_duration = record["clip_duration"]
    if clip_duration is None or clip_duration <= 0:
        raise ValueError(f"LOCATA audio has no duration: {audio_path}")

    sample_count = round(clip_duration * AUDIO_SAMPLE_RATE)
    sample_indices, audio_start = _audio_alignment(
        array_dir / f"audio_array_timestamps_{array}.txt",
        frames,
        sample_count,
    )
    events = []
    for source_id, source_path, vad_path in _source_pairs(array_dir, array):
        events.extend(
            _source_events(
                source_id,
                frames,
                array_rows,
                _positions(source_path, frames, SOURCE_FIELDS),
                sample_indices,
                audio_start,
                _vad_frames(vad_path, sample_indices, sample_count),
                clip_duration,
            )
        )

    record["events"] = sorted(
        events,
        key=lambda event: (event["offset"], event["additional_metadata"]["source_id"]),
    )
    record.update({"task": task, "recording": recording, "array": array})
    return record


def _scan(
    split_dir: Path,
    split: str,
    arrays: Sequence[str],
    tasks: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing LOCATA split directory: {split_dir}")
    selected_arrays = set(arrays)
    selected_tasks = set(VALID_TASKS if tasks is None else tasks)
    records = []
    for audio_path in sorted(split_dir.rglob("audio_array_*.wav")):
        task_name, recording_name, array, _ = audio_path.parts[-4:]
        if array not in selected_arrays or audio_path.name != f"audio_array_{array}.wav":
            continue
        if not task_name.startswith("task") or not recording_name.startswith("recording"):
            continue
        task = int(task_name.removeprefix("task"))
        recording = int(recording_name.removeprefix("recording"))
        if task in selected_tasks:
            records.append(_record(audio_path, split, task, recording, array))
    return records


def _selection(name: str, values: Sequence[str], allowed: Sequence[str]) -> tuple[str, ...]:
    items = (values,) if isinstance(values, str) else tuple(values)
    unknown = [value for value in items if value not in allowed]
    if not items or unknown:
        raise ValueError(f"Invalid LOCATA {name} {unknown or list(items)}; valid: {list(allowed)}")
    return items


class LOCATALoader(ZenodoLoader):
    """Build LOCATA array recordings with gold spatial speech trajectories."""

    record_id = ZENODO_RECORD
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        splits: Sequence[str] = ("train", "eval"),
        arrays: Sequence[str] = VALID_ARRAYS,
        tasks: Iterable[int] | None = None,
        prepare: bool = True,
    ) -> None:
        self.splits = _selection("splits", splits, tuple(SPLIT_ARCHIVES))
        self.arrays = _selection("arrays", arrays, VALID_ARRAYS)
        self.tasks = None if tasks is None else tuple(tasks)
        if self.tasks is not None and (
            not self.tasks or any(task not in VALID_TASKS for task in self.tasks)
        ):
            raise ValueError(f"LOCATA tasks must be in {list(VALID_TASKS)}")
        super().__init__(
            root=root,
            only=archive_stems_for_splits(SPLIT_ARCHIVES, self.splits),
            prepare=prepare,
        )

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError("LOCATALoader requires a root path")
        data = {}
        for split in self.splits:
            split_dir = Path(extracted_archive_dir(str(self.root), SPLIT_ARCHIVES, split))
            rows = _scan(split_dir, split, self.arrays, self.tasks)
            if not rows:
                raise RuntimeError(f"No matching LOCATA clips found under {split_dir}")
            data[split] = rows
        dataset = splits_to_audio_dataset(
            data,
            features=LOCATA_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = self.splits[0]
        return dataset


def main() -> AudioDataset:
    # dev.zip is LOCATA's smallest independently downloadable archive (~6.2 GB).
    return LOCATALoader(splits=("train",), arrays=("dummy",), tasks=(1,))()


if __name__ == "__main__":
    dataset = main()
    print(dataset.info())
