"""Shared parsing for the TUT / DCASE acoustic-scene and sound-event datasets.

The TUT Acoustic Scenes and TUT Sound Events benchmarks (2016, 2017, ...) share one
packaging convention from the DCASE challenge toolbox: the development record provides four
official cross-validation folds, and the selected fold's train/evaluate lists become
``train``/``valid``. A separately published evaluation record becomes ``eval``. Only the
Zenodo record ids, release names, and the sound-event scene list differ between years, so each
loader configures this shared machinery with a per-task :class:`TaskConfig`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset._common import read_tsv_rows, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, Event, LabelSource

VALID_SPLITS = ("train", "valid", "eval")


@dataclass(frozen=True)
class TaskConfig:
    """Packaging constants for one TUT benchmark task.

    ``scenes`` lists the sound-event acoustic scenes whose per-scene annotations should be
    read; an empty tuple marks the acoustic-scene-classification task, which has no scene
    partitioning.
    """

    source_dataset: str
    development_record: str
    evaluation_record: str
    release_prefix: str
    scenes: tuple[str, ...] = ()

    @property
    def is_sound_events(self) -> bool:
        return bool(self.scenes)

    def release(self, subset: str) -> str:
        return f"{self.release_prefix}-{subset}"


def _releases(tasks: Mapping[str, TaskConfig]) -> frozenset[str]:
    return frozenset(
        config.release(subset)
        for config in tasks.values()
        for subset in ("development", "evaluation")
    )


def _normalize_relative_path(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./")


class LayoutIndex:
    """Index all extracted TUT releases with one recursive filesystem walk."""

    def __init__(self, root: Path, releases: frozenset[str]) -> None:
        self.root = root
        self.releases = releases
        self.audio: dict[tuple[str, str], Path] = {}
        self.metadata: dict[tuple[str, str], Path] = {}

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            release_positions = [
                position for position, part in enumerate(path.parts) if part in releases
            ]
            if not release_positions:
                continue
            release_position = release_positions[-1]
            release = path.parts[release_position]
            relative = Path(*path.parts[release_position + 1 :]).as_posix()
            target = self.audio if path.suffix.lower() == ".wav" else self.metadata
            key = (release, relative)
            if key in target:
                kind = "audio files" if target is self.audio else "metadata files"
                raise ValueError(f"Multiple TUT {kind} match {relative!r} under {root}.")
            target[key] = path

    def metadata_path(self, release: str, relative_path: str) -> Path:
        relative = _normalize_relative_path(relative_path)
        try:
            return self.metadata[(release, relative)]
        except KeyError:
            wanted = (Path(release) / relative).as_posix()
            raise FileNotFoundError(
                f"TUT metadata {wanted!r} not found under {self.root}."
            ) from None

    def audio_path(self, release: str, relative_path: str) -> Path:
        relative = _normalize_relative_path(relative_path)
        try:
            return self.audio[(release, relative)]
        except KeyError:
            raise FileNotFoundError(
                f"TUT audio file {relative!r} not found under {self.root}."
            ) from None

    def audio_relatives(self, release: str, prefix: str) -> list[str]:
        return sorted(
            relative
            for indexed_release, relative in self.audio
            if indexed_release == release and relative.startswith(prefix)
        )


def _build_record(
    audio_path: Path,
    *,
    split: str,
    source_dataset: str,
    metadata_path: Path,
    class_list: list[str],
    environment: str,
    events: list[dict],
) -> dict:
    audio = AudioDecoder(str(audio_path)).metadata
    if audio.sample_rate is None:
        raise ValueError(f"Could not determine the sample rate for {audio_path}.")
    if audio.duration_seconds is None:
        raise ValueError(f"Could not determine the duration for {audio_path}.")
    if audio.num_channels is None:
        raise ValueError(f"Could not determine the channel count for {audio_path}.")

    channel_format = (
        "binaural"
        if audio.num_channels == 2
        else "mono"
        if audio.num_channels == 1
        else f"{audio.num_channels}-channel"
    )
    return {
        "audio_path": str(audio_path),
        "sample_rate": audio.sample_rate,
        "clip_offset": 0.0,
        "clip_duration": audio.duration_seconds,
        "class_list": class_list,
        "split": split,
        "source_dataset": source_dataset,
        "metadata_path": str(metadata_path),
        "events": events,
        "num_channels": audio.num_channels,
        "channel_format": channel_format,
        "environment": environment,
    }


def _acoustic_scene_specs(
    layout: LayoutIndex,
    release: str,
    split: str,
    fold: int,
) -> list[tuple[str, str, Path]]:
    relative_metadata = (
        "meta.txt"
        if split == "eval"
        else f"evaluation_setup/fold{fold}_{'train' if split == 'train' else 'evaluate'}.txt"
    )
    metadata_path = layout.metadata_path(release, relative_metadata)
    specs = []
    for row in read_tsv_rows(metadata_path):
        if len(row) < 2:
            raise ValueError(
                f"Malformed acoustic-scene metadata row in {metadata_path}: {row}"
            )
        specs.append((_normalize_relative_path(row[0]), row[1], metadata_path))
    return specs


def _sound_event_specs_for_scene(
    layout: LayoutIndex,
    release: str,
    split: str,
    fold: int,
    scene: str,
) -> list[tuple[str, str, list[str], list[dict], Path]]:
    excluded_path: Path | None = None
    if split == "train":
        annotation_name = f"evaluation_setup/{scene}_fold{fold}_train.txt"
        excluded_path = layout.metadata_path(
            release, f"evaluation_setup/{scene}_fold{fold}_test.txt"
        )
        file_list_path = None
    elif split == "valid":
        annotation_name = f"evaluation_setup/{scene}_fold{fold}_evaluate.txt"
        file_list_path = layout.metadata_path(
            release, f"evaluation_setup/{scene}_fold{fold}_test.txt"
        )
    else:
        annotation_name = f"evaluation_setup/{scene}_evaluate.txt"
        file_list_path = layout.metadata_path(release, f"evaluation_setup/{scene}_test.txt")

    annotation_path = layout.metadata_path(release, annotation_name)
    annotations: dict[str, tuple[str, list[dict]]] = {}
    order = []
    for row in read_tsv_rows(annotation_path):
        if len(row) < 5:
            raise ValueError(
                f"Malformed sound-event metadata row in {annotation_path}: {row}"
            )
        relative = _normalize_relative_path(row[0])
        if relative not in annotations:
            annotations[relative] = (row[1], [])
            order.append(relative)
        annotations[relative][1].append(
            Event.build_start_end(float(row[2]), float(row[3]), row[4]).to_dict()
        )

    if split == "train":
        assert excluded_path is not None
        excluded = {
            _normalize_relative_path(row[0]) for row in read_tsv_rows(excluded_path)
        }
        overlap = excluded.intersection(order)
        if overlap:
            raise ValueError(
                f"TUT train annotations overlap the test list: {sorted(overlap)}"
            )
        candidates = layout.audio_relatives(release, f"audio/{scene}/")
        order.extend(
            relative
            for relative in candidates
            if relative not in excluded and relative not in annotations
        )
        environments = {
            relative: annotations.get(relative, (scene, []))[0] for relative in order
        }
    elif file_list_path is not None:
        file_rows = read_tsv_rows(file_list_path)
        order = [_normalize_relative_path(row[0]) for row in file_rows]
        environments = {
            _normalize_relative_path(row[0]): row[1] if len(row) > 1 else scene
            for row in file_rows
        }
    else:
        environments = {relative: annotations[relative][0] for relative in order}

    specs = []
    for relative in order:
        events = annotations.get(relative, (environments[relative], []))[1]
        classes = sorted({event["label"] for event in events})
        specs.append((relative, environments[relative], classes, events, annotation_path))
    return specs


def collect(
    root: Path,
    *,
    task_config: TaskConfig,
    split: str,
    fold: int,
    releases: frozenset[str],
    _layout: LayoutIndex | None = None,
) -> Iterable[dict]:
    layout = _layout or LayoutIndex(root, releases)
    subset = "evaluation" if split == "eval" else "development"
    release = task_config.release(subset)
    source_dataset = task_config.source_dataset

    if not task_config.is_sound_events:
        for relative, scene, metadata_path in _acoustic_scene_specs(
            layout, release, split, fold
        ):
            yield _build_record(
                layout.audio_path(release, relative),
                split=split,
                source_dataset=source_dataset,
                metadata_path=metadata_path,
                class_list=[scene],
                environment=scene,
                events=[],
            )
        return

    for scene in task_config.scenes:
        for (
            relative,
            environment,
            classes,
            events,
            metadata_path,
        ) in _sound_event_specs_for_scene(layout, release, split, fold, scene):
            yield _build_record(
                layout.audio_path(release, relative),
                split=split,
                source_dataset=source_dataset,
                metadata_path=metadata_path,
                class_list=classes,
                environment=environment,
                events=events,
            )


class TUTDCASELoader(ZenodoLoader):
    """Shared lifecycle for TUT benchmark tasks packaged with the DCASE toolbox layout.

    Subclasses set ``TASKS``, ``DATASET_LABEL`` (used in validation messages), and
    ``DEFAULT_ROOT``. The selected development fold maps to ``train``/``valid`` and the
    evaluation record maps to ``eval``.
    """

    label_source = LabelSource.GOLD
    TASKS: Mapping[str, TaskConfig] = {}
    DATASET_LABEL: str = "TUT"
    DEFAULT_ROOT: Path = Path("data/tut")

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        task: str = "sound_events",
        fold: int = 1,
        splits: Iterable[str] = VALID_SPLITS,
        prepare: bool = True,
    ) -> None:
        if task not in self.TASKS:
            raise ValueError(
                f"Unknown {self.DATASET_LABEL} task {task!r}; valid tasks: {list(self.TASKS)}"
            )
        if fold not in range(1, 5):
            raise ValueError(f"{self.DATASET_LABEL} fold must be between 1 and 4.")
        self.splits = tuple(splits)
        unknown_splits = [split for split in self.splits if split not in VALID_SPLITS]
        if unknown_splits:
            raise ValueError(
                f"Unknown {self.DATASET_LABEL} split(s) {unknown_splits}; "
                f"valid splits: {list(VALID_SPLITS)}"
            )
        if not self.splits:
            raise ValueError(f"{type(self).__name__} requires at least one split.")
        self.task = task
        self.fold = fold

        record_ids = []
        task_config = self.TASKS[self.task]
        if any(split in self.splits for split in ("train", "valid")):
            record_ids.append(task_config.development_record)
        if "eval" in self.splits:
            record_ids.append(task_config.evaluation_record)

        loader_root = root if root is not None or not prepare else self.DEFAULT_ROOT / self.task
        super().__init__(
            root=loader_root,
            record_ids=record_ids,
            only=["audio", "meta"],
            prepare=prepare,
        )

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                f"{type(self).__name__} needs a root path. Use prepare=False with an existing "
                "local extraction, or prepare=True to download the requested task and splits."
            )
        releases = _releases(self.TASKS)
        layout = LayoutIndex(self.root, releases)
        task_config = self.TASKS[self.task]
        records = {
            split: collect(
                self.root,
                task_config=task_config,
                split=split,
                fold=self.fold,
                releases=releases,
                _layout=layout,
            )
            for split in self.splits
        }
        dataset = splits_to_audio_dataset(
            records,
            features=DATA_FEATURES,
            label_source=self.label_source,
        )
        dataset.split = "train" if "train" in dataset.data else self.splits[0]
        return dataset
