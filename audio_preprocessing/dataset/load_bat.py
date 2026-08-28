"""Load the BAT / SpatialSoundQA spatial-audio question-answering dataset.

SpatialSoundQA (from "BAT: Learning to Reason about Spatial Sounds with Large
Language Models", ICML 2024) has no pre-rendered audio. Each QA item names an
anechoic AudioSet clip and a binaural (or mono) room impulse response; the
spatial clip the question is about is synthesized by convolving the source with
the RIR. "Mixup" items name two sources and average the two convolutions.

This loader renders each QA item's spatial audio once (cached under
``root/rendered``) and emits one canonical record per item: ``audio_path`` points
at the rendered clip, the direction/distance of each source is carried as a
``ContinuousEvent`` (per the spatial-dataset convention in ``_spatial_common``),
and the question/answer are kept as extra columns.

The direction-of-arrival convention was derived from the gold DOA answers: in the
SoundSpaces world frame (y up) the source-minus-receiver vector reads +x right,
+y up, -z front, so azimuth = atan2(x, -z) (0 deg front, +90 deg right) and
elevation = asin(y / distance). Distance is the exact geometric range (the gold
answers quantise it into bins).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from datasets.features.features import Features, Value  # pyright: ignore[reportMissingImports]
from huggingface_hub import hf_hub_download
from scipy import signal
from scipy.io import wavfile
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import (
    DATA_FEATURES,
    AudioDataset,
    ContinuousData,
    ContinuousEvent,
    LabelSource,
)

HF_REPO = "zhisheng01/SpatialAudio"
SOURCE_DATASET = "SpatialSoundQA"
DEFAULT_ROOT = Path("data/bat")

# Repository layout is rooted at this prefix inside the HuggingFace dataset repo.
_BASE = "SpatialSoundQA"
_CLASS_LABELS = f"{_BASE}/AudioSet/metadata/class_labels_indices.csv"

TARGET_SR = 32000
CLIP_SECONDS = 10
CLIP_SAMPLES = TARGET_SR * CLIP_SECONDS
CHANNEL_TYPES = ("binaural", "mono")

# Each stage's evaluation QA is split across task-specific files; training QA for
# a stage lives in a single ``train.json`` spanning every task.
STAGE_TASKS = {
    "stage1-clsdoa": {
        "classification": "eval-stage1-classification.json",
        "doa": "eval-stage1-doa.json",
    },
    "stage2-single": {
        "classification": "eval-stage2-classification.json",
        "doa": "eval-stage2-doa.json",
    },
    "stage3-mixup": {
        "direction": "eval-stage3-direction.json",
        "distance": "eval-stage3-distance.json",
        "nonbinary": "eval-stage3-nonbinary.json",
    },
}
STAGES = tuple(STAGE_TASKS)

# Per-split source audio archive, weak-label metadata, and reverb geometry table.
SPLIT_CONFIG = {
    "train": {
        "audio_zip": f"{_BASE}/AudioSet/balanced_train.zip",
        "audioset_meta": f"{_BASE}/AudioSet/metadata/balanced.json",
        "reverb_json": "mp3d_reverb/train_reverberation.json",
    },
    "eval": {
        "audio_zip": f"{_BASE}/AudioSet/eval.zip",
        "audioset_meta": f"{_BASE}/AudioSet/metadata/eval.json",
        "reverb_json": "mp3d_reverb/eval_reverberation.json",
    },
}
_REVERB_ZIP = f"{_BASE}/mp3d_reverb.zip"

BAT_FEATURES = Features({
    **DATA_FEATURES,
    "sample_id": Value("string"),
    "question": Value("string"),
    "answer": Value("string"),
    "question_type": Value("string"),
    "question_id": Value("int64"),
    "stage": Value("string"),
})


def _doa_from_positions(sensor: Sequence[float], source: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(azimuth_deg, elevation_deg, distance_m)`` of a source from the receiver.

    Azimuth is positive to the left, matching the other spatial loaders (STARSS23, BBC SAQAS)
    and the ``Spatialize`` transforms.
    """
    dx = source[0] - sensor[0]  # +right
    dy = source[1] - sensor[1]  # +up
    dz = source[2] - sensor[2]  # +back (front is -z)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    azimuth = math.degrees(math.atan2(-dx, -dz))  # negate x so positive azimuth is to the left
    elevation = math.degrees(math.asin(dy / distance)) if distance > 0 else 0.0
    return azimuth, elevation, distance


def _parse_xyz(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def _load_reverb_geometry(reverb_zip: zipfile.ZipFile, member: str) -> dict[str, tuple[list[float], list[float], str]]:
    """Map ``reverb_id`` to ``(sensor_xyz, source_xyz, room)`` from a reverberation table."""
    data = json.load(reverb_zip.open(member))["data"]
    geometry = {}
    for entry in data:
        fname = entry["fname"]
        geometry[fname] = (
            _parse_xyz(entry["sensor_position"]),
            _parse_xyz(entry["source_position"]),
            fname.split("/")[0],
        )
    return geometry


def _load_audioset_labels(meta_path: Path, class_labels_path: Path) -> dict[str, str]:
    """Map an AudioSet clip id to its semicolon-joined human-readable class names."""
    mid_to_name = {}
    with class_labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):  # columns: index, mid, display_name
            mid_to_name[row["mid"]] = row["display_name"]

    labels = {}
    for entry in json.loads(meta_path.read_text(encoding="utf-8")):
        names = [mid_to_name.get(mid, mid) for mid in entry["label"]]
        labels[entry["id"]] = "; ".join(names)
    return labels


def _read_source(audio_zip: zipfile.ZipFile, audio_id: str) -> np.ndarray:
    """Decode an anechoic AudioSet clip to a normalised 32 kHz mono waveform."""
    samples = AudioDecoder(audio_zip.read(f"{audio_id}.wav")).get_all_samples()
    waveform = samples.data.numpy()[0]
    if samples.sample_rate != TARGET_SR:
        waveform = signal.resample_poly(waveform, TARGET_SR, samples.sample_rate)

    rms = np.sqrt(np.mean(waveform**2))
    if rms > 0:  # normalise to -14 dBFS, matching the reference renderer
        waveform = waveform * 10 ** ((-14.0 - 20 * np.log10(rms)) / 20)
    return waveform


def _fix_length(waveform: np.ndarray, length: int) -> np.ndarray:
    current_length = int(waveform.shape[1])
    if current_length >= length:
        return waveform[:, :length]
    pad_width = np.asarray(((0, 0), (0, length - current_length)), dtype=np.intp)
    return np.pad(waveform, pad_width)


def _spatialise(
    audio_zip: zipfile.ZipFile,
    reverb_zip: zipfile.ZipFile,
    channel_type: str,
    audio_id: str,
    reverb_id: str,
) -> np.ndarray:
    waveform = _read_source(audio_zip, audio_id)
    rir = np.load(io.BytesIO(reverb_zip.read(f"mp3d_reverb/{channel_type}/{reverb_id}")))
    convolved = signal.fftconvolve(waveform[None, :], rir, mode="full")
    return _fix_length(convolved, CLIP_SAMPLES)


def _render_clip(
    audio_zip: zipfile.ZipFile,
    reverb_zip: zipfile.ZipFile,
    channel_type: str,
    item: Mapping[str, object],
    out_path: Path,
) -> None:
    """Render one QA item's spatial audio to ``out_path`` unless it is already cached."""
    if out_path.is_file():
        return
    waveform = _spatialise(
        audio_zip, reverb_zip, channel_type, str(item["audio_id"]), str(item["reverb_id"])
    )
    audio_id2, reverb_id2 = item["audio_id2"], item["reverb_id2"]
    if audio_id2 is not None and reverb_id2 is not None:
        other = _spatialise(audio_zip, reverb_zip, channel_type, str(audio_id2), str(reverb_id2))
        waveform = (waveform + other) / 2

    tmp_path = out_path.with_suffix(".part.wav")
    wavfile.write(tmp_path, TARGET_SR, waveform.T.astype(np.float32))
    tmp_path.replace(out_path)


def _clip_name(item: Mapping[str, object], channel_type: str) -> str:
    key = "|".join([
        str(item["audio_id"]),
        str(item["reverb_id"]),
        str(item["audio_id2"] or ""),
        str(item["reverb_id2"] or ""),
        channel_type,
    ])
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest() + ".wav"


def _source_event(
    audio_id: str,
    reverb_id: str,
    geometry: Mapping[str, tuple[list[float], list[float], str]],
    label: str,
) -> dict[str, object]:
    sensor, source, room = geometry[reverb_id]
    azimuth, elevation, distance = _doa_from_positions(sensor, source)
    # The source is a static point for the whole clip; sample its DOA at 1 Hz.
    frames = [
        ContinuousData(azimuth=azimuth, elevation=elevation, distance=distance).to_dict()
        for _ in range(CLIP_SECONDS)
    ]
    event = ContinuousEvent.build_start_end(
        start=0.0,
        end=float(CLIP_SECONDS),
        label=label,
        frame_array=frames,
        frames_per_sec=1,
        additional_metadata={"audio_id": audio_id, "reverb_id": reverb_id, "room": room},
    )
    return event.to_dict()


class BATLoader(BaseLoader):
    """Render and index the SpatialSoundQA (BAT) spatial-audio QA dataset.

    The dataset is enormous (millions of QA items across the training stages), so
    ``max_items`` caps how many are rendered; leave it ``None`` to build the whole
    stage/split/task.
    """

    label_source = LabelSource.SYNTHETIC

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        stage: str = "stage1-clsdoa",
        split: str = "eval",
        task: str | None = None,
        channel_type: str = "binaural",
        max_items: int | None = None,
        seed: int = 42,
        prepare: bool = True,
    ) -> None:
        super().__init__(root=root, prepare=prepare)
        if stage not in STAGE_TASKS:
            raise ValueError(f"Unknown stage {stage!r}. Choose from: {', '.join(STAGES)}")
        if split not in SPLIT_CONFIG:
            raise ValueError(f"Unknown split {split!r}. Choose from: {', '.join(SPLIT_CONFIG)}")
        if channel_type not in CHANNEL_TYPES:
            raise ValueError(
                f"Unknown channel_type {channel_type!r}. Choose from: {', '.join(CHANNEL_TYPES)}"
            )

        if split == "eval":
            tasks = STAGE_TASKS[stage]
            if task is None:
                task = next(iter(tasks))
            if task not in tasks:
                raise ValueError(
                    f"Unknown task {task!r} for {stage}. Choose from: {', '.join(tasks)}"
                )
            self.qa_file = tasks[task]
        else:
            if task is not None:
                raise ValueError("The train split has a single QA file; leave task unset.")
            self.qa_file = "train.json"

        self.stage = stage
        self.split = split
        self.task = task
        self.channel_type = channel_type
        self.max_items = max_items
        self.seed = seed

    def _qa_repo_path(self) -> str:
        return f"{_BASE}/closed-end/{self.stage}/{self.qa_file}"

    def prepare_raw(self) -> None:
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

        config = SPLIT_CONFIG[self.split]
        wanted = [
            _CLASS_LABELS,
            config["audioset_meta"],
            config["audio_zip"],
            _REVERB_ZIP,
            self._qa_repo_path(),
        ]
        for repo_path in wanted:
            hf_hub_download(HF_REPO, repo_path, repo_type="dataset", local_dir=str(root))

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError("BATLoader needs a root path; use prepare=True to download the dataset.")
        root = self.root
        config = SPLIT_CONFIG[self.split]

        qa_path = root / self._qa_repo_path()
        items = json.loads(qa_path.read_text(encoding="utf-8"))["data"]
        items = self._select_items(items)

        labels = _load_audioset_labels(root / config["audioset_meta"], root / _CLASS_LABELS)

        rendered_dir = root / "rendered" / self.channel_type
        rendered_dir.mkdir(parents=True, exist_ok=True)
        canonical_split = self.split

        records = []
        with (
            zipfile.ZipFile(root / config["audio_zip"]) as audio_zip,
            zipfile.ZipFile(root / _REVERB_ZIP) as reverb_zip,
        ):
            geometry = _load_reverb_geometry(reverb_zip, config["reverb_json"])
            for item in items:
                clip_path = rendered_dir / _clip_name(item, self.channel_type)
                _render_clip(audio_zip, reverb_zip, self.channel_type, item, clip_path)
                records.append(
                    self._build_record(clip_path, qa_path, item, geometry, labels, canonical_split)
                )

        dataset = splits_to_audio_dataset(
            {canonical_split: records},
            features=BAT_FEATURES,
            label_source=self.label_source,
        )
        # A build holds a single split, so point the active split at it rather than the
        # default "train" (which is absent for an eval build and would KeyError on access).
        dataset.split = canonical_split
        return dataset

    def _select_items(self, items: list[dict]) -> list[dict]:
        if self.max_items is None or self.max_items >= len(items):
            return items
        rng = random.Random(self.seed)
        chosen = sorted(rng.sample(range(len(items)), self.max_items))
        return [items[index] for index in chosen]

    def _build_record(self, clip_path, qa_path, item, geometry, labels, split):
        sources = [(item["audio_id"], item["reverb_id"])]
        if item["audio_id2"] is not None and item["reverb_id2"] is not None:
            sources.append((item["audio_id2"], item["reverb_id2"]))

        # One pass over the item's sources builds both the DOA events and the
        # clip-level weak labels, looking each source's class names up once.
        events = []
        class_names = set()
        for audio_id, reverb_id in sources:
            label = labels.get(audio_id.split("/")[-1], "")
            events.append(_source_event(audio_id, reverb_id, geometry, label))
            class_names.update(name for name in label.split("; ") if name)

        room = geometry[item["reverb_id"]][2]
        record = build_audio_record(
            str(clip_path),
            "",
            split=split,
            source_dataset=SOURCE_DATASET,
            metadata_path=str(qa_path),
            channel_format=self.channel_type,
            environment=room,
        )
        record["class_list"] = sorted(class_names)
        record["events"] = events
        record.update({
            # Distinct rows can share one rendered clip (same audio + reverb, different question),
            # so give each a stable id instead of letting MDS fall back to hashing the audio.
            "sample_id": f"{self.stage}:{self.split}:{item['question_type']}:{item['question_id']}",
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "question_id": item["question_id"],
            "stage": self.stage,
        })
        return record
