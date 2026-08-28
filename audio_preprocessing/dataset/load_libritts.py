"""Load the official LibriTTS corpus (LibriSpeech re-segmented and re-recorded at 24 kHz
for TTS) into an AudioDataset. Licensed CC BY 4.0.

Each utterance ships as a triplet under ``<speaker_id>/<chapter_id>/``: the WAV audio plus
sibling ``.original.txt`` and ``.normalized.txt`` transcript files, unlike LibriSpeech's single
per-chapter ``.trans.txt``. Both transcripts are kept as separate columns since TTS training
typically uses the normalized text while the original text preserves the literal transcription.

``train-clean-100``/``train-clean-360``/``train-other-500`` all map to ``train``,
``dev-clean``/``dev-other`` to ``valid``, and ``test-clean``/``test-other`` to ``eval`` (see
``SUBSETS``). ``subsets`` accepts any combination, including a partial selection with no train
or no eval subset (e.g. ``main()`` below, for a cheap smoke test) and a mixed clean/other
combination (e.g. training on ``train-clean-360`` and evaluating on ``test-other`` to measure
generalization to harder conditions) -- both are standard, intentional uses of this corpus and
mirror ``LibriSpeechLoader``.
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Iterable
from pathlib import Path

import requests
from datasets.features.features import Features, Value
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from audio_preprocessing.dataset._common import (
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

SOURCE_DATASET = "LibriTTS"
BASE_URL = "https://www.openslr.org/resources/60"
DEFAULT_ROOT = Path("data/libritts")
DEFAULT_SUBSETS = ("train-clean-100", "dev-clean", "test-clean")

SUBSETS = {
    "train-clean-100": ("train", "4a8c202b78fe1bc0c47916a98f3a2ea8"),
    "train-clean-360": ("train", "a84ef10ddade5fd25df69596a2767b2d"),
    "train-other-500": ("train", "7b181dd5ace343a5f38427999684aa6f"),
    "dev-clean": ("valid", "0c3076c1e5245bb3f0af7d82087ee207"),
    "dev-other": ("valid", "815555d8d75995782ac3ccd7f047213d"),
    "test-clean": ("eval", "7bed3bdb047c4c197f1ad3bc412db59f"),
    "test-other": ("eval", "ae3258249472a13b5abef2a816f733e4"),
}

LIBRITTS_FEATURES = Features({
    **DATA_FEATURES,
    "utterance_id": Value("string"),
    "original_text": Value("string"),
    "normalized_text": Value("string"),
    "speaker_id": Value("int64"),
    "chapter_id": Value("int64"),
    "libritts_subset": Value("string"),
})


def _checksum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(url: str, destination: Path, checksum: str) -> None:
    if destination.is_file():
        if _checksum(destination) == checksum:
            return
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if offset and response.status_code == 416:
                partial.unlink()
                continue
            response.raise_for_status()
            resumed = bool(offset and response.status_code == 206)
            start = offset if resumed else 0
            length = int(response.headers.get("content-length", 0))
            total = start + length if length else None
            with (
                partial.open("ab" if resumed else "wb") as handle,
                tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=destination.name) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
            break

    partial.replace(destination)
    if _checksum(destination) != checksum:
        destination.unlink()
        raise ValueError(f"Checksum mismatch for {destination.name}.")


def _prepare_subset(root: Path, subset: str) -> None:
    _, checksum = SUBSETS[subset]
    marker = root / ".prepared" / subset
    extracted = root / "LibriTTS" / subset
    if extracted.is_dir() and marker.is_file() and marker.read_text().strip() == checksum:
        return

    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{subset}.tar.gz"
    _download_archive(f"{BASE_URL}/{archive.name}", archive, checksum)

    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root, filter="data")
    if not extracted.is_dir():
        raise RuntimeError(f"{archive.name} did not contain LibriTTS/{subset}.")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(checksum, encoding="utf-8")
    archive.unlink()


def _records(root: Path, subset: str):
    split, _ = SUBSETS[subset]
    subset_root = root / "LibriTTS" / subset
    audio_paths = sorted(subset_root.glob("*/*/*.wav"))
    if not audio_paths:
        raise FileNotFoundError(
            f"No audio found for {subset!r} under {root}. Use prepare=True to download it."
        )

    for audio_path in tqdm(audio_paths, desc=f"Scanning {subset}", unit="file"):
        utterance_id = audio_path.stem
        # LibriTTS filenames are "<speaker_id>_<chapter_id>_<segment_id>_<utterance_id>".
        parts = utterance_id.split("_")
        if len(parts) != 4:
            raise ValueError(f"Malformed LibriTTS utterance id {utterance_id!r} in {audio_path}.")
        speaker_id, chapter_id = int(parts[0]), int(parts[1])

        original_path = audio_path.with_name(f"{utterance_id}.original.txt")
        normalized_path = audio_path.with_name(f"{utterance_id}.normalized.txt")
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original transcript for {utterance_id}: {original_path}")
        if not normalized_path.is_file():
            raise FileNotFoundError(f"Missing normalized transcript for {utterance_id}: {normalized_path}")
        original_text = original_path.read_text(encoding="utf-8").strip()
        normalized_text = normalized_path.read_text(encoding="utf-8").strip()
        if not original_text or not normalized_text:
            raise ValueError(f"Empty LibriTTS transcript for {utterance_id}.")

        audio = AudioDecoder(str(audio_path)).metadata
        if audio.sample_rate != 24000 or audio.num_channels != 1:
            raise ValueError(f"Expected 24 kHz mono audio for {utterance_id}.")
        if audio.duration_seconds is None:
            raise ValueError(f"Could not determine the duration for {audio_path}.")

        yield {
            "audio_path": str(audio_path),
            "sample_rate": audio.sample_rate,
            "clip_offset": 0.0,
            "clip_duration": audio.duration_seconds,
            "class_list": [],
            "split": split,
            "source_dataset": SOURCE_DATASET,
            "metadata_path": str(original_path),
            "events": [],
            "num_channels": audio.num_channels,
            "channel_format": "mono",
            "environment": "",
            "utterance_id": utterance_id,
            "original_text": original_text,
            "normalized_text": normalized_text,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "libritts_subset": subset,
        }


class LibriTTSLoader(BaseLoader):
    """Download and index selected official LibriTTS subsets."""

    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        subsets: Iterable[str] = DEFAULT_SUBSETS,
        prepare: bool = True,
    ) -> None:
        super().__init__(root=root, prepare=prepare)
        self.subsets = validate_choices(
            subsets,
            name="subset",
            context="LibriTTS",
            allowed=SUBSETS,
        )

    def prepare_raw(self) -> None:
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        for subset in self.subsets:
            _prepare_subset(root, subset)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "LibriTTSLoader needs a root path. Use prepare=False with an existing "
                "extraction or prepare=True to download it."
            )

        records = {
            split: []
            for split in ("train", "valid", "eval")
            if any(SUBSETS[subset][0] == split for subset in self.subsets)
        }
        for subset in self.subsets:
            split = SUBSETS[subset][0]
            records[split].extend(_records(self.root, subset))

        dataset = splits_to_audio_dataset(
            records,
            features=LIBRITTS_FEATURES,
            label_source=self.label_source,
        )
        # Point the active split at the first split actually built so a train-less
        # selection (e.g. subsets=("dev-clean",)) does not leave it on an absent "train".
        dataset.split = next(
            (split for split in ("train", "valid", "eval") if split in dataset.data),
            dataset.split,
        )
        return dataset


def main() -> AudioDataset:
    # dev-other is the smallest official archive (~0.9 GB) that still has a full split.
    return LibriTTSLoader(subsets=("dev-other",))()


if __name__ == "__main__":
    libritts = main()
    print(libritts.info())
