"""Load the original FLAC release of Multilingual LibriSpeech (MLS)."""

from __future__ import annotations

import csv
import hashlib
import re
import tarfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

import requests
from datasets.features.features import Features, Value
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset._common import splits_to_audio_dataset
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

SOURCE_DATASET = "Multilingual LibriSpeech"
BASE_URL = "https://dl.fbaipublicfiles.com/mls"
DEFAULT_ROOT = Path("data/mls")
DEFAULT_LANGUAGES = ("polish",)
SOURCE_SPLITS = {"train": "train", "dev": "valid", "test": "eval"}

# ISO 639-1 code and the checksum published by OpenSLR 94 for each FLAC archive.
LANGUAGES = {
    "english": ("en", "9d4249911e318c2b8dcfcfecb484d865"),
    "german": ("de", "91ac982bf63869307f1b8950dfb7c776"),
    "dutch": ("nl", "6b171a16baff0108efd320b1ad65b9d1"),
    "french": ("fr", "4172e807697259bff9ad63661aecabf6"),
    "spanish": ("es", "6c34698dd522dde28fdc43309e9cc1ac"),
    "italian": ("it", "dc77f5805aecc7182aa20786032e5dc1"),
    "portuguese": ("pt", "12d54613fae75ae5fb1d55836408f3ee"),
    "polish": ("pl", "ce1a1278006cc373c9d1cb6dbfc03d47"),
}

MLS_FEATURES = Features(
    {
        **DATA_FEATURES,
        "audio_id": Value("string"),
        "transcript": Value("string"),
        "language": Value("string"),
        "language_code": Value("string"),
        "mls_split": Value("string"),
        "speaker_id": Value("int64"),
        "book_id": Value("int64"),
        "chapter_id": Value("string"),
        "utterance_id": Value("int64"),
        "source_audio_url": Value("string"),
        "source_segment_start": Value("float64"),
        "source_segment_end": Value("float64"),
    }
)

_AUDIO_ID = re.compile(r"(?P<speaker>\d+)_(?P<book>\d+)_(?P<utterance>\d+)")


def _select_languages(languages: Iterable[str]) -> tuple[str, ...]:
    selected = (languages,) if isinstance(languages, str) else tuple(languages)
    if not selected:
        raise ValueError("MLSLoader requires at least one language.")
    unknown = [language for language in selected if language not in LANGUAGES]
    if unknown:
        raise ValueError(
            f"Unknown MLS language {unknown[0]!r}; valid languages: {', '.join(LANGUAGES)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("MLSLoader languages must not contain duplicates.")
    return selected


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
    if partial.is_file() and _checksum(partial) == checksum:
        partial.replace(destination)
        return

    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if offset and response.status_code == 416:
                partial.unlink()
                continue
            response.raise_for_status()
            resumed = bool(offset and response.status_code == 206)
            with partial.open("ab" if resumed else "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
            break

    partial.replace(destination)
    if _checksum(destination) != checksum:
        destination.unlink()
        raise ValueError(f"Checksum mismatch for {destination.name}.")


def _prepare_language(root: Path, language: str) -> None:
    _, checksum = LANGUAGES[language]
    release = root / f"mls_{language}"
    marker = root / ".prepared" / language
    if release.is_dir() and marker.is_file() and marker.read_text().strip() == checksum:
        return

    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"mls_{language}.tar.gz"
    _download_archive(f"{BASE_URL}/{archive.name}", archive, checksum)

    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root, filter="data")
    if not release.is_dir():
        raise RuntimeError(f"{archive.name} did not contain {release.name}/.")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(checksum, encoding="utf-8")
    archive.unlink()


def _read_tsv(path: Path, field_count: int) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            values = [value.strip() for value in row]
            if len(values) != field_count or any(not value for value in values):
                raise ValueError(f"Malformed MLS metadata row in {path}:{line_number}.")
            audio_id, *metadata = values
            if audio_id in rows:
                raise ValueError(f"Duplicate MLS audio ID {audio_id!r} in {path}.")
            rows[audio_id] = metadata
    if not rows:
        raise ValueError(f"MLS metadata file {path} is empty.")
    return rows


def _release_root(root: Path, language: str, language_count: int) -> Path:
    release = root / f"mls_{language}"
    if release.is_dir():
        return release
    if language_count == 1 and root.name == f"mls_{language}" and root.is_dir():
        return root
    raise FileNotFoundError(
        f"MLS {language} release not found under {root}. Use prepare=True to download it."
    )


def _split_records(release: Path, language: str, source_split: str):
    split_root = release / source_split
    transcript_path = split_root / "transcripts.txt"
    segment_path = split_root / "segments.txt"
    if not transcript_path.is_file() or not segment_path.is_file():
        raise FileNotFoundError(
            f"MLS {language} {source_split} requires transcripts.txt and segments.txt."
        )

    transcripts = _read_tsv(transcript_path, 2)
    segments = _read_tsv(segment_path, 4)
    if transcripts.keys() != segments.keys():
        missing_segments = sorted(transcripts.keys() - segments.keys())
        missing_transcripts = sorted(segments.keys() - transcripts.keys())
        raise ValueError(
            f"MLS {language} {source_split} metadata IDs disagree: "
            f"missing segments={missing_segments[:3]}, "
            f"missing transcripts={missing_transcripts[:3]}."
        )

    language_code = LANGUAGES[language][0]
    split = SOURCE_SPLITS[source_split]
    for audio_id, (transcript,) in transcripts.items():
        match = _AUDIO_ID.fullmatch(audio_id)
        if match is None:
            raise ValueError(
                f"Malformed MLS audio ID {audio_id!r} in {transcript_path}."
            )
        speaker_text = match["speaker"]
        book_text = match["book"]
        speaker_id = int(speaker_text)
        book_id = int(book_text)
        utterance_id = int(match["utterance"])

        source_url, start_text, end_text = segments[audio_id]
        try:
            source_start = float(start_text)
            source_end = float(end_text)
        except ValueError as error:
            raise ValueError(f"Invalid MLS segment times for {audio_id!r}.") from error
        if source_start < 0 or source_end <= source_start:
            raise ValueError(f"Invalid MLS segment times for {audio_id!r}.")
        chapter_id = Path(urlsplit(source_url).path).stem
        if not chapter_id:
            raise ValueError(f"Invalid MLS source audio URL for {audio_id!r}.")

        audio_path = (
            split_root / "audio" / speaker_text / book_text / f"{audio_id}.flac"
        )
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing MLS audio for {audio_id!r}: {audio_path}")
        audio = AudioDecoder(str(audio_path)).metadata
        if audio.sample_rate != 16000 or audio.num_channels != 1:
            raise ValueError(f"MLS audio {audio_path} must be mono 16 kHz FLAC.")
        if audio.duration_seconds is None or audio.duration_seconds <= 0:
            raise ValueError(f"Could not determine the duration for {audio_path}.")

        yield {
            "audio_path": str(audio_path),
            "sample_rate": audio.sample_rate,
            "clip_offset": 0.0,
            "clip_duration": audio.duration_seconds,
            "class_list": [],
            "split": split,
            "source_dataset": SOURCE_DATASET,
            "metadata_path": str(transcript_path),
            "events": [],
            "num_channels": audio.num_channels,
            "channel_format": "mono",
            "environment": "",
            "audio_id": audio_id,
            "transcript": transcript,
            "language": language,
            "language_code": language_code,
            "mls_split": source_split,
            "speaker_id": speaker_id,
            "book_id": book_id,
            "chapter_id": chapter_id,
            "utterance_id": utterance_id,
            "source_audio_url": source_url,
            "source_segment_start": source_start,
            "source_segment_end": source_end,
        }


def _validate_speaker_splits(records: dict[str, list[dict]]) -> None:
    speakers = {
        split: {(row["language"], row["speaker_id"]) for row in rows}
        for split, rows in records.items()
    }
    for left, right in (("train", "valid"), ("train", "eval"), ("valid", "eval")):
        overlap = speakers[left] & speakers[right]
        if overlap:
            raise ValueError(
                f"MLS official splits {left!r} and {right!r} share speakers: "
                f"{sorted(overlap)[:3]}."
            )


class MLSLoader(BaseLoader):
    """Download and index selected languages from OpenSLR 94's FLAC release."""

    # Train transcripts are generated automatically; dev/test are human-corrected.
    label_source = LabelSource.MIXED

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        languages: Iterable[str] = DEFAULT_LANGUAGES,
        prepare: bool = True,
    ) -> None:
        super().__init__(root=root, prepare=prepare)
        self.languages = _select_languages(languages)

    def prepare_raw(self) -> None:
        if "english" in self.languages:
            raise ValueError(
                "MLS English's 2.4 TB FLAC archive is not downloaded automatically. "
                "Use prepare=False with an existing mls_english extraction."
            )
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        for language in self.languages:
            _prepare_language(root, language)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "MLSLoader needs a root path. Use prepare=False with an existing extraction "
                "or prepare=True to download it."
            )

        records = {"train": [], "valid": [], "eval": []}
        for language in self.languages:
            release = _release_root(self.root, language, len(self.languages))
            for source_split, split in SOURCE_SPLITS.items():
                records[split].extend(_split_records(release, language, source_split))
        _validate_speaker_splits(records)
        return splits_to_audio_dataset(
            records,
            features=MLS_FEATURES,
            label_source=self.label_source,
        )


def main() -> AudioDataset:
    return MLSLoader()()


if __name__ == "__main__":
    mls = main()
    print(mls.info())
