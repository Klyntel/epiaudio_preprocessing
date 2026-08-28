"""Load selected official FLEURS language configurations from Hugging Face.

FLEURS is a non-spatial multilingual ASR benchmark. Its 102 language configurations
share source-sentence IDs, which this loader preserves so parallel utterances remain
joinable across languages. Audio and metadata are materialized under ``root`` so
``prepare=False`` is network-free.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Audio
from datasets.features.features import Features, Value
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.dataset import base_loader
from audio_preprocessing.dataset._common import (
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import HFLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

DATASET_NAME = "google/fleurs"
SOURCE_DATASET = "FLEURS"
DEFAULT_ROOT = Path("data/fleurs")
DEFAULT_LANGUAGES = ("is_is",)
SPLITS = {"train": "train", "valid": "validation", "eval": "test"}
GENDERS = ("male", "female", "other")
LANGUAGE_GROUPS = (
    "western_european_we",
    "eastern_european_ee",
    "central_asia_middle_north_african_cmn",
    "sub_saharan_african_ssa",
    "south_asian_sa",
    "south_east_asian_sea",
    "chinese_japanase_korean_cjk",
)
_LANGUAGE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

FLEURS_FEATURES = Features(
    {
        **DATA_FEATURES,
        "source_sentence_id": Value("int64"),
        "num_samples": Value("int64"),
        "path": Value("string"),
        "transcription": Value("string"),
        "raw_transcription": Value("string"),
        "gender_id": Value("int64"),
        "gender": Value("string"),
        "lang_id": Value("int64"),
        "language_code": Value("string"),
        "language": Value("string"),
        "lang_group_id": Value("int64"),
        "language_group": Value("string"),
    }
)


def _language_selection(languages: Iterable[str]) -> tuple[str, ...]:
    selected = validate_choices(languages, name="language", context="FLEURS")
    invalid = [
        language for language in selected if not _LANGUAGE_RE.fullmatch(language)
    ]
    if invalid:
        raise ValueError(f"Invalid FLEURS language configuration(s): {invalid}.")
    if "all" in selected:
        raise ValueError(
            "Select explicit FLEURS language configurations instead of 'all'."
        )
    return selected


def _validate_and_normalize_row_metadata(
    row: dict[str, Any],
    expected_language_code: str,
    audio_bytes: bytes,
    language_feature: Any,
) -> dict[str, Any]:
    required = {
        "id",
        "num_samples",
        "path",
        "transcription",
        "raw_transcription",
        "gender",
        "lang_id",
        "language",
        "lang_group_id",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"FLEURS row is missing fields: {missing}.")

    source_path = str(row["path"])
    audio_name = Path(source_path).name
    if not audio_name or Path(audio_name).suffix != ".wav":
        raise ValueError(f"Unexpected FLEURS path {source_path!r}.")

    try:
        source_sentence_id = int(row["id"])
        num_samples = int(row["num_samples"])
        lang_id = int(row["lang_id"])
    except (TypeError, ValueError):
        raise ValueError(
            "FLEURS id, num_samples, and lang_id must be integers."
        ) from None
    if source_sentence_id < 0 or num_samples <= 0 or lang_id < 0:
        raise ValueError(
            "FLEURS id and lang_id must be non-negative and num_samples positive."
        )

    try:
        language_code = str(language_feature.int2str(lang_id))
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError(f"Invalid FLEURS lang_id value {row['lang_id']!r}.") from None
    if language_code != expected_language_code:
        raise ValueError(
            f"FLEURS lang_id {lang_id} maps to {language_code!r}, not requested "
            f"configuration {expected_language_code!r}."
        )

    try:
        gender_id = int(row["gender"])
        group_id = int(row["lang_group_id"])
    except (TypeError, ValueError):
        raise ValueError("FLEURS gender and lang_group_id must be integers.") from None
    if not 0 <= gender_id < len(GENDERS):
        raise ValueError(f"Invalid FLEURS gender value {row['gender']!r}.")
    if not 0 <= group_id < len(LANGUAGE_GROUPS):
        raise ValueError(
            f"Invalid FLEURS lang_group_id value {row['lang_group_id']!r}."
        )
    gender = GENDERS[gender_id]
    language_group = LANGUAGE_GROUPS[group_id]

    audio = AudioDecoder(audio_bytes).metadata
    if audio.sample_rate != 16_000 or audio.num_channels != 1:
        raise ValueError(
            f"FLEURS audio {source_path!r} must be mono 16 kHz; found "
            f"{audio.num_channels} channel(s) at {audio.sample_rate} Hz."
        )
    duration = audio.duration_seconds
    if duration is None:
        raise ValueError(
            f"Could not determine the duration for FLEURS audio {source_path!r}."
        )
    # FLEURS metadata contains incorrect sample counts; the WAV is authoritative.
    num_samples = round(duration * 16_000)
    expected_duration = num_samples / 16_000

    return {
        "audio_name": audio_name,
        "sample_rate": 16_000,
        "clip_duration": expected_duration,
        "num_channels": 1,
        "source_sentence_id": source_sentence_id,
        "num_samples": num_samples,
        "path": source_path,
        "transcription": str(row["transcription"]),
        "raw_transcription": str(row["raw_transcription"]),
        "gender_id": gender_id,
        "gender": gender,
        "lang_id": lang_id,
        "language_code": language_code,
        "language": str(row["language"]),
        "lang_group_id": group_id,
        "language_group": language_group,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_bytes(content)
    temporary.replace(path)


def _replace_split(staged: Path, destination: Path) -> None:
    """Install a staged split, restoring the existing split if promotion fails."""
    if not destination.exists():
        staged.replace(destination)
        return

    backup = staged.with_name(f"{staged.name}.old")
    destination.replace(backup)
    try:
        staged.replace(destination)
    except BaseException:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def _build_record(metadata_path: Path, split: str) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audio_path = metadata_path.with_suffix(".wav")
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Missing FLEURS audio for {metadata_path}: {audio_path}"
        )

    audio = AudioDecoder(str(audio_path)).metadata
    if (
        audio.sample_rate != metadata["sample_rate"]
        or audio.num_channels != metadata["num_channels"]
        or audio.duration_seconds is None
        or not math.isclose(
            audio.duration_seconds,
            metadata["clip_duration"],
            rel_tol=0,
            abs_tol=1 / metadata["sample_rate"],
        )
    ):
        raise ValueError(f"FLEURS audio header disagrees with {metadata_path}.")

    return {
        "audio_path": str(audio_path),
        "sample_rate": metadata["sample_rate"],
        "clip_offset": 0.0,
        "clip_duration": metadata["clip_duration"],
        "class_list": [],
        "split": split,
        "source_dataset": SOURCE_DATASET,
        "metadata_path": str(metadata_path),
        "events": [],
        "num_channels": metadata["num_channels"],
        "channel_format": "mono",
        "environment": "",
        **{
            field: metadata[field]
            for field in FLEURS_FEATURES
            if field not in DATA_FEATURES
        },
    }


class FLEURSLoader(HFLoader):
    """Download and index selected FLEURS languages and official splits."""

    dataset_name = DATASET_NAME
    label_source = LabelSource.MIXED

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        languages: Iterable[str] = DEFAULT_LANGUAGES,
        splits: Iterable[str] = SPLITS,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        prepare: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            revision=revision,
            cache_dir=cache_dir,
            streaming=False,
            prepare=prepare,
        )
        self.languages = _language_selection(languages)
        self.splits = validate_choices(
            splits,
            name="split",
            context="FLEURS",
            allowed=SPLITS,
        )

    def prepare_raw(self) -> None:
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        cache_dir = (
            Path(self.cache_dir) if self.cache_dir is not None else root / ".hf_cache"
        )
        self.cache_dir = cache_dir

        for language in self.languages:
            for split in self.splits:
                kwargs: dict[str, Any] = {
                    "split": SPLITS[split],
                    "streaming": False,
                    "cache_dir": str(cache_dir),
                }
                if self.revision is not None:
                    kwargs["revision"] = self.revision
                rows = base_loader.load_dataset(DATASET_NAME, language, **kwargs)
                rows = rows.cast_column("audio", Audio(decode=False))
                try:
                    language_feature = rows.features["lang_id"]
                except (AttributeError, KeyError, TypeError):
                    raise ValueError(
                        "FLEURS source is missing the lang_id class-label feature."
                    ) from None
                split_dir = root / "materialized" / language / split
                split_dir.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    dir=split_dir.parent,
                    prefix=f".{split}.",
                    suffix=".part",
                ) as staging:
                    staged_split = Path(staging)
                    destinations: set[Path] = set()
                    for row in rows:
                        audio = row.get("audio")
                        if not isinstance(audio, dict):
                            raise ValueError("FLEURS row is missing its audio object.")
                        content = audio.get("bytes")
                        if not isinstance(content, bytes) or not content:
                            path = audio.get("path")
                            if not path or not Path(path).is_file():
                                raise ValueError(
                                    f"FLEURS row {row.get('id')!r} has no readable "
                                    "audio content."
                                )
                            content = Path(path).read_bytes()
                        metadata = _validate_and_normalize_row_metadata(
                            row,
                            language,
                            content,
                            language_feature,
                        )
                        audio_path = staged_split / metadata.pop("audio_name")
                        if audio_path in destinations:
                            raise ValueError(
                                f"Duplicate FLEURS path {audio_path.name!r} in "
                                f"{language}/{split}."
                            )
                        destinations.add(audio_path)
                        _atomic_write(audio_path, content)
                        _atomic_write(
                            audio_path.with_suffix(".json"),
                            json.dumps(
                                metadata,
                                ensure_ascii=False,
                                sort_keys=True,
                            ).encode("utf-8"),
                        )
                    _replace_split(staged_split, split_dir)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "FLEURSLoader needs a root path. Use prepare=True to download selected "
                "configurations or prepare=False with an existing materialization."
            )

        records: dict[str, list[dict[str, Any]]] = {split: [] for split in self.splits}
        for language in self.languages:
            for split in self.splits:
                split_dir = self.root / "materialized" / language / split
                metadata_paths = (
                    sorted(split_dir.glob("*.json")) if split_dir.is_dir() else []
                )
                if not metadata_paths:
                    raise FileNotFoundError(
                        f"No prepared FLEURS metadata for {language}/{split} under {self.root}."
                    )
                records[split].extend(
                    _build_record(path, split) for path in metadata_paths
                )
        return splits_to_audio_dataset(
            records,
            features=FLEURS_FEATURES,
            label_source=self.label_source,
        )


def main() -> AudioDataset:
    return FLEURSLoader(languages=("is_is",), splits=("valid",))()


if __name__ == "__main__":
    fleurs = main()
    print(fleurs.info())
