"""Load selected VoxPopuli transcribed language configurations from Hugging Face.

VoxPopuli cuts 2009-2020 European Parliament plenary recordings into utterances. Only the
transcribed set is on the Hub; the unlabelled and speech-to-speech portions ship through
facebookresearch/voxpopuli.

``normalized_text`` is the canonical ``transcript`` -- the dataset card's own example row
leaves ``raw_text`` empty on a populated record. Transcripts are human-authored but utterance
boundaries come from automatic force-alignment, hence ``label_source`` is ``mixed``.

Rows are pre-cut, so ``clip_offset`` is always 0.0; the time inside ``audio_id`` locates the
utterance in the original plenary recording, not in the materialized clip. Clips are written
under ``root`` so ``prepare=False`` is network-free.

    uv run audio_preprocessing/dataset/load_voxpopuli.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Audio
from datasets.features.features import Features, Value
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from audio_preprocessing.dataset._common import (
    build_audio_record,
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import HFLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

DATASET_NAME = "facebook/voxpopuli"
DEFAULT_REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
SOURCE_DATASET = "VoxPopuli"
DEFAULT_ROOT = Path("data/voxpopuli")
# lt is the smallest configuration (501 rows, ~0.29 GB); multilang is ~334 GB.
DEFAULT_LANGUAGES = ("lt",)

# Source class-label order; row["language"] indexes into it.
LANGUAGES = (
    "en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs",
    "nl", "fi", "hr", "sk", "sl", "et", "lt",
    "en_accented", "multilang",
)

SPLITS = {"train": "train", "valid": "validation", "eval": "test"}

TEST_ONLY_LANGUAGES = frozenset({"en_accented"})

# Mixes languages, so a row's language cannot be checked against the configuration name.
MULTILINGUAL_LANGUAGES = frozenset({"multilang"})

# The card's not-applicable sentinels: "None" for accent, "na" for gender.
_UNREPORTED = frozenset({"", "None", "na"})

# audio_id carries no extension and the release may not stay in one container.
_AUDIO_MAGIC = ((b"RIFF", ".wav"), (b"OggS", ".ogg"), (b"fLaC", ".flac"), (b"ID3", ".mp3"))

VOXPOPULI_FEATURES = Features(
    {
        **DATA_FEATURES,
        "clip_name": Value("string"),
        "audio_id": Value("string"),
        "language": Value("string"),
        "transcript": Value("string"),
        "raw_text": Value("string"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "accent": Value("string"),
        "is_gold_transcript": Value("bool"),
    }
)


def _reported(value: str) -> str:
    return "" if value in _UNREPORTED else value


def _audio_suffix(content: bytes) -> str:
    for magic, suffix in _AUDIO_MAGIC:
        if content.startswith(magic):
            return suffix
    raise ValueError(
        f"Unrecognized VoxPopuli audio container starting with {content[:4]!r}."
    )


def _clip_name(audio_id: str, suffix: str) -> str:
    """audio_id embeds a wall-clock time, whose colons are not portable in filenames."""
    name = f"{audio_id.replace(':', '-')}{suffix}"
    if Path(name).name != name:
        raise ValueError(f"VoxPopuli audio_id {audio_id!r} is not a usable filename.")
    return name


def _audio_bytes(row: dict[str, Any]) -> bytes:
    content = row["audio"].get("bytes")
    if content:
        return content
    path = row["audio"].get("path")
    if not path or not Path(path).is_file():
        raise ValueError(
            f"VoxPopuli row {row.get('audio_id')!r} has no readable audio content."
        )
    return Path(path).read_bytes()


def _metadata(row: dict[str, Any], language: str, languages: Any) -> dict[str, Any]:
    """Normalize one row into the columns VoxPopuli adds to ``DATA_FEATURES``."""
    audio_id = str(row["audio_id"])
    row_language = str(languages.int2str(int(row["language"])))
    if language not in MULTILINGUAL_LANGUAGES and row_language != language:
        raise ValueError(
            f"VoxPopuli clip {audio_id!r} reports language {row_language!r}, not the "
            f"requested configuration {language!r}."
        )

    transcript = str(row["normalized_text"])
    raw_text = str(row["raw_text"])
    # raw_text is legitimately empty on some gold rows; only a row with neither is unusable.
    if not transcript.strip() and not raw_text.strip():
        raise ValueError(f"VoxPopuli clip {audio_id!r} has no transcript text.")

    return {
        "audio_id": audio_id,
        "language": row_language,
        "transcript": transcript,
        "raw_text": raw_text,
        "speaker_id": str(row["speaker_id"]),
        "gender": _reported(str(row["gender"])),
        "accent": _reported(str(row["accent"])),
        "is_gold_transcript": bool(row["is_gold_transcript"]),
    }


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


def _record(sidecar: Path, split: str) -> dict[str, Any]:
    audio_path = sidecar.with_suffix("")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing VoxPopuli audio for {sidecar}: {audio_path}")

    return {
        **build_audio_record(
            audio_path,
            None,
            split=split,
            source_dataset=SOURCE_DATASET,
            metadata_path=str(sidecar),
        ),
        "clip_name": audio_path.name,
        **json.loads(sidecar.read_text(encoding="utf-8")),
    }


class VoxPopuliLoader(HFLoader):
    """Materialize and index selected VoxPopuli languages and official splits."""

    dataset_name = DATASET_NAME
    label_source = LabelSource.MIXED

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        languages: Iterable[str] = DEFAULT_LANGUAGES,
        splits: Iterable[str] = SPLITS,
        revision: str | None = DEFAULT_REVISION,
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
        self.languages = validate_choices(
            languages, name="language", context="VoxPopuli", allowed=LANGUAGES
        )
        self.splits = validate_choices(
            splits, name="split", context="VoxPopuli", allowed=SPLITS
        )

        test_only = sorted(set(self.languages) & TEST_ONLY_LANGUAGES)
        unavailable = [split for split in self.splits if split != "eval"]
        if test_only and unavailable:
            raise ValueError(
                f"VoxPopuli {test_only} ships only a test split, so {unavailable} cannot "
                f"be built. Request splits=('eval',) for those configurations."
            )

    def prepare_raw(self) -> None:
        self.root = root = self.root or DEFAULT_ROOT
        self.cache_dir = self.cache_dir or root / ".hf_cache"
        for language in self.languages:
            for split in self.splits:
                self._materialize(root, language, split)

    def _fetch(self, language: str, split: str) -> Any:
        """Download one configuration and split through ``HFLoader.prepare_raw``."""
        self.config_name = language
        self.load_kwargs = {"split": SPLITS[split]}
        super().prepare_raw()
        return self.raw.cast_column("audio", Audio(decode=False))

    def _materialize(self, root: Path, language: str, split: str) -> None:
        rows = self._fetch(language, split)
        languages = rows.features["language"]
        split_dir = root / "materialized" / language / split
        split_dir.parent.mkdir(parents=True, exist_ok=True)

        # Promote only once every row is written, so an interrupted run never leaves a
        # split that looks complete but is short.
        with tempfile.TemporaryDirectory(
            dir=split_dir.parent, prefix=f".{split}.", suffix=".part"
        ) as staging:
            staged = Path(staging)
            written: set[str] = set()
            skipped: list[str] = []
            for row in tqdm(rows, desc=f"VoxPopuli {language}/{split}", unit="clip"):
                content = _audio_bytes(row)
                # A few rows are a bare header with no payload (13 of 8387 in en_accented).
                if not AudioDecoder(content).metadata.duration_seconds:
                    skipped.append(str(row["audio_id"]))
                    continue

                metadata = _metadata(row, language, languages)
                clip_name = _clip_name(metadata["audio_id"], _audio_suffix(content))
                if clip_name in written:
                    raise ValueError(
                        f"Duplicate VoxPopuli clip {clip_name!r} in {language}/{split}."
                    )
                written.add(clip_name)

                (staged / clip_name).write_bytes(content)
                # Append rather than replace the suffix: audio_id may contain a dot.
                (staged / f"{clip_name}.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )

            if not written:
                raise ValueError(
                    f"Every VoxPopuli row in {language}/{split} had unreadable audio, so "
                    f"there is no split to promote."
                )
            if skipped:
                warnings.warn(
                    f"VoxPopuli {language}/{split}: skipped {len(skipped)} of "
                    f"{len(skipped) + len(written)} rows whose audio has no decodable "
                    f"samples, first {skipped[0]!r}.",
                    stacklevel=2,
                )
            _replace_split(staged, split_dir)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "VoxPopuliLoader needs a root path. Use prepare=True to materialize "
                f"selected languages under {DEFAULT_ROOT}, or pass root= pointing at an "
                "existing materialization when prepare=False."
            )

        records: dict[str, list[dict[str, Any]]] = {split: [] for split in self.splits}
        for language in self.languages:
            for split in self.splits:
                split_dir = self.root / "materialized" / language / split
                sidecars = sorted(split_dir.glob("*.json")) if split_dir.is_dir() else []
                if not sidecars:
                    raise FileNotFoundError(
                        f"No prepared VoxPopuli metadata for {language}/{split} under "
                        f"{self.root}."
                    )
                records[split].extend(_record(path, split) for path in sidecars)

        dataset = splits_to_audio_dataset(
            records, features=VOXPOPULI_FEATURES, label_source=self.label_source
        )
        # The active split defaults to "train", which an en_accented build never has.
        if "train" not in dataset.data:
            dataset.split = next(iter(dataset.data))
        return dataset


def main() -> AudioDataset:
    return VoxPopuliLoader(languages=("lt",), splits=("eval",))()


if __name__ == "__main__":
    voxpopuli = main()
    print(voxpopuli.info())
