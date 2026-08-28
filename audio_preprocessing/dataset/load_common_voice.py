"""Load selected Mozilla Common Voice 17 locales from a pinned Hugging Face mirror.

Mozilla withdrew Common Voice from the Hub in October 2025, leaving the ``mozilla-foundation``
repositories empty, so this reads the third-party ``fixie-ai`` Parquet mirror at a pinned
revision. Clips are materialized under ``root``, so ``prepare=False`` needs no network.

    uv run audio_preprocessing/dataset/load_common_voice.py
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Audio
from datasets.features.features import Features, List, Value
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from audio_preprocessing.dataset._common import (
    build_audio_record,
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import HFLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

DATASET_NAME = "fixie-ai/common_voice_17_0"
DEFAULT_REVISION = "34f78a43893414e7b6e271ba94c1d5e05f18b239"
SOURCE_DATASET = "CommonVoice"
DEFAULT_ROOT = Path("data/common_voice")
DEFAULT_LOCALES = ("ast",)

# validated/other/invalidated are deliberately unselectable: validated is a superset of
# train/dev/test and would double-count, and the other two failed or never received votes.
SPLITS = {"train": "train", "valid": "validation", "eval": "test"}

# A locale becomes a path segment under root, so keep it to a bare corpus code.
LOCALE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]+)*$")

# The mirror's model-generated ``continuation`` column is excluded to keep label_source gold.
COMMON_VOICE_FEATURES = Features({
    **DATA_FEATURES,
    "clip_name": Value("string"),
    "client_id": Value("string"),
    "transcript": Value("string"),
    "up_votes": Value("int64"),
    "down_votes": Value("int64"),
    "age": Value("string"),
    "gender": Value("string"),
    "accents": List(Value("string")),
    "locale": Value("string"),
    "variant": Value("string"),
    "segment": Value("string"),
})


class CommonVoiceLoader(HFLoader):
    """Materialize and index selected Common Voice locales and official splits.

    Documentation: https://commonvoice.mozilla.org/
    Mirror: https://huggingface.co/datasets/fixie-ai/common_voice_17_0
    License: CC0-1.0
    """

    dataset_name = DATASET_NAME
    # Prompts are human-authored and human-read, and every official-split clip passed
    # contributor votes. Demographics are self-reported but still human-supplied.
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        locales: Iterable[str] = DEFAULT_LOCALES,
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
        self.root = self.root or DEFAULT_ROOT
        self.locales = validate_choices(locales, name="locale", context="Common Voice")
        if invalid := [name for name in self.locales if not LOCALE_RE.fullmatch(name)]:
            raise ValueError(f"Invalid Common Voice locale(s): {invalid}.")
        self.splits = validate_choices(
            splits, name="split", context="Common Voice", allowed=SPLITS
        )

    @staticmethod
    def _accents(value: Any) -> list[str]:
        """Split the comma-joined accents; unreported is an empty string, not a null."""
        return [part.strip() for part in str(value or "").split(",") if part.strip()]

    def _metadata(self, row: dict[str, Any], locale: str) -> dict[str, Any]:
        # ``path`` is absolute on the mirror author's machine, so only the basename means
        # anything; it doubles as the corpus clip ID and the materialized filename.
        clip_name = Path(str(row["path"])).name
        if Path(clip_name).suffix != ".mp3":
            raise ValueError(f"Unexpected Common Voice path {row['path']!r}.")
        if str(row["locale"]) != locale:
            raise ValueError(
                f"Common Voice clip {clip_name!r} reports locale {str(row['locale'])!r}, not "
                f"the requested {locale!r}."
            )

        return {
            "clip_name": clip_name,
            "client_id": str(row["client_id"]),
            "transcript": str(row["sentence"]),
            "up_votes": row["up_votes"],
            "down_votes": row["down_votes"],
            # Parquet nulls must not stringify into a literal "None" demographic.
            "age": str(row["age"] or ""),
            "gender": str(row["gender"] or ""),
            "accents": self._accents(row["accent"]),
            "locale": locale,
            "variant": str(row["variant"] or ""),
            "segment": str(row["segment"] or ""),
        }

    def _materialize(self, locale: str, split: str) -> None:
        self.config_name = locale
        self.load_kwargs = {"split": SPLITS[split]}
        super().prepare_raw()
        rows = self.raw.cast_column("audio", Audio(decode=False))

        split_dir = self.root / "materialized" / locale / split
        split_dir.parent.mkdir(parents=True, exist_ok=True)
        # Promote only after every row is written, so an interrupted run leaves no short split.
        with tempfile.TemporaryDirectory(dir=split_dir.parent, prefix=f".{split}.") as staging:
            written: set[str] = set()
            for row in tqdm(rows, desc=f"Common Voice {locale}/{split}", unit="clip"):
                content = row["audio"]["bytes"] or Path(row["audio"]["path"]).read_bytes()
                metadata = self._metadata(row, locale)
                clip_name = metadata["clip_name"]
                if clip_name in written:
                    raise ValueError(
                        f"Duplicate Common Voice clip {clip_name!r} in {locale}/{split}."
                    )
                written.add(clip_name)
                # Decode now so a corrupt clip fails before the split is promoted.
                if not AudioDecoder(content).metadata.duration_seconds:
                    raise ValueError(f"Common Voice clip {clip_name!r} has no decodable audio.")

                clip = Path(staging) / clip_name
                clip.write_bytes(content)
                clip.with_suffix(".json").write_text(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding="utf-8"
                )

            # Staging sits beside the split, so this rename is atomic.
            if split_dir.exists():
                shutil.rmtree(split_dir)
            Path(staging).replace(split_dir)

    def prepare_raw(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.cache_dir or self.root / ".hf_cache"
        for locale in self.locales:
            for split in self.splits:
                self._materialize(locale, split)

    def build_dataset(self) -> AudioDataset:
        records: dict[str, list[dict[str, Any]]] = {split: [] for split in self.splits}
        for locale in self.locales:
            for split in self.splits:
                split_dir = self.root / "materialized" / locale / split
                sidecars = sorted(split_dir.glob("*.json")) if split_dir.is_dir() else []
                if not sidecars:
                    raise FileNotFoundError(
                        f"No prepared Common Voice metadata for {locale}/{split} under "
                        f"{self.root}."
                    )
                records[split] += [
                    build_audio_record(
                        sidecar.with_suffix(".mp3"),
                        None,
                        split=split,
                        source_dataset=SOURCE_DATASET,
                        metadata_path=str(sidecar),
                    ) | json.loads(sidecar.read_text(encoding="utf-8"))
                    for sidecar in sidecars
                ]

        dataset = splits_to_audio_dataset(
            records, features=COMMON_VOICE_FEATURES, label_source=self.label_source
        )
        # The active split defaults to "train", which a valid- or eval-only build never has.
        if "train" not in dataset.data:
            dataset.split = next(iter(dataset.data))
        return dataset


def main() -> AudioDataset:
    return CommonVoiceLoader(locales=("ast",), splits=("valid",))()


if __name__ == "__main__":
    print(main().info())
