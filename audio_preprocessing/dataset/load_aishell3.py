"""Load the official AISHELL-3 multi-speaker Mandarin TTS corpus into an AudioDataset.

AISHELL-3 (OpenSLR SLR93, Apache License 2.0) contains 88,035 utterances read by 218
native Mandarin speakers at 44.1 kHz / 16-bit mono. It ships as a single 19 GB archive
(``data_aishell3.tgz``) whose members sit flat at the archive root (verified directly
against the real archive's headers, not inferred from prose): extracting it into a
destination directory drops ``spk-info.txt`` plus a ``train/`` and ``test/`` folder
directly into that directory, with no wrapping ``data_aishell3/`` folder. Each split has
a ``content.txt`` transcript index and a ``wav/<speaker_id>/<utterance_id>.wav`` tree.
There is no smaller official subset and no official validation split, so this loader
honors the two splits it actually ships: ``train`` maps to ``train`` and ``test`` maps to
this project's ``eval``.

``content.txt`` pairs each utterance with a whitespace-separated, alternating
character/pinyin transcript (e.g. ``guang3`` after ``广``); this holds for all 88,035
utterances in the released corpus (verified directly, including erhua/儿化 contractions
such as ``可儿 ker3``, which still pair one fused token to one pinyin syllable), so an odd
token count is treated as a hard error rather than something to work around.

``spk-info.txt`` carries each speaker's age group, gender, and accent; some real rows
have trailing whitespace on the gender field, so parsing goes through
``_common.read_tsv_rows``, which strips every field. Manual transcription was reviewed
for quality, so ``label_source`` is ``gold``.

OpenSLR does not publish a checksum for this archive, so completeness is tracked with a
``.prepared`` marker written only after a full extraction succeeds, rather than a hash
comparison.

Citation: Shi et al., "AISHELL-3: A Multi-speaker Mandarin TTS Corpus and the
Baselines" (https://arxiv.org/abs/2010.11567).

Run standalone with an existing local extraction: ``python -m
audio_preprocessing.dataset.load_aishell3``.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import requests
from datasets.features.features import Features, Value
from tqdm import tqdm

from audio_preprocessing.dataset._common import (
    build_audio_record,
    read_tsv_rows,
    splits_to_audio_dataset,
)
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import AudioDataset, DATA_FEATURES, LabelSource

SOURCE_DATASET = "AISHELL-3"
ARCHIVE_URL = "https://www.openslr.org/resources/93/data_aishell3.tgz"
DEFAULT_ROOT = Path("data/aishell3")

# A file that only exists once the archive has actually been extracted into a root; used
# to sanity-check an extraction (and, via read_tsv_rows, is real corpus content anyway).
SENTINEL_FILE = "spk-info.txt"

# Official directory name per split; both map to this project's split names.
SPLIT_DIRS = {"train": "train", "eval": "test"}

AISHELL3_FEATURES = Features({
    **DATA_FEATURES,
    "sample_id": Value("string"),
    "speaker_id": Value("string"),
    "text": Value("string"),
    "pinyin": Value("string"),
    "gender": Value("string"),
    "age_group": Value("string"),
    "accent": Value("string"),
})


def _stream_to_partial(url: str, partial: Path) -> None:
    """Stream ``url`` into ``partial``, resuming an existing prefix when possible."""
    resume = partial.exists()
    while True:
        offset = partial.stat().st_size if resume else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if offset and response.status_code == 416:
                # The .part is already at least as long as the server file; it can only
                # be stale (a prior run wrote a bad tail), so discard and restart clean.
                partial.unlink()
                resume = False
                continue
            response.raise_for_status()
            resumed = bool(offset and response.status_code == 206)
            start = offset if resumed else 0
            length = int(response.headers.get("content-length", 0))
            total = start + length if length else None
            with (
                partial.open("ab" if resumed else "wb") as handle,
                tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=partial.name) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
            return


def _prepare(root: Path) -> None:
    """Download and extract ``data_aishell3.tgz`` into ``root`` unless already done."""
    marker = root / ".prepared"
    if marker.is_file():
        return

    archive = root / "data_aishell3.tgz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if not archive.is_file():
        _stream_to_partial(ARCHIVE_URL, partial)
        partial.replace(archive)

    with tarfile.open(archive) as handle:
        handle.extractall(root, filter="data")
    if not (root / SENTINEL_FILE).is_file():
        raise RuntimeError(
            f"{archive.name} did not extract the expected AISHELL-3 layout under {root}."
        )
    marker.write_text("", encoding="utf-8")
    archive.unlink()


def _speaker_info(corpus_dir: Path) -> dict[str, dict[str, str]]:
    """Parse ``spk-info.txt`` into ``{speaker_id: {age_group, gender, accent}}``."""
    path = corpus_dir / "spk-info.txt"
    info: dict[str, dict[str, str]] = {}
    for row in read_tsv_rows(path):
        if row[0].startswith("#"):
            continue
        if len(row) != 4:
            raise ValueError(f"Malformed spk-info.txt row in {path}: {row!r}")
        speaker_id, age_group, gender, accent = row
        info[speaker_id] = {"age_group": age_group, "gender": gender, "accent": accent}
    return info


def _records(corpus_dir: Path, split_dir: str, split: str, speaker_info: dict[str, dict[str, str]]):
    content_path = corpus_dir / split_dir / "content.txt"
    wav_root = corpus_dir / split_dir / "wav"

    rows = read_tsv_rows(content_path)
    for row in tqdm(rows, desc=f"AISHELL-3 {split_dir}"):
        if len(row) != 2:
            raise ValueError(f"Malformed content.txt row in {content_path}: {row!r}")
        utterance_id, raw_text = row
        if not utterance_id.endswith(".wav"):
            raise ValueError(
                f"Expected a .wav utterance id in {content_path}, got {utterance_id!r}."
            )

        tokens = raw_text.split()
        if len(tokens) % 2 != 0:
            raise ValueError(
                f"Expected an alternating character/pinyin transcript with an even token "
                f"count for {utterance_id} in {content_path}, got {len(tokens)} tokens."
            )

        speaker_id = utterance_id[:7]
        speaker = speaker_info.get(speaker_id)
        if speaker is None:
            raise ValueError(
                f"No spk-info.txt entry for speaker {speaker_id!r} (utterance {utterance_id})."
            )

        audio_path = wav_root / speaker_id / utterance_id
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing AISHELL-3 audio for {utterance_id}: {audio_path}")

        record = build_audio_record(
            audio_path,
            None,
            split=split,
            source_dataset=SOURCE_DATASET,
            metadata_path=str(content_path),
        )
        if record["sample_rate"] != 44100 or record["num_channels"] != 1:
            raise ValueError(
                f"Expected 44.1 kHz mono audio for {utterance_id}, got "
                f"{record['sample_rate']} Hz / {record['num_channels']} channel(s)."
            )
        record.update({
            "sample_id": utterance_id.removesuffix(".wav"),
            "speaker_id": speaker_id,
            "text": "".join(tokens[0::2]),
            "pinyin": " ".join(tokens[1::2]),
            "gender": speaker["gender"],
            "age_group": speaker["age_group"],
            "accent": speaker["accent"],
        })
        yield record


class AISHELL3Loader(BaseLoader):
    """Index AISHELL-3, optionally downloading it from OpenSLR.

    Transcripts were professionally annotated and quality-checked, so ``label_source`` is
    gold. The archive has no smaller official subset and no official validation split;
    preparation is opt-in because it is ~19 GB.
    """

    label_source = LabelSource.GOLD

    def __init__(self, root: str | Path | None = None, *, prepare: bool = False) -> None:
        super().__init__(root=root, prepare=prepare)

    def prepare_raw(self) -> None:
        root = self.root or DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        _prepare(root)

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "AISHELL3Loader needs a root path. Use prepare=False with an existing "
                "extraction or prepare=True to download it (~19 GB)."
            )

        corpus_dir = self.root
        if not (corpus_dir / SENTINEL_FILE).is_file():
            raise FileNotFoundError(
                f"Expected an extracted AISHELL-3 layout ({SENTINEL_FILE}, train/, test/) "
                f"under {corpus_dir}. Use prepare=True to download and extract it."
            )

        speaker_info = _speaker_info(corpus_dir)
        records = {
            split: list(_records(corpus_dir, split_dir, split, speaker_info))
            for split, split_dir in SPLIT_DIRS.items()
        }
        return splits_to_audio_dataset(
            records,
            features=AISHELL3_FEATURES,
            label_source=self.label_source,
        )


def main():
    # The 19 GB archive has no smaller official subset (see the module docstring), so the
    # standalone smoke test reads an existing local extraction instead of downloading.
    return AISHELL3Loader(prepare=False)()


if __name__ == "__main__":
    dataset = main()
    print(dataset.info())
