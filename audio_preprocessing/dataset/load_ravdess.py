"""Load the RAVDESS emotional speech and song corpus into an AudioDataset.

RAVDESS (Zenodo record 1188976) holds 7356 files of 24 professional actors (12 male, 12
female) performing two lexically matched statements in North American English, released in
audio-only, audio-video, and video-only modalities. Only the two audio-only archives carry
audio this pipeline can use, so this loader never requests the record's 47 per-actor video
archives, which are the other ~24 GB of it:

- ``Audio_Speech_Actors_01-24.zip`` (~199 MiB) -- 1440 clips, 60 per actor x 24 actors.
- ``Audio_Song_Actors_01-24.zip`` (~215 MiB) -- 1012 clips, 44 per actor x 23 actors.
  There are no song files for Actor_18.

Both archives hold ``Actor_NN/`` folders at their top level with no wrapping directory
(verified against the real archives' zip central directories rather than inferred from the
record's prose), so ``download_zenodo`` leaves each one at
``<root>/<archive stem>/Actor_NN/*.wav``. The song archive still ships an ``Actor_18/``
folder; it is simply empty.

Every clip is 48 kHz 16-bit, but they are not uniformly mono: decoding both real archives
shows 6 of the 2452 clips (5 speech, 1 song) were exported as 2-channel files whose two
channels are bit-identical. The corpus was recorded through a single microphone, so these
are a dual-mono export artifact rather than real stereo. ``channel_format`` is therefore
taken from each file's own header instead of being hardcoded to ``mono``, which would have
mislabelled those six rows.

Labels live entirely in the filename: seven hyphen-separated two-digit codes,
``modality-vocal_channel-emotion-intensity-statement-repetition-actor`` (e.g.
``03-01-06-01-02-01-12.wav``), per the record's own "Filename identifiers" section. Odd
actor ids are male and even are female. Two combinations the codes allow but the corpus
never uses: neutral is only ever recorded at normal intensity, and song covers just
neutral/calm/happy/sad/angry/fearful, leaving disgust and surprised to speech. Emotion
becomes the row's ``class_list``; the rest become columns, with the actor id under
``speaker_id`` and the spoken statement under ``text`` to match the other speech loaders.

RAVDESS ships no train/test split. Each actor's own clips are divided across train/valid/eval
at ``split_ratios``, so all 24 actors appear in all three splits. Grouping is by actor alone
rather than by (actor, emotion): neutral is recorded at one intensity only, so an
(actor, emotion) group holds 8 clips for most emotions but just 4 for neutral, and
``split_counts`` guarantees every group at least one valid and one eval row -- a floor that
would pull an 80/10/10 request out to roughly 73/13/13. Keyed on the actor the groups are far
larger than that floor, so the requested ratios survive: at the default ``(0.8, 0.1)`` an
actor's 104 speech-plus-song clips divide 84/10/10, giving 1980/236/236 over the whole corpus
(60 clips for speech alone divide 48/6/6, and 44 for song alone divide 36/4/4). Pooling 24
actors also leaves every emotion well represented in every split.

This is deliberately a speaker-dependent split: an actor's second take of a statement can
land in train while the first lands in eval. Speaker-independent evaluation is the norm in the
emotion literature, but on 24 actors it buys an eval set of only two of them, whose scores say
as much about those two people as about the model. Consumers who do want speaker-independent
folds can regroup on the ``speaker_id`` column.

Every clip was rated 10 times for emotional validity by 247 human raters, so ``label_source``
is gold rather than merely acted intent.

License: CC BY-NC-SA 4.0 -- attribution, non-commercial, share-alike. Commercial use
requires a separate paid license from the authors (ravdess@gmail.com), so do not publish
this loader's output into anything commercial without clearing that first. Cite Livingstone
SR, Russo FA (2018), PLoS ONE 13(5): e0196391, https://doi.org/10.1371/journal.pone.0196391.

Run ``python -m audio_preprocessing.dataset.load_ravdess`` from the repo root to download the
speech archive and build the dataset.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets.features.features import Features, Value
from tqdm import tqdm

from audio_preprocessing.dataset._common import (
    build_audio_record,
    split_records_by_key,
    splits_to_audio_dataset,
    validate_choices,
)
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import DATA_FEATURES, AudioDataset, LabelSource

RECORD_ID = "1188976"
SOURCE_DATASET = "RAVDESS"

# Recorded in a professional studio (Livingstone & Russo 2018, Methods). RAVDESS labels no
# per-clip scene, so every row shares this one value rather than leaving it blank.
ENVIRONMENT = "studio"

# Modality code carried by every clip in the two audio-only archives; 01 (full-AV) and 02
# (video-only) only ever appear on the video archives' .mp4 files.
AUDIO_ONLY_MODALITY = "03"

VOCAL_CHANNEL_ARCHIVES = {
    "speech": "Audio_Speech_Actors_01-24",
    "song": "Audio_Song_Actors_01-24",
}

# Clip counts published in the record description. A mismatch is a cheap signal that an
# extraction is incomplete or that stray files landed in the tree.
OFFICIAL_CLIP_COUNTS = {"speech": 1440, "song": 1012}

VOCAL_CHANNELS = {"01": "speech", "02": "song"}
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITIES = {"01": "normal", "02": "strong"}
STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
REPETITIONS = {"01": 1, "02": 2}

ACTOR_IDS = tuple(f"{number:02d}" for number in range(1, 25))

FILENAME_RE = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion>\d{2})-(?P<intensity>\d{2})"
    r"-(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor>\d{2})$"
)

# ``speaker_id`` holds what RAVDESS itself calls the actor id, and ``text`` the statement the
# actor spoke. Both use the names the other speech loaders here already expose, so speaker-
# and transcript-aware code keeps working across datasets without a RAVDESS special case.
RAVDESS_FEATURES = Features({
    **DATA_FEATURES,
    "sample_id": Value("string"),
    "speaker_id": Value("string"),
    "gender": Value("string"),
    "vocal_channel": Value("string"),
    "intensity": Value("string"),
    "text": Value("string"),
    "repetition": Value("int64"),
})


def _gender(actor_id: str) -> str:
    """Male for odd actor ids and female for even, per the record's filename documentation."""
    return "male" if int(actor_id) % 2 else "female"


def _decode(table: dict[str, Any], code: str, field: str, path: Path) -> Any:
    try:
        return table[code]
    except KeyError:
        raise ValueError(f"Unknown RAVDESS {field} code {code!r} in {path}.") from None


def _parse_stem(path: Path) -> dict[str, Any]:
    """Decode one clip's filename into its labelled fields, including ``emotion``."""
    match = FILENAME_RE.match(path.stem)
    if match is None:
        raise ValueError(
            f"Unexpected RAVDESS filename {path.name!r} under {path.parent}; expected seven "
            "hyphen-separated two-digit codes."
        )

    codes = match.groupdict()
    if codes["modality"] != AUDIO_ONLY_MODALITY:
        raise ValueError(
            f"{path.name!r} carries modality {codes['modality']!r} rather than audio-only "
            f"({AUDIO_ONLY_MODALITY}); this loader reads only the two Audio_* archives."
        )

    actor_id = codes["actor"]
    if actor_id not in ACTOR_IDS:
        raise ValueError(f"Unknown RAVDESS actor id {actor_id!r} in {path}.")

    return {
        "sample_id": path.stem,
        "speaker_id": actor_id,
        "gender": _gender(actor_id),
        "vocal_channel": _decode(VOCAL_CHANNELS, codes["vocal_channel"], "vocal channel", path),
        "emotion": _decode(EMOTIONS, codes["emotion"], "emotion", path),
        "intensity": _decode(INTENSITIES, codes["intensity"], "emotional intensity", path),
        "text": _decode(STATEMENTS, codes["statement"], "statement", path),
        "repetition": _decode(REPETITIONS, codes["repetition"], "repetition", path),
    }


def scan(root: Path, vocal_channels: Iterable[str]) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``(wav_path, fields)`` for every requested clip beneath ``root``.

    Prefers each archive's own extraction directory, which is what ``download_zenodo``
    produces, and falls back to scanning ``root`` itself for a manually flattened
    extraction. Either way the filename's vocal-channel code decides which corpus a clip
    belongs to, so the two corpora stay separable even when unpacked into one directory.

    Raises:
        FileNotFoundError: If a requested vocal channel has no clips under ``root``.
    """
    wanted = set(vocal_channels)
    search_dirs = [
        root / VOCAL_CHANNEL_ARCHIVES[channel]
        for channel in wanted
        if (root / VOCAL_CHANNEL_ARCHIVES[channel]).is_dir()
    ]
    if not search_dirs:
        search_dirs = [root]

    found = []
    for search_dir in sorted(search_dirs):
        for path in sorted(search_dir.rglob("*.wav")):
            # Some unzip tools drop an AppleDouble resource fork beside each member. They
            # are not audio and never parse as a RAVDESS filename.
            if path.name.startswith("._"):
                continue
            fields = _parse_stem(path)
            if fields["vocal_channel"] in wanted:
                found.append((path, fields))

    counts = Counter(fields["vocal_channel"] for _, fields in found)
    for channel in sorted(wanted):
        if not counts[channel]:
            raise FileNotFoundError(
                f"No RAVDESS {channel} clips under {root}; expected "
                f"{VOCAL_CHANNEL_ARCHIVES[channel]}/Actor_NN/*.wav. Use prepare=True to "
                "download and extract the archive."
            )
        if counts[channel] != OFFICIAL_CLIP_COUNTS[channel]:
            warnings.warn(
                f"Found {counts[channel]} RAVDESS {channel} clips under {root}, but the "
                f"record publishes {OFFICIAL_CLIP_COUNTS[channel]}; the extraction may be "
                "incomplete.",
                stacklevel=2,
            )

    return found


def _split_key(record: dict[str, Any]) -> str:
    """Group clips by actor, so each actor contributes to all three splits."""
    return record["speaker_id"]


def _records(found: list[tuple[Path, dict[str, Any]]]):
    """Yield one canonical record per clip, reading each clip's audio header."""
    for path, fields in tqdm(found, desc=f"{SOURCE_DATASET} clips"):
        emotion = fields.pop("emotion")
        record = build_audio_record(
            path,
            emotion,
            source_dataset=SOURCE_DATASET,
            environment=ENVIRONMENT,
        )
        record.update(fields)
        yield record


class RAVDESSLoader(ZenodoLoader):
    """Prepare and build the RAVDESS emotional speech and song corpus.

    The record's 47 video archives (~24 GB) are unreachable through this loader: the Zenodo
    ``only`` filter is derived from ``vocal_channels`` instead of being accepted from the
    caller, so no configuration of it can start a video download.
    """

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        vocal_channels: Iterable[str] = ("speech", "song"),
        split_ratios: list[float] | tuple[float, float] = (0.8, 0.1),
        seed: int = 42,
        prepare: bool = True,
    ) -> None:
        channels = validate_choices(
            vocal_channels,
            name="vocal channel",
            context="RAVDESSLoader",
            allowed=VOCAL_CHANNEL_ARCHIVES,
        )
        super().__init__(
            root=root,
            only=[VOCAL_CHANNEL_ARCHIVES[channel] for channel in channels],
            prepare=prepare,
        )
        self.vocal_channels = channels
        self.split_ratios = split_ratios
        self.seed = seed

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "RAVDESSLoader needs a root path. Use prepare=False with an existing local "
                "extraction."
            )

        found = scan(Path(self.root), self.vocal_channels)
        split_records = split_records_by_key(
            _records(found), _split_key, self.split_ratios, self.seed
        )
        for split, rows in split_records.items():
            for record in rows:
                record["split"] = split
        return splits_to_audio_dataset(
            split_records,
            features=RAVDESS_FEATURES,
            label_source=self.label_source,
        )


def main():
    # Speech alone (~199 MiB) is the cheapest build that still covers all 8 emotions; pass
    # vocal_channels=("speech", "song") for the full ~414 MiB audio corpus.
    return RAVDESSLoader(vocal_channels=("speech",))()


if __name__ == "__main__":
    ravdess = main()
    print(ravdess.info())
