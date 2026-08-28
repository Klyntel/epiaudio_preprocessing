"""The small dataset interface consumed by EpiAudio."""

from enum import StrEnum

from datasets import DatasetDict


class LabelSource(StrEnum):
    """Origin of a dataset's labels."""

    GOLD = "gold"
    UNKNOWN = "unknown"


class Event:
    """Strong label attached to a time span."""

    def __init__(self, label: str, offset: float, duration: float):
        self.label = label
        self.offset = float(offset)
        self.duration = float(duration)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "event_type": "event",
            "offset": self.offset,
            "duration": self.duration,
            "additional_metadata": {},
            "frames_per_sec": None,
            "frame_array": [],
        }


class AudioDataset:
    """Container for named Hugging Face Dataset splits of audio metadata.

    EpiAudio consumes the public ``data`` and ``label_source`` attributes.
    Dataset transformation and publishing methods from the internal package are
    deliberately not part of this reproduction interface.
    """

    def __init__(
        self,
        *,
        data: DatasetDict,
        label_source: LabelSource | str = LabelSource.UNKNOWN,
        split: str = "train",
    ) -> None:
        self.data = data
        self.label_source = LabelSource(label_source).value
        self.split = split

    def __len__(self) -> int:
        return len(self.data[self.split])

    def __getitem__(self, index: int):
        return self.data[self.split][index]

    def info(self) -> dict:
        return {
            "splits": {name: len(rows) for name, rows in self.data.items()},
            "label_source": self.label_source,
        }

