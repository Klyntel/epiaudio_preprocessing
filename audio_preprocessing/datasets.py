"""Dataset metadata types and the small container interface consumed by EpiAudio."""

import math
import warnings
from enum import StrEnum
from typing import Any, TypedDict

from datasets import DatasetDict
from datasets.features.features import Features, Json, List, Value

from audio_preprocessing.utils import is_missing, to_non_negative_seconds

_EPSILON = 1e-9


class EventDict(TypedDict):
    label: str
    event_type: str
    offset: float
    duration: float
    additional_metadata: dict[str, Any]
    frames_per_sec: int | None
    frame_array: list["ContinuousDataDict"]


class ContinuousDataDict(TypedDict, total=False):
    # See audio_preprocessing/dataset/README.md#spatial-conventions.
    azimuth: float | None
    elevation: float | None
    distance: float | None  # Meters.
    label: str | None


class DataDict(TypedDict):
    audio_path: str
    sample_rate: int
    clip_offset: float
    clip_duration: float
    class_list: list[str]  # Weakly labeled data, outside the strongly labeled events.
    split: str
    source_dataset: str  # Either explicitly set or a random UUID hex string.
    metadata_path: str
    events: list[EventDict]
    num_channels: int
    channel_format: str  # See dataset/README.md#spatial-conventions for "foa".
    environment: str


Data = DataDict


class EventType(StrEnum):
    EVENT = "event"
    CONTINUOUS_EVENT = "continuous_event"


class LabelSource(StrEnum):
    """Dataset-level provenance for the labels attached to a dataset."""

    GOLD = "gold"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ContinuousData:
    """Optional per-frame spatial metadata.

    See ``audio_preprocessing/dataset/README.md#spatial-conventions`` for the
    canonical angle and distance conventions.
    """

    def __init__(self, azimuth=None, elevation=None, distance=None, label=None):
        self.azimuth = self._optional_float("azimuth", azimuth)
        self.elevation = self._optional_float("elevation", elevation)
        self.distance = self._optional_float("distance", distance)
        if label is not None and not isinstance(label, str):
            raise TypeError(f"ContinuousData label must be a string or None, got {type(label).__name__}")
        self.label = label

    @classmethod
    def from_dict(cls, data):
        return cls(
            azimuth=data.get("azimuth"),
            elevation=data.get("elevation"),
            distance=data.get("distance"),
            label=data.get("label"),
        )

    def to_dict(self):
        return {
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "label": self.label,
        }

    @staticmethod
    def _optional_float(name, value):
        if is_missing(value):
            return None
        if isinstance(value, bool):
            raise TypeError(f"ContinuousData {name} must be a number or None, got bool")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(f"ContinuousData {name} must be a number or None, got {type(value).__name__}") from None


CONTINUOUS_DATA_FEATURE = {
    "azimuth": Value("float64"),
    "elevation": Value("float64"),
    "distance": Value("float64"),
    "label": Value("string"),
}

EVENT_FEATURE = List({
    "label": Value("string"),
    "event_type": Value("string"),
    "offset": Value("float64"),
    "duration": Value("float64"),
    "additional_metadata": Json(),
    "frames_per_sec": Value("int64"),
    "frame_array": List(CONTINUOUS_DATA_FEATURE),
})

DATA_FEATURES = Features({
    "audio_path": Value("string"),
    "sample_rate": Value("int64"),
    "clip_offset": Value("float64"),
    "clip_duration": Value("float64"),
    "class_list": List(Value("string")),
    "split": Value("string"),
    "source_dataset": Value("string"),
    "metadata_path": Value("string"),
    "events": EVENT_FEATURE,
    "num_channels": Value("int64"),
    "channel_format": Value("string"),
    "environment": Value("string"),
})


class Event:
    """Strongly labeled event metadata for one audio span."""

    def __init__(self, label, offset, duration, additional_metadata=None, event_type: EventType | str = EventType.EVENT):
        if not isinstance(label, str):
            raise TypeError(f"Event label must be a string, got {type(label).__name__}")

        self.label = label
        self.event_type = self._validate_event_type(event_type)
        if type(self) is Event and self.event_type == EventType.CONTINUOUS_EVENT:
            raise ValueError("Use ContinuousEvent for event_type='continuous_event'.")
        self.offset = self._validate_seconds("offset", offset)
        self.duration = self._validate_seconds("duration", duration)
        self.additional_metadata = {} if additional_metadata is None else dict(additional_metadata)

    @classmethod
    def build_start_end(cls, start, end, label, additional_metadata=None, event_type: EventType | str = EventType.EVENT):
        start = cls._validate_seconds("start", start)
        end = cls._validate_seconds("end", end)
        if end < start:
            raise ValueError(f"end must be greater than or equal to start, got start={start}, end={end}")
        return cls(
            label=label,
            offset=start,
            duration=end - start,
            additional_metadata=additional_metadata,
            event_type=event_type,
        )

    @classmethod
    def from_dict(cls, data):
        event_type = cls._validate_event_type(data.get("event_type", EventType.EVENT))
        if event_type == EventType.CONTINUOUS_EVENT:
            return ContinuousEvent.from_dict(data)

        return cls(
            label=data["label"],
            offset=data["offset"],
            duration=data["duration"],
            additional_metadata=data.get("additional_metadata"),
            event_type=event_type,
        )

    def to_dict(self):
        return {
            "label": self.label,
            "event_type": self.event_type.value,
            "offset": self.offset,
            "duration": self.duration,
            "additional_metadata": dict(self.additional_metadata),
            "frames_per_sec": None,
            "frame_array": [],
        }

    def get_start_end(self):
        return self.offset, self.offset + self.duration

    def slice_event(self, window_offset, window_duration):
        window_offset = self._validate_seconds("window_offset", window_offset)
        window_duration = self._validate_seconds("window_duration", window_duration)
        window_end = window_offset + window_duration
        event_start, event_end = self.get_start_end()

        clipped_start = max(event_start, window_offset)
        clipped_end = min(event_end, window_end)
        if clipped_end <= clipped_start:
            return None

        return Event(
            label=self.label,
            offset=clipped_start,
            duration=clipped_end - clipped_start,
            additional_metadata=self.additional_metadata,
        )

    @staticmethod
    def _validate_event_type(value):
        try:
            return EventType(value)
        except ValueError:
            allowed = ", ".join(item.value for item in EventType)
            raise ValueError(f"event_type must be one of {allowed}, got {value!r}") from None

    @staticmethod
    def _validate_seconds(name, value):
        seconds = to_non_negative_seconds(value)
        if is_missing(seconds):
            raise ValueError(f"{name} must be a non-negative seconds value, got {value!r}")
        return float(seconds)


class ContinuousEvent(Event):
    """Strongly labeled event with optional per-frame continuous metadata."""

    def __init__(self, label, offset, duration, frame_array, frames_per_sec, additional_metadata=None):
        super().__init__(
            label=label,
            offset=offset,
            duration=duration,
            additional_metadata=additional_metadata,
            event_type=EventType.CONTINUOUS_EVENT,
        )
        self.frames_per_sec = self._validate_frames_per_sec(frames_per_sec)
        self.frame_array = [
            frame if isinstance(frame, ContinuousData) else ContinuousData.from_dict(frame)
            for frame in frame_array
        ]

    @classmethod
    def build_start_end(
        cls,
        start,
        end,
        label,
        additional_metadata=None,
        event_type: EventType | str = EventType.CONTINUOUS_EVENT,
        *,
        frame_array=None,
        frames_per_sec=None,
    ):
        start = cls._validate_seconds("start", start)
        end = cls._validate_seconds("end", end)
        if end < start:
            raise ValueError(f"end must be greater than or equal to start, got start={start}, end={end}")
        if cls._validate_event_type(event_type) != EventType.CONTINUOUS_EVENT:
            raise ValueError("ContinuousEvent.build_start_end requires event_type='continuous_event'.")
        if frames_per_sec is None:
            raise ValueError("frames_per_sec is required for ContinuousEvent.build_start_end.")
        return cls(
            label=label,
            offset=start,
            duration=end - start,
            frame_array=[] if frame_array is None else frame_array,
            frames_per_sec=frames_per_sec,
            additional_metadata=additional_metadata,
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            label=data["label"],
            offset=data["offset"],
            duration=data["duration"],
            additional_metadata=data.get("additional_metadata"),
            frames_per_sec=data["frames_per_sec"],
            frame_array=data.get("frame_array") or [],
        )

    def to_dict(self):
        data = super().to_dict()
        data["frames_per_sec"] = self.frames_per_sec
        data["frame_array"] = [frame.to_dict() for frame in self.frame_array]
        return data

    def slice_event(self, window_offset, window_duration):
        basic_event = super().slice_event(window_offset, window_duration)
        if basic_event is None:
            return None

        event_first_frame, event_last_frame, event_frame_count = self._event_frame_span()
        actual_frame_count = len(self.frame_array)
        if actual_frame_count != event_frame_count:
            raise ValueError(
                "frame_array length does not match event duration at frames_per_sec: "
                f"expected {event_frame_count}, got {actual_frame_count}"
            )

        window_first_frame, window_last_frame = self._window_frame_span(*basic_event.get_start_end())
        overlap_first_frame = max(window_first_frame, event_first_frame)
        overlap_last_frame = min(window_last_frame, event_last_frame)
        if overlap_first_frame > overlap_last_frame:
            return None

        start_index = overlap_first_frame - event_first_frame
        stop_index = overlap_last_frame - event_first_frame + 1
        return type(self)(
            label=basic_event.label,
            offset=basic_event.offset,
            duration=basic_event.duration,
            frame_array=[frame.to_dict() for frame in self.frame_array[start_index:stop_index]],
            frames_per_sec=self.frames_per_sec,
            additional_metadata=basic_event.additional_metadata,
        )

    def _event_frame_span(self):
        frame_count = max(0, math.ceil(self.duration * self.frames_per_sec - _EPSILON))
        return 0, frame_count - 1, frame_count

    def _window_frame_span(self, window_offset, window_end):
        # frame_array is event-local: frame 0 starts at self.offset, not at a global time-grid.
        window_start_frame = (window_offset - self.offset) * self.frames_per_sec
        window_end_frame = (window_end - self.offset) * self.frames_per_sec
        window_off_grid = (
            abs(window_start_frame - round(window_start_frame)) > _EPSILON
            or abs(window_end_frame - round(window_end_frame)) > _EPSILON
        )
        if window_off_grid:
            warnings.warn(
                "continuous event window is not frame-aligned; including all overlapping event-label frames",
                stacklevel=3,
            )

        return math.floor(window_start_frame + _EPSILON), math.ceil(window_end_frame - _EPSILON) - 1

    @staticmethod
    def _validate_frames_per_sec(value):
        if isinstance(value, bool):
            raise TypeError("frames_per_sec must be a positive integer, got bool")
        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, int):
            raise TypeError(f"frames_per_sec must be a positive integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"frames_per_sec must be positive, got {value}")
        return value

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
        **metadata: Any,
    ) -> None:
        self.data = data
        self.label_source = LabelSource(label_source).value
        self.split = split
        self.__dict__.update(metadata)

    def __len__(self) -> int:
        return len(self.data[self.split])

    def __getitem__(self, index: int):
        return self.data[self.split][index]

    def info(self) -> dict[str, Any]:
        return {
            "splits": {name: len(rows) for name, rows in self.data.items()},
            "label_source": self.label_source,
        }
