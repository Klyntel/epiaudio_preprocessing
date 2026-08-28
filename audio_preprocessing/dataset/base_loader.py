"""Minimal loader lifecycle shared by the public sweep datasets."""

from abc import ABC, abstractmethod
from pathlib import Path

from audio_preprocessing.dataset.zenodo_downloader import download_zenodo
from audio_preprocessing.datasets import AudioDataset, LabelSource


class BaseLoader(ABC):
    label_source: LabelSource | str = LabelSource.UNKNOWN

    def __init__(self, root=None, *, prepare: bool = True) -> None:
        self.root = Path(root) if root is not None else None
        self.should_prepare = prepare

    def prepare_raw(self) -> None:
        pass

    @abstractmethod
    def build_dataset(self) -> AudioDataset:
        pass

    def __call__(self) -> AudioDataset:
        if self.should_prepare:
            self.prepare_raw()
        return self.build_dataset()


class ZenodoLoader(BaseLoader):
    record_id: str

    def __init__(
        self,
        root=None,
        *,
        only: list[str] | None = None,
        do_not_download: list[str] | None = None,
        prepare: bool = True,
    ) -> None:
        super().__init__(root, prepare=prepare)
        self.only = only
        self.do_not_download = do_not_download

    def prepare_raw(self) -> None:
        self.root = download_zenodo(
            self.record_id,
            output_dir=str(self.root) if self.root is not None else "",
            only=self.only,
            do_not_download=self.do_not_download,
        )


__all__ = ["AudioDataset", "BaseLoader", "ZenodoLoader"]

