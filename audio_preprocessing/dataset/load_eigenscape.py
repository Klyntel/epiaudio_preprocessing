"""Load the EigenScape Lite dataset into an AudioDataset.

The Lite zip extracts to a single folder of flat files named ``<Scene>.<N>.flac``
(e.g. ``Beach.1.flac``), so the label is the scene (the part before the first dot).

Run ``python -m audio_preprocessing.dataset.load_eigenscape`` from the repo root to download the Lite zip
and build the dataset.

Note: EigenScape cannot be subset at download time. The Lite variant is a single ~12.6 GB zip
holding all 64 recordings (the per-scene zips are 12-15 GB each), so we download it once and
read only a couple of scenes below via the ``scenes`` filter.
"""

from pathlib import Path

from audio_preprocessing.dataset._common import build_audio_dataset
from audio_preprocessing.dataset.base_loader import ZenodoLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource

RECORD_ID = "1012809"


def collect(input_path, scenes=None):
    """Yield ``(flac_path, scene_label)`` for every recording under ``input_path``.

    ``scenes`` optionally limits which scenes are read (e.g. ``["Beach", "Woodland"]``);
    ``None`` reads all.
    """
    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.iterdir():
            if not file.is_file() or file.suffix != ".flac":
                continue
            scene = file.name.split(".")[0]
            if scenes is None or scene in scenes:
                yield file, scene


class EigenScapeLoader(ZenodoLoader):
    """Prepare and build EigenScape, optionally filtering scene labels."""

    record_id = RECORD_ID
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        split_ratios: list[float] | tuple[float, float] = (0.8, 0.1),
        seed: int = 42,
        scenes: list[str] | None = None,
        only: list[str] | None = None,
        do_not_download: list[str] | None = None,
        prepare: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            only=only,
            do_not_download=do_not_download,
            prepare=prepare,
        )
        self.split_ratios = split_ratios
        self.seed = seed
        self.scenes = scenes

    def build_dataset(self) -> AudioDataset:
        if self.root is None:
            raise ValueError(
                "EigenScapeLoader needs a root path. Use prepare=False with an existing local root."
            )

        return build_audio_dataset(
            collect(self.root, self.scenes),
            self.split_ratios,
            self.seed,
            source_dataset="EigenScape",
            channel_format="foa",
            label_source=self.label_source,
        )


def main():
    # Full Lite dataset: EigenScapeLoader(only=["Lite-EigenScape", "Metadata-EigenScape"])()
    # Small subset for testing transforms: two scenes only.
    return EigenScapeLoader(
        only=["Lite-EigenScape", "Metadata-EigenScape"],
        scenes=["Beach", "Woodland"],
    )()


if __name__ == "__main__":
    eigenscape = main()
    print(eigenscape.info())
