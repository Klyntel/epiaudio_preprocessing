import subprocess
import ast
import zipfile
import warnings
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
from datasets.features.features import Features, List, Value
from audio_preprocessing.dataset.base_loader import BaseLoader
from audio_preprocessing.datasets import AudioDataset, LabelSource
from audio_preprocessing.dataset._common import build_audio_record, splits_to_audio_dataset
from audio_preprocessing.datasets import DATA_FEATURES


def dtype_to_feature(dtype: Any) -> Value | List | None:
    if isinstance(dtype, np.dtypes.Int64DType):
        return Value("int64")
    if isinstance(dtype, np.dtypes.Float64DType):
        return Value("float64")
    if isinstance(dtype, pd.StringDtype):
        return Value("string")
    if isinstance(dtype, np.dtypes.ObjectDType):
        return List(Value("string"))

    raise TypeError(f"Unsupported dtype: {dtype!r}")


def get_features(dtypes: pd.Series) -> Features:
    feature_map = dtypes.apply(dtype_to_feature)
    features = Features({**DATA_FEATURES, **feature_map.to_dict()})

    return features


class FMALoader(BaseLoader):
    data_base_url: str = "https://os.unil.cloud.switch.ch/fma"
    metadata_file_name: Path = Path("fma_metadata.zip")
    subsets = ["small", "medium", "large", "full"]
    split_conversion = {"training": "train", "validation": "valid", "test": "eval"}
    label_source = LabelSource.GOLD

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        prepare: bool = True,
        subset: str = "small",
        **config: Any,
    ) -> None:
        super().__init__(root=root, prepare=prepare, **config)

        if subset not in self.subsets:
            raise ValueError(f"Invalid subset ({subset}). subset must be in {self.subsets}")

        self.subset = subset
        if isinstance(self.root, str):
            self.root = Path(self.root)
        elif self.root is None:
            self.root = Path.cwd()

        self.root.mkdir(parents=True, exist_ok=True)

    def _download(self, file_name: str) -> None:
        url = f"{self.data_base_url}/{file_name}"
        command = [
            "curl",
            "--fail",
            "--output-dir",
            str(self.root),
            "-O",
            url
        ]
        subprocess.run(command, check=True)

        download_path = self.root / file_name
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        download_path.unlink()

    def _find_mp3_files(self) -> pd.Series:
        input_dir = self.root / f"fma_{self.subset}"
        mp3_paths = sorted(input_dir.rglob("*.mp3"))
        series = pd.Series(mp3_paths, index=[path.stem for path in mp3_paths])

        return series

    def _get_genre_titles(self, string: str, genre_titles: pd.Series) -> list[str]:
        """
        Each track has lists of genres which are literal strings.
        This converts such a string into a list of genres.
        """
        try:
            genre_idx_list = ast.literal_eval(string)
        except Exception:
            msg = f"Genre id list {string} could not be parsed as a list."
            warnings.warn(msg)
            genre_idx_list = []
        genre_list = []
        for idx in genre_idx_list:
            try:
                genre_list.append(genre_titles.loc[idx])
            except Exception:
                msg = f"Genre id {idx} could not be converted to a genre title, skipping."
                warnings.warn(msg)

        return genre_list

    def _get_root_genres(self, genres: list[str], root_genre_map: dict[str, str]) -> list[str]:
        """
        Each track has lists of genres which are literal strings.
        This gets the titles of the corresponding root genres.
        """
        top_genres = set()
        for genre in genres:
            try:
                top_genres.add(root_genre_map[genre])
            except Exception:
                msg = f"Root genre of genre {genre} could not be found, skipping."
                warnings.warn(msg)

        return sorted(list(top_genres))

    def _process_metadata(self) -> pd.DataFrame:
        """
        The metadata for each audio track is stored in tracks.csv. This file uses ids
        for genres rather than genre names, which are found in genres.csv. This function
        produces a single data frame giving all track data (with genre names) for each
        track in the subset of interest.
        """
        base_path = self.root / self.metadata_file_name.stem

        track_data_path = base_path / "tracks.csv"
        all_track_data = pd.read_csv(str(track_data_path), index_col=0, header=[0, 1])
        prefixes = {
            "track": "track_",
            "album": "albums_",
            "artist": "artist_",
            "set": ""
        }
        tracks = all_track_data[list(prefixes)]
        tracks.columns = [f"{prefixes[top]}{sub}" for top, sub in tracks.columns]
        subset_index = self.subsets.index(self.subset)
        include = pd.Series(
            index=tracks.index,
            data=[self.subsets.index(ss) <= subset_index for ss in tracks["subset"]]
        )
        tracks = tracks.loc[include].drop("subset", axis=1)

        genre_data_path = base_path / "genres.csv"
        genre_df = pd.read_csv(genre_data_path, index_col=0)
        genre_titles = pd.Series(genre_df["title"])
        root_genres = genre_df["top_level"]
        root_genre_map = dict()
        for idx, name in genre_titles.items():
            root_genre_idx = root_genres.loc[idx]
            root_genre_map[name] = genre_titles.loc[root_genre_idx]

        for col in ["track_genres", "track_genres_all"]:
            column = pd.Series(tracks[col])
            tracks[col] = column.apply(lambda x: self._get_genre_titles(x, genre_titles))

        tracks["track_genres_top"] = tracks["track_genres_all"].apply(lambda x: self._get_root_genres(x, root_genre_map))
        tracks = tracks.drop("track_genre_top", axis=1)

        audio_paths = self._find_mp3_files()
        audio_paths.index = audio_paths.index.astype("int64")
        tracks["audio_path"] = audio_paths

        return pd.DataFrame(tracks)

    def prepare_raw(self) -> None:
        self._download(str(self.metadata_file_name))
        self._download(f"fma_{self.subset}.zip")

    def build_dataset(self) -> AudioDataset:
        track_df = self._process_metadata()
        rows = track_df.to_dict("records")
        splits = {split: [] for split in self.split_conversion.values()}

        for row in rows:
            split = self.split_conversion.get(row["split"], None)
            if split is None:
                warnings.warn(f"Split not found for {row["audio_path"]}, skipping")
                continue

            audio_path = row["audio_path"]
            if pd.isna(audio_path):
                warnings.warn("No audio file found for an FMA metadata row.", stacklevel=2)
                continue
            try:
                record: dict[str, Any] = build_audio_record(
                    str(audio_path),
                    row["track_genres_top"],
                    split=split,
                    source_dataset=f"FMA_{self.subset}",
                    metadata_path="fma_metadata/tracks.csv",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Could not read FMA audio file {audio_path}: {exc}",
                    stacklevel=2,
                )
                continue

            metadata: dict[str, Any] = {
                str(key): value
                for key, value in row.items()
                if key not in {"audio_path", "split"}
            }
            record.update(metadata)
            splits[split].append(record)

        dtypes = track_df.dtypes.drop(["audio_path", "split"])
        features = get_features(dtypes)
        audio_dataset = splits_to_audio_dataset(
            splits,
            features=features,
            label_source=self.label_source
        )

        return audio_dataset


def main():
    return FMALoader()()


if __name__ == "__main__":
    fma = main()
    print(fma.info())
