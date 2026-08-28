# EpiAudio preprocessing

Minimal public preprocessing code required to reproduce the EpiAudio sweeps.
It contains the `AudioDataset` container and only loaders whose datasets were
observed in the `epi-audio` Comet workspace. Transformation, realization,
provenance, and publishing utilities remain intentionally out of scope.

Included loaders cover AISHELL-1/3, AVSpeech, BAT, Clotho-AQA, CochlScene,
Common Voice, DataSED, DEMAND, EigenScape, ESD, Fake-or-Real, FLEURS, FMA,
LibriTTS, MELD, MLS, MultiVox, NonSpeech7k, RAVDESS, SONYC-UST, Spatial
LibriSpeech, TAU-NIGENS21, TAU Urban 2022, ToyADMOS, TUT 2016/2017,
UrbanSound8K, VGGSound, VocalSound, and VoxPopuli.

```python
from audio_preprocessing.dataset.load_urbansound import UrbanSoundLoader

dataset = UrbanSoundLoader()()
print(dataset.info())
```

Loaders download data into `data/zenodo_<record id>` by default. Pass
`root=...` and `prepare=False` to build metadata from an existing download.
