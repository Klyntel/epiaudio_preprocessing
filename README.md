# EpiAudio preprocessing

Minimal public preprocessing code required to reproduce the EpiAudio sweeps.
It contains the `AudioDataset` container and the DEMAND and UrbanSound8K
loaders; the private project's other loaders and transformation utilities are
intentionally out of scope.

```python
from audio_preprocessing.dataset.load_urbansound import UrbanSoundLoader

dataset = UrbanSoundLoader()()
print(dataset.info())
```

Loaders download data into `data/zenodo_<record id>` by default. Pass
`root=...` and `prepare=False` to build metadata from an existing download.

