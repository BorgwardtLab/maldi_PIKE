# maldi_PIKE &mdash; Peak Information KErnel for MALDI-TOF MS spectra

`maldi_PIKE` is a small library for Python 3 that include the code used for 
'Topological and kernel-based microbial phenotype prediction from MALDI-TOF mass 
spectra'. The main method includes PIKE, the Peak Information KErnel for MALDI-TOF MS spectra, 
embedded in a Gaussian Process. We developed PIKE based on heat diffusion on structured 
objects. It is well suited for MALDI-TOF mass spectra and able to capture interactions between
mass peaks.

# Dependencies

- Python 3.7
- packages listed in `pyproject.toml`

# Installation

- Clone the repository
- `poetry install`

Follow the instructions given by `poetry`.

# Example behaviour of PIKE

Figure 2:

![PIKE_behaviour](PIKE_behaviour.png)

Code to recreate this graphic in matplotlib can be found in
`ismb2020_maldi/visualise_feature_map.py`. Use the script as
follows:

```
poetry run python ismb2020_maldi/visualise_feature_map.py data/Example_peaks.txt
```

This should result in the following plot:

![PIKE_behaviour_matplotlib](PIKE_behaviour_matplotlib.png)

This repository is work in progress.
