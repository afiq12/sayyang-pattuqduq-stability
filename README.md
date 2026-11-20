# Sayyang pattuqduq: parametric stability of the dancing-horse rider

Code and derived data for the article

> A. Agung, *From the Kapitza pendulum to Hill's equation: parametric stability under non-sinusoidal drive, and why the Mandar dancing-horse rider stays upright* (submitted to the American Journal of Physics).

The rider's trunk is modelled as an inverted pendulum with hip stiffness and damping on a vertically oscillating saddle. The scripts compute Floquet stability maps for the Mathieu and Hill equations, check the analytical tongue thresholds, and measure the rider's head motion and the drum tempo in public festival videos.

## Contents

```
theory/    Floquet, Hill-tongue, second-order, Hill-determinant and square-wave computations; Figs. 1 to 3
video/     face tracking, phase correlation, camera stabilization, injection tests, tempo; Fig. 5
data/      derived data (CSV and JSON), CC BY 4.0
figures/   figures as used in the article
videos/    list of the analysed YouTube videos (the videos themselves are not included)
```

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

The theory scripts need only NumPy, SciPy, Matplotlib and SymPy. The video scripts also need OpenCV with the contrib modules (for the CSRT tracker).

## Reproducing the results

Run the scripts from the repository root.

### Theory (no videos needed)

| Article | Command |
|---|---|
| Sec. 2.1, Lagrangian forms | `python theory/lagrangian_check.py` |
| Figs. 1 and 2 | `python theory/fig1_fig2.py` |
| Fig. 3 | `python theory/fig3_kappa_b.py` |
| Table 2 (minimum stiffness) | `python theory/literature_floquet_check.py kmin` |
| Sec. 3.3, first-order tongues | `python theory/first_order_tongues.py` |
| Sec. 3.3, second-order correction | `python theory/second_order_halfwidth.py`, `python theory/second_order_threshold.py` |
| Sec. 3.3, Hill determinant | `python theory/hill_determinant.py` |
| Sec. 3.3, square-wave benchmark | `python theory/square_wave_benchmark.py` |
| Sec. 3.4, Floquet check with literature parameters | `python theory/literature_floquet_check.py lit` |

`theory/floquet_core.py` contains the monodromy-matrix integrator and the impulsive stamping waveform used throughout.

### Video analysis

Download the videos listed in `videos/videos.csv` and save them as described in `videos/README.md`. Then:

| Article | Command |
|---|---|
| Shot A face track | `python video/shotA_track.py` |
| Shot A, Table 4 (injection test, phase correlation) | `python video/shotA_phasecorr_injection.py AMP_CM FREQ_HZ TAG` (e.g. `0 2.5 zero`, `0.05 2.5 a05`) |
| Shot A, Table 4 (CSRT branch) | `python video/shotA_csrt_injection.py AMP_CM FREQ_HZ TAG` |
| Shots B to G, measurement and injection | `python video/rider_phasecorr.py VIDEO T0 T1 X,Y,W,H INJECT_CM 2.9 TAG` (see below) |
| Table 5 summary | `python video/rider_summary.py` |
| Fig. 5 | `python video/fig5_spectra.py` |
| Table 6 (rebana tempo) | `python video/tempo_v2.py videos/V2_audio.wav` |
| Finding uncut shots | `python video/find_shots.py VIDEO` |

Shots B to G (run the `0` measurement first, then the `0.1` injection test, which reuses the face track):

| Shot | Tag | Video | T0 | T1 | Initial face box |
|---|---|---|---|---|---|
| B | `b076` | V1 | 76.0 | 84.6 | 670,210,170,190 |
| C | `b134` | V1 | 134.3 | 149.1 | 760,130,140,170 |
| D | `b282` | V1 | 282.5 | 319.3 | 1025,360,95,130 |
| E | `p787` | V2 | 787.6 | 800.9 | 560,110,95,110 |
| F | `p833` | V2 | 832.8 | 840.6 | 575,60,140,140 |
| G | `p1105` | V2 | 1105.3 | 1113.8 | 582,75,72,110 |

Example: `python video/rider_phasecorr.py videos/V1_1080p.mp4 76.0 84.6 670,210,170,190 0 2.9 b076`, then the same with `0.1` and tag `b076i`.

## Data

| File | Content |
|---|---|
| `data/shotA/face_track.csv` | CSRT face box of shot A (centre and height in pixels) |
| `data/shotA/head_phasecorr.csv` | shot A vertical head motion: stabilized, raw face and background (cm) |
| `data/shotA/inj_*.json` | shot A injection-test results |
| `data/riders/head_TAG.csv` | head motion of shots B to G (cm, high-passed at 0.8 Hz); `TAGi` = run with 1 mm injected at 2.9 Hz |
| `data/riders/facebox_TAG.csv` | face box track used to place the correlation window |
| `data/riders/res_TAG.json`, `summary.json` | per-shot results and the summary behind Table 5 |
| `data/tempo/tempo_v2_windows.txt` | tempo windows behind Table 6 |

Scale: the face height is taken as 20 cm. No video frames or images of faces are included.

## License

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0 (see `data/LICENSE`). The analysed videos remain the property of their creators.

## Citation

See `CITATION.cff`. Please cite the article and the Zenodo archive of this repository.
