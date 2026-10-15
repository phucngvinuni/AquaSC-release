# Environments (two separate interpreters, do not merge)

| Env | File | Used for |
|---|---|---|
| torch env (conda, Python 3.10, torch 2.1.2 + CUDA 12.1) | `torch-env.yml` | training, encoder/decoder, detector, statistics, plotting, WebP/JPEG decoding |
| Sionna venv (CPU is enough) | `sionna-venv.txt` (versions of the venv that produced every result: Sionna 1.2.2, TensorFlow 2.21.0, NumPy 2.5.3, SciPy 1.18.1) | 5G-NR LDPC BG2, QAM/QPSK, Rayleigh channel, demapper, decoder (`phy.py`, `digital_pipeline.py`) |

```bash
conda env create -f requirements/torch-env.yml                      # env name: siso-project-env
python -m venv venv-sionna && venv-sionna/bin/pip install -r requirements/sionna-venv.txt
```
TensorFlow 2.21 needs NumPy >= 1.26, torch 2.1.2 needs NumPy < 1.26: this is why the LDPC chain lives in its own venv and the torch
process talks to it through a JSON-lines subprocess bridge (`digital_client.py` / `digital_bridge.py`) or through the two-phase files of `experiments/matched5/`.
ROI-JSCC additionally needs `einops` (torch env). `pip freeze` of the torch env is not pinned beyond `torch-env.yml`.

Optional: `data_tools/quantify_leakage.py` (near-duplicate leakage audit of a split) additionally needs `imagehash`.
