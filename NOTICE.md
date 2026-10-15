# Third-party material and licensing notes

The MIT licence in `LICENSE` covers the original contributions of the AquaSC authors. It does not relicense third-party material:

| Material | Source | Status |
|---|---|---|
| Masked-VQ-VAE semantic-communication code (ViT encoder/decoder skeleton, training loop scaffolding: `engine1.py`, `optim_factory.py`, parts of `model.py`, `model_util.py`, `utils.py`) | Hu et al., *Robust Semantic Communications with Masked VQ-VAE Enabled Codebook*, https://github.com/hqyyqh888/RobustSemanComm | **The upstream repository carries no licence file** (checked 2026-10-15). The authors distribute their derived files under MIT on the understanding above; anyone needing certainty should contact the upstream authors. |
| ROI-JSCC baseline | https://github.com/hansung-choi/ROI-JSCC (MIT) | Not vendored: `scripts/setup_baselines.sh` clones it and overlays our evaluation adapters (`baselines/overlay/ROI-JSCC/`). |
| JSCC-CNN baseline, SwinJSCC baseline | forks listed in `baselines/BASELINES.txt` (SwinJSCC: Yang et al., https://arxiv.org/abs/2308.09361) | Not vendored; the forks carry no licence file. Only our adapters/evaluators are shipped (`baselines/overlay/`). |
| YOLO detector | Ultralytics (AGPL-3.0) | Used through the `ultralytics` package; trained detector weights are distributed separately (`docs/CHECKPOINTS.md`) and are subject to that licence. |
| NVIDIA Sionna 1.2.2 (LDPC/QAM/channel), TensorFlow, PyTorch | their respective licences | Installed from PyPI/conda (`requirements/`). |
| VinFish dataset (Site A) and FishDet-NB2 data (Site B) | data owners | **Not redistributed here.** Licence/terms: see `docs/DATA.md`. |
