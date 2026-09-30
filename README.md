# SEN2NEON pipeline — run order

## 1. Directory layout

`evaluation/evaluate.py` and `evaluation/visualize.py` import `from models...`
and `from evaluation...`, so the project needs this shape:

```
project_root/
├── SEN2NEON_5GB/                 # your raw data
│   ├── s2_l2a_10m/
│   ├── neon_2.5m_linearized/
│   └── metadata.csv              # optional
├── config.yaml
├── train.py
├── losses.py
├── dataset.py
├── setup_pipeline.sh
├── preprocessing/
│   ├── create_splits.py
│   ├── create_patches.py
│   ├── normalize.py
│   └── verify_dataset.py
├── models/
│   ├── __init__.py
│   ├── baseline_cnn.py
│   ├── transformer.py
│   └── proposed_model.py
└── evaluation/
    ├── __init__.py
    ├── metrics.py
    ├── evaluate.py
    └── visualize.py
```

This folder is already arranged exactly this way — just unzip it and `cd`
into it. Put your `SEN2NEON_5GB/` data folder inside it (or point `--data-root`
at wherever it lives).

## 2. Install dependencies

```bash
pip install torch torchvision numpy pandas rasterio matplotlib pyyaml
```

## 3. Run the pipeline, in order

```bash
# 1) Leakage-safe train/val/test split at the acquisition level
python preprocessing/create_splits.py --data-root SEN2NEON_5GB

# 2) Build the patch index (no image data is duplicated)
python preprocessing/create_patches.py --data-root SEN2NEON_5GB

# 3) Compute normalization stats from the train split only
python preprocessing/normalize.py --data-root SEN2NEON_5GB

# 4) Sanity-check the resulting dataset
python preprocessing/verify_dataset.py --data-root SEN2NEON_5GB

# 5) Smoke-test the Dataset/DataLoader itself
python dataset.py --data-root SEN2NEON_5GB --samples 2

# 6) Smoke-test each model builds and trains a step (CPU-safe self-tests)
python models/baseline_cnn.py --device cpu
python models/transformer.py --device cpu
python models/proposed_model.py --device cpu

# 7) Smoke-test the loss functions
python losses.py --self-test

# 8) Smoke-test the metrics
python evaluation/metrics.py --self-test

# 9) Train (pick one model: cnn | transformer | hybrid)
python train.py --config config.yaml --model hybrid --data-root SEN2NEON_5GB

#    Quick smoke test first, before a full run, is recommended:
python train.py --config config.yaml --model cnn --epochs 1 \
    --max-train-samples 8 --max-val-samples 4 --num-workers 0

#    Resume an interrupted run:
python train.py --config config.yaml --resume runs/sen2neon/checkpoints/last.pt

# 10) Evaluate the best checkpoint on the held-out test set
python evaluation/evaluate.py --data-root SEN2NEON_5GB \
    --checkpoint runs/sen2neon/checkpoints/best.pt --model hybrid

# 11) Visualize a few predictions
python evaluation/visualize.py --data-root SEN2NEON_5GB \
    --checkpoint runs/sen2neon/checkpoints/best.pt --model hybrid --patch-index 0
```

## 4. What each new file does

- **`config.yaml`** — single source of truth for data paths, model choice
  and hyperparameters, loss weights, and output locations. Any field can be
  overridden from the `train.py` command line (e.g. `--epochs`, `--lr`).
- **`losses.py`** — `CompositeSRLoss`: Charbonnier (or plain L1) pixel loss +
  spectral-angle loss + differentiable SSIM loss, all mask-aware using the
  same `[B,C,H,W]` mask broadcasting rules as `evaluation/metrics.py`. Run
  `python losses.py --self-test` to verify gradients flow correctly.
- **`train.py`** — full training loop for all three models (`cnn`,
  `transformer`, `hybrid`): AdamW/Adam/SGD, cosine/plateau/no LR schedule
  with warmup, optional AMP, gradient clipping, early stopping, resumable
  checkpoints, and a `history.csv` log. Checkpoints are written in exactly
  the format `evaluation/evaluate.py` and `evaluation/visualize.py` already
  expect (`model_state_dict` + `model_config`), so no changes were needed
  to those two files.

## 5. Notes

- I could not execute a full training run in this environment (the sandbox's
  torch install is missing its CUDA runtime library), so `train.py` and
  `losses.py` are verified by static syntax compilation and a careful,
  line-by-line match against the function signatures in your `dataset.py`,
  `models/*.py`, and `evaluation/evaluate.py`. Please run the smoke tests in
  step 6–9 above first, and the 1-epoch smoke run in step 9, before a full
  training run.
- `config.yaml`'s `training.early_stopping_metric` accepts any key returned
  by `calculate_all_metrics` (`mae`, `rmse`, `psnr_db`, `ssim`,
  `sam_degrees`, `ergas`, `spectral_correlation`). Set
  `training.higher_is_better: false` if you switch to `mae`, `rmse`,
  `sam_degrees`, or `ergas` (lower is better for those).
