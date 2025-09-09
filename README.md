Classify irrigated vs. non-irrigated cropland using Google’s AlphaEarth embeddings with a PyTorch Lightning MLP. Includes data export from Google Earth Engine (GEE), single split or k-fold training, and TensorBoard logging.

### Setup
#### pytorch-irr-model
git clone https://github.com/tcreed880/pytorch_irr_model.git
cd pytorch_irr_model

#### install deps
poetry install

#### set Python version if prompted
poetry env use python3.11

#### verify it runs
poetry run python -c "import torch, pytorch_lightning as pl; print('ok')"

### Get data from Google Earth Engine
poetry run python -c "import ee; ee.Authenticate()"
#### Follow the link, paste the code
#### Run this exporter. There is a version that generates a balanced 10k sample CSV per year, or generates data for all cropland pixels, grouped by county. Script currently set to generate data from Washington state.
poetry run python irr/cli/gee_python_api.py

#### balanced per-class sample (default)
poetry run python irr/cli/gee_python_api.py --mode balanced --years 2019 2020 2021
#### OR
#### all cropland pixels, chunked per county (large export)
poetry run python irr/cli/gee_python_api.py --mode all --years 2019 2020 2021

#### Exports go a Google Drive folder configured in the script. Download CSVs locally into raw_data/
#### Columns should include FEATURES (64 AlphaEarth embeddings) and LABEL_COL (0 or 1, based on IrrMapper v1.2)

### Training
#### K-fold cross-validation method:
poetry run python -m irr.cli.kfold \
  --data-glob "raw_data/*.csv" \
  --k 5 \
  --batch-size 512 \
  --seed 88 \
  --monitor val_auprc \
  --patience 10 \
  --max-epochs 40 \
  --hidden 256 --depth 2 --dropout 0.10 --act silu \
  --lr 1e-3 --weight-decay 1e-4

#### Single train/val split method
poetry run python -m irr.cli.train_tiny_head \
  --data-glob "raw_data/*.csv" \
  --batch-size 512 \
  --val-ratio 0.2 \ # validation split ratio
  --seed 88 \
  --monitor val_auprc \
  --patience 10 \
  --hidden 256 --depth 2 --dropout 0.10 --act silu \
  --lr 1e-3 --weight-decay 1e-4


### Model description
irr/models/tiny_head.py
MLP with residual blocks at width hidden (depth blocks), final logit head.

Loss: BCEWithLogitsLoss
Metrics: AUROC, AUPRC (TorchMetrics)
Optimizer/scheduler: AdamW + CosineAnnealingLR (configured in configure_optimizers)

Inputs: for AlphaEarth unit-norm embeddings, the model’s standardizer is set to a no-op (mean=0, std=1) in run_train.

### TensorBoard logging
TensorBoard events: outputs/logs/tiny_head_tb/version_*
CSV logs: outputs/logs/tiny_head/version_*
Start Tensorboard:
poetry run tensorboard --logdir outputs/logs/tiny_head_tb --port 6006
open http://localhost:6006

### Prediction on new data using best checkpoint model
poetry run python -m irr.cli.predict \
  --ckpt "outputs/logs/tiny_head/version_20/checkpoints/best.ckpt" \
  --data-glob "new_data/*.csv" \
  --out-csv "outputs/predictions/new_data_with_preds.csv" \
  --batch-size 4096 \
  --threshold 0.5
