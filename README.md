Classify irrigated vs. non-irrigated cropland using Google’s AlphaEarth embeddings with a PyTorch Lightning MLP. Includes data export from Google Earth Engine (GEE), single split or k-fold training, Optuna hyperparameter tuning, model probability calibration, and TensorBoard logging. Three companion notebooks cover EDA/QC, training+calibration, and prediction+metrics.

Example run shown at bottom.

## Setup

```bash
git clone https://github.com/tcreed880/pytorch_irr_model.git
cd pytorch_irr_model

# install deps
poetry install

# set Python version if prompted
poetry env use python3.11

# verify it runs
poetry run python -c "import torch, pytorch_lightning as pl; print('ok')"
```
### Get data from Google Earth Engine
```bash
poetry run python -c "import ee; ee.Authenticate()"
# Follow the link, paste the code
```

Run this exporter. There is a version that generates a balanced 10k sample CSV per year, or generates data for all cropland pixels, grouped by county. Script currently set to generate data from Washington state.

```bash
poetry run python irr.cli.gee_data_export.gee_export_cropland_points \
  --states OR ID MT \
  --years 2018 2019 2020 2021 2022 \
  --points 20000 \
  --balance random \
  --exclude-near-state WA \ 
  --buffer-m 10000 \ 
```
balanced per-class sample (default)
```bash
poetry run gee-export --balance stratified --pos-frac 0.5 --years 2019 2020 2021
```
OR
all cropland pixels, chunked per county (large export)
```bash
poetry run python irr/cli/gee_python_api.py --mode all --years 2019 2020 2021
```
Exports go a Google Drive folder configured in the script. Download CSVs locally into raw_data/  
Columns should include FEATURES (64 AlphaEarth embeddings) and LABEL_COL (0 or 1, based on IrrMapper v1.2), and metadata like county_name, state, .geo.

### Model description
irr/models/mlp_classifier.py
MLP with residual blocks at width {hidden} ({depth} blocks), final 1-logit head  
Loss: BCEWithLogitsLoss, supports {pos_weight} from training for imbalanced classes  
Metrics: AUROC, AUPRC (TorchMetrics done at epoch level)  
Optimizer/scheduler: AdamW + CosineAnnealingLR (configured in configure_optimizers)  
Calibration (optional): temperature + bias on validation  
H3 group splits to avoid spatial leakage (set via --group-col h3_r{res})  

Inputs: for AlphaEarth unit-norm embeddings, the model’s standardizer is set to a no-op (mean=0, std=1) in run_train.



## Training
Supports H3 group-aware splits to avoid spatial leakage and standard label-stratified splits. H3 builds hex ids from .geo at the requested resolution (r5, r6, etc.) and enforces no hex overlap between train/val.

#### K-fold cross-validation method:
```bash
poetry run python -m irr.cli.kfold \
  --data-glob "raw_data/*.csv" \
  --k 5 \
  --batch-size 512 \
  --seed 88 \
  --monitor val_auprc \
  --patience 10 \
  --max-epochs 40 \
  --hidden 256 --depth 2 --dropout 0.10 --act silu \
  --lr 1e-3 --weight-decay 1e-4 \
  --group-col h3_r5 \
  --include-states MT OR ID
```
--group-col accepts h3_r{res} (e.g., h3_r5, h3_r7), .geo, county_fips, or none.


#### Single train/val split method
```bash
poetry run python -m irr.cli.train \
  --data-glob "raw_data/*.csv" \
  --batch-size 512 \
  --val-ratio 0.2 \  # validation split ratio
  --seed 88 \
  --monitor val_auprc \
  --patience 10 \
  --max-epochs 40 \
  --hidden 256 --depth 2 --dropout 0.10 --act silu \
  --lr 1e-3 --weight-decay 1e-4 \
  --group-col h3_r5 \
  --include-states MT OR ID
```

### TensorBoard logging
TensorBoard events: outputs/logs/mlp_classifier_tb/version_*  
CSV logs: outputs/logs/mlp_classifier/version_*  
Start Tensorboard:
```bash
poetry run tensorboard --logdir outputs/logs/mlp_classifier_tb --port 6006
```
Then open http://localhost:6006


### Optuna hyperparameter tuning

Optuna tuning CLI included to sweep batch size, depth/width, dropout, activation function, LR, and weight decay. By default it optimizes validation AUPRC with early stopping and checkpointing.

```bash
poetry run python -m irr.cli.optuna_tune \
  --data-glob "raw_data/*.csv" \
  --include-states MT OR ID \
  --group-col h3_r5 \
  --val-ratio 0.20 \
  --max-epochs 60 \
  --patience 10 \
  --n-trials 40 \
  --objective auprc \
  --study-name mlp_r5_auprc_seed88 \
  --storage "sqlite:///outputs/optuna/optuna.db"
```
To view optuna results:

```bash
poetry run optuna-dashboard sqlite:///outputs/optuna/optuna.db
```
Best checkpoint per trial: irr/outputs/optuna_tb/{study-name}/trial_X/checkpoints/best.ckpt


After using trials to choose best hyperparameters, re-train with the desired split and enable probability calibration: 
ModelConfig(calibrate_on_val=True), then run prediction.

At prediction/inference model.predict_proba(x) applies T,b if use_calibration is set.

The model can learn a temperature (T) and bias (b) on the validation set to make probabilities better calibrated. Then a working threshold is chosen (default: F1-optimal on the calibrated curve) and stored in the checkpoint.

## Prediction on new data using best checkpoint model
```bash
poetry run python -m irr.cli.predict \
  --ckpt "outputs/logs/mlp_classifier/version_20/checkpoints/best.ckpt" \
  --data-glob "new_data/*.csv" \
  --out-csv "outputs/predictions/new_data_with_preds.csv" \
  --batch-size 1028 \
  --threshold 0.5
```
## Notebooks

01_explore_and_qc.ipynb – quick data checks, label distributions, simple feature–label correlations, and sanity plots.

02_train_and_calibrate.ipynb – trains with your chosen hyperparameters, logs to TensorBoard, fits (T,b), and records the threshold.

03_predict_and_metrics.ipynb – loads a checkpoint, applies calibration, computes AUROC/AUPRC if labels are present, finds the best-F1 threshold, plots CM/ROC/PR, and writes a predictions CSV.


## Outputs

Training logs/checkpoints live under outputs/ and are git-ignored by default (see .gitignore).  
Predictions from notebooks save to notebooks/predictions/*.csv (also ignored).  

## Example Run
#### Predicting WA cropland irrigation using MT+OR+ID training data

Training/val data export:
```bash
poetry run python irr/cli/gee_data_export/gee_export_cropland_points.py \
  --states OR ID MT \
  --years 2018 2019 2020 2021 2022 \
  --points 20000 \
  --balance random \
  --exclude-near-state WA \
  --buffer-m 10000
```
Washington data for prediction
```bash
  poetry run python irr/cli/gee_data_export/gee_export_cropland_points.py \
  --states WA \
  --years 2018 2019 2020 2021 2022 \
  --points 20000 \
  --balance random \
```

Training with h3_r5 resolution grouping to improve generalization and calibration enabled. Hyperparameters selected basen on best performing Optuna trials using val_auprc as objective:
```bash
poetry run python -m irr.cli.train \
  --data-glob "raw_data/*.csv" \
  --include-states MT OR ID \
  --group-col h3_r5 \
  --val-ratio 0.10 \
  --seed 92 \
  --monitor val_auprc \
  --patience 20 \
  --min-delta 1e-4 \
  --max-epochs 120 \
  --batch-size 1024 \
  --hidden 256 --depth 2 --dropout 0.0511 --act gelu \
  --lr 2.427e-4 --weight-decay 9.208e-5 \
  --calibrate-on-val
  ```

Prediciton/evaluation results on WA. Metrics below are computed at the best-F1 threshold on calibrated probabilities:

Confusion counts: TN=88590, FP=853, FN=908, TP=4476 (N=94,827)  
Accuracy: 0.9814  
Precision: 0.8399  
Recall: 0.8314  
F1: 0.8356  