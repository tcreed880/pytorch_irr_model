# pytorch-irr-model
git clone https://github.com/<you>/pytorch_irr_model_modular.git
cd pytorch_irr_model_modular
python -m venv .venv && source .venv/bin/activate   # or conda/mamba if preferred
pip install -U pip
pip install -e .[viz] jupyter                       # optional extras like [viz]
jupyter lab                                         # open notebooks/…
