.PHONY: install install-cuda kernel notebook part-a part-b part-c clean

# ─── Dependency management ──────────────────────────────────────────────────

## Install all dependencies (CPU + MPS support — Mac Apple Silicon and Windows/Linux CPU)
install:
	poetry install --no-root

## Override PyTorch with CUDA 12.1 build (Windows/Linux GPU users only)
install-cuda:
	poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade

## Register the Jupyter kernel for this environment
kernel:
	poetry run python -m ipykernel install --user --name challenge7 --display-name "Python (challenge7)"

# ─── Development ────────────────────────────────────────────────────────────

## Launch JupyterLab
notebook: kernel
	poetry run jupyter lab --notebook-dir=notebooks/

## Run Part A notebook non-interactively (requires nbconvert)
part-a:
	poetry run jupyter nbconvert --to notebook --execute notebooks/part_a_classification.ipynb \
		--output notebooks/part_a_classification_executed.ipynb --ExecutePreprocessor.timeout=3600

## Run Part B notebook non-interactively
part-b:
	poetry run jupyter nbconvert --to notebook --execute notebooks/part_b_style_transfer.ipynb \
		--output notebooks/part_b_style_transfer_executed.ipynb --ExecutePreprocessor.timeout=36000

## Run Part C notebook non-interactively
part-c:
	poetry run jupyter nbconvert --to notebook --execute notebooks/part_c_domain_adaptation.ipynb \
		--output notebooks/part_c_domain_adaptation_executed.ipynb --ExecutePreprocessor.timeout=7200

## Run all parts sequentially
all: part-a part-b part-c

# ─── Utilities ───────────────────────────────────────────────────────────────

## Show the active Python interpreter and installed packages
env-info:
	poetry run python -c "import torch; print('torch:', torch.__version__); \
		import torchvision; print('torchvision:', torchvision.__version__); \
		print('CUDA available:', torch.cuda.is_available()); \
		print('MPS available:', torch.backends.mps.is_available())"

## Remove generated checkpoints and figures (keep data)
clean:
	rm -f checkpoints/*.pt figures/*.png figures/*.pdf figures/*.csv
	find notebooks/ -name '*_executed.ipynb' -delete
