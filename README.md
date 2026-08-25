# Plain Transformers Can be Powerful Graph Learners

Code and experiment configurations for
[**Plain Transformers Can be Powerful Graph Learners**](https://openreview.net/forum?id=bEmDvP0fdv),
*Transactions on Machine Learning Research* (TMLR), 2026 (Expert Certification).

Liheng Ma, Soumyasundar Pal, Yingxue Zhang, Philip Torr, Mark Coates.


## Repository Structure

- `ppgt/`: model, layer, encoder, dataset, training, loss, metric, and utility code.
- `configs/`: experiment configuration files.
- `scripts/`: example commands for running experiments.
- `main.py`: entry point for training / evaluation.
- `requirements.txt`: Python dependencies (see *Environment Setup*).

## Environment Setup

This project was developed across two environments:

- **Legacy environment** (`torch==2.1.2`, Python 3.9): the environment most of the
  experiments in the paper were run in. It is documented below **for provenance
  only**.
- **Later environment** (`torch==2.4.1`, Python 3.12): used for the large-scale
  PCQM4Mv2 experiments and for multi-GPU training. **This is the environment to
  use.**

> **Reproducibility disclaimer.** The released code requires **PyTorch >= 2.4**:
> every config in `configs/` uses `norm_fn: ada_rms_norm`, which is built on
> `torch.nn.functional.rms_norm`, introduced in PyTorch 2.4. The legacy
> environment therefore cannot run this code as released. It is recorded here
> because some numbers reported in the paper were produced under it, so small
> deviations from the paper's results are possible when re-running in the later
> environment.

### Legacy Environment

*Recorded for provenance; see the disclaimer above. Use the later environment
to run this code.*

```bash
conda create -n ppgt-legacy python=3.9
conda activate ppgt-legacy

# Change the CUDA version as needed for your machine.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.
conda install openbabel fsspec rdkit -c conda-forge
# Alternative if conda does not work:
# pip install rdkit

pip install torchmetrics==0.9.1
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym
pip install pytorch-lightning
pip install scikit-learn==1.3
pip install setuptools==59.5.0
pip install timm
pip install einops
pip install mlflow

# Optional:
# pip install performer-pytorch
# pip install wandb
```

### Later Environment (recommended)

```bash
conda create -n ppgt python=3.12
conda activate ppgt

# PyTorch with CUDA 11.8. Change the CUDA version as needed for your machine.
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118

# PyG.
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# The rest of the dependencies.
pip install -r requirements.txt

# Optional: experiment tracking.
pip install mlflow
pip install wandb
```

Or, equivalently, in place of `requirements.txt`:

```bash
pip install rdkit==2024.3.1 ogb brec loguru
pip install torchmetrics==0.9.1
pip install "numpy==1.26.3" "scipy==1.13.1" "scikit-learn==1.3.2"
pip install yacs pytorch-lightning tensorboardX
pip install opt_einsum timm einops
```

Notes:

- The `numpy` / `scipy` / `scikit-learn` versions must be pinned **together**.
  `scipy >= 1.16` requires `numpy >= 2`, which `rdkit==2024.3.1` does not
  support, so pinning `numpy` alone leaves a broken environment in which
  `torch-sparse` silently fails to import.
- `torchmetrics==0.9.1` is required: the metric wrappers in `ppgt/logger.py`
  use its pre-1.0 functional API.
- Pin `scikit-learn==1.3.2`, not `1.3`: `1.3` resolves to `1.3.0`, which has
  no CPython 3.12 wheel and is built from source (slow, and against whichever
  `numpy` happens to be installed at that moment).
- `brec` is only needed for the BREC experiments (it also provides `loguru`,
  which `ppgt/train/brec_train.py` imports). For BREC we do **not** use
  `graph-tool`; installing it tends to create further environment conflicts.
- The standalone `graphgym` package is **not** required. This code uses
  `torch_geometric.graphgym`, which ships with PyG.

## Logging

MLflow is supported for experiment logging. Start an MLflow server with:

```bash
mlflow server --backend-store-uri mlruns --port 5000
```

Weights & Biases is also wired up (`wandb.use True`), inherited from GraphGPS,
but it is not required for any of the experiments and is less exercised than the
MLflow path. Set `wandb.entity` to your own entity before enabling it.

## Running Experiments

Example commands are provided in [`scripts/example.sh`](scripts/example.sh).

The general command format is:

```bash
python main.py --cfg configs/{config_name}.yaml mlflow.use True accelerator "cuda:0" seed 0
```

Replace:

- `configs/{config_name}.yaml` with the target configuration file.
- `"cuda:0"` with the device you want to use (`cpu` also works).
- `seed 0` with the desired random seed.

Any config entry can be overridden on the command line in the same
`key value` form, e.g. `optim.max_epoch 100` or `train.batch_size 64`.

`--repeat N` runs the experiment `N` times with seeds `seed, seed+1, ...`, which
is how the multi-seed results in the paper were produced:

```bash
python main.py --cfg configs/zinc-PPGT.yaml --repeat 4 accelerator "cuda:0" seed 0
```

Results are written to `{out_dir}/{config_name}/{seed}/`, with `out_dir` taken
from the config (`results` by default).

Datasets are downloaded automatically to `./datasets` when supported by the dataset loader.

### Example

```bash
python main.py \
  --cfg configs/zinc-PPGT.yaml \
  mlflow.use True \
  accelerator "cuda:0" \
  seed 0
```

## Multi-GPU Training

Multi-GPU devices are specified with a comma-separated accelerator string:

```bash
accelerator "cuda:0,cuda:1"
```

This launches one `DistributedDataParallel` process per device via
`torch.multiprocessing.spawn`, sharding the **training** set with a
`DistributedSampler`. See the PCQM4Mv2 example in
[`scripts/example.sh`](scripts/example.sh).

Caveats:

- Only graph-level tasks (`dataset.task: graph`) are supported; other tasks
  raise `NotImplementedError`.
- Validation and test sets are *not* sharded: every rank evaluates the full
  split, so that all ranks drive their LR scheduler with the same metric.
- Logging is not rank-aware: all ranks write into the same run directory, and
  enabling MLflow/W&B starts one run per rank. Prefer a single device when you
  care about the logs.
- `MASTER_ADDR` / `MASTER_PORT` can be set in the environment to run several
  multi-GPU jobs on one machine.

## BREC Dataset Note


For BREC experiments, download the data from
[GraphPKU/BREC](https://github.com/GraphPKU/BREC), unzip `BREC_data_all.zip`,
and place the extracted files under `datasets/BREC/raw/`. The loader looks for:

```text
datasets/BREC/raw/brec_v3.npy
```

To run a subset of BREC, see the example in [`scripts/example.sh`](scripts/example.sh):
```bash
python main.py \
  --cfg configs/brec-I2GIN+PPGT.yaml \
  mlflow.use True \
  accelerator "cuda:0" \
  seed 0 \
  dataset.name subset-Regular
```
- dataset support: set `dataset.name`
    - `brec`: for all data in BREC 
    - `subset-{Regular/Basic/Extension/CFI}`: for a subset of BREC -- Regular/Basic/Extension/CFI graphs.




## Citation

If you find this work useful, please cite:

```bibtex
@article{
ma2026plain,
title={Plain Transformers Can be Powerful Graph Learners},
author={Liheng Ma and Soumyasundar Pal and Yingxue Zhang and Philip Torr and Mark Coates},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=bEmDvP0fdv},
note={Expert Certification}
}
```

## Acknowledgements

This codebase is adapted from the official code for [Graph Inductive Biases in Transformers without Message Passing (ICML 2023)](https://github.com/LiamMa/GRIT) and [GraphGPS: General Powerful Scalable Graph Transformers (NeurIPS 2022)](https://github.com/rampasek/graphgps).
The experiment of BREC follows [An Empirical Study of Realized GNN Expressiveness (ICML 2024)](https://github.com/GraphPKU/BREC).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
Portions of the code are adapted from GRIT and GraphGPS; please also respect their respective licenses.
