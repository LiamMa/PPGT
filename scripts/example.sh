#!/bin/bash
set -euo pipefail

COMMON_ARGS=(mlflow.use True accelerator cuda:0 seed 0)

# Small/medium graph benchmarks
python main.py --cfg configs/zinc-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/struct-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/pattern-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/sp-mnist-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/func-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/cluster-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/sp-cifar-PPGT.yaml "${COMMON_ARGS[@]}"

# BREC
python main.py --cfg configs/brec-PPGT.yaml "${COMMON_ARGS[@]}"
python main.py --cfg configs/brec-I2GIN+PPGT.yaml "${COMMON_ARGS[@]}" dataset.name subset-Regular

# PCQM4Mv2 with 2 GPUs
python main.py --cfg configs/pcqm4mv2-PPGT-ViT-B-BS256x2GPU.yaml accelerator "cuda:0,cuda:1" seed 0 mlflow.use True num_workers 0

# OGBN-Arxiv
python main.py --cfg configs/ogbn-arxiv-PPGT-2x128+CA-SG10.yaml "${COMMON_ARGS[@]}"
