# Transformer-Driven Cost Matrix Adaptation in Model Predictive Control of DC Motors

Transformer-based adaptive control framework for automatic MPC cost matrix tuning in DC motor control, learning scenario-dependent P, Q, and R parameters from operating conditions and motor dynamics.

The repository includes the project code and the datasets created for it in `MPC_dataset/` and `MPC_src/mpc_pqr_dataset_streaming_debug_500_samples/`. Training checkpoints, evaluation outputs, and generated diagrams are kept outside Git.

## Docker (micromamba) setup

## Build

```bash
docker build -t mpc-mamba .
```

## Run training (CPU)

From the project folder:

```bash
docker run --rm -it \
  -v "$(pwd)/runs:/app/MPC_new/runs" \
  mpc-mamba
```

## Run with custom args

```bash
docker run --rm -it \
  -v "$(pwd)/runs:/app/MPC_new/runs" \
  mpc-mamba \
  python MPC_src/train.py --device cpu --epochs 30 --batch_size 128 --out_dir runs/run_cpu
```

## If you want GPU

1) Install NVIDIA Container Toolkit on your host.
2) Build the same image, then run with:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/runs:/app/MPC_new/runs" \
  mpc-mamba \
  python MPC_src/train.py --device cuda --epochs 30 --out_dir runs/run_gpu
```

> Note: for best GPU support, you may want a CUDA-specific PyTorch install. If you tell me your CUDA version, I can adjust `environment.yml` accordingly.
