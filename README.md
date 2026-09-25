# Transformer-Driven Cost Matrix Adaptation in Model Predictive Control of DC Motors

An orchestrated research workflow for DC motor control: offline MPC tuning generates scenario-specific cost matrices, a Transformer learns to estimate their log-diagonal entries from temporal and static features, and closed-loop simulation evaluates the adapted controller.

**Paper:** [Transformer-Driven Cost Matrix Adaptation in Model Predictive Control of DC Motors](paper/Transformer_Based_Estimation_of_MPC_Cost_Matrices_for_DC_Motor_Control.pdf). The manuscript is forthcoming; publication details will be added when available.

## Repository contents

| Path | Contents |
| --- | --- |
| `MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_spec_midanchor/` | Final recalibrated dataset used in the paper: 3,000 DC motor scenarios with `meta.json`, `signals.npz`, and `closed_loop.npz` per sample. |
| `MPC_dataset/mpc_qr_dataset/*.py` | Dataset generation, P/Q/R target construction, and recalibration code. |
| `MPC_src/dataset.py`, `model.py`, `train.py` | Dataset loader, Transformer with attention pooling, and training. |
| `MPC_src/evaluate_motor_pqr.py`, `generate_control_plots.py` | Parameter prediction and closed-loop evaluation. |
| `paper/` | Prepublication manuscript. |

The paper uses a 90%/10% train/validation split of the 3,000-sample dataset. It predicts the logarithms of the diagonal entries of the MPC matrices P, Q, and R. The reported final configuration uses two Transformer encoder layers, four attention heads, attention pooling, Huber loss for P and Q, and MSE for R.

## Setup

Use Python 3.11 or later and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the paper configuration

Run from the repository root:

```bash
python MPC_src/train.py \
  --data_root MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_spec_midanchor \
  --arch transformer --seq_pool attention --target_mode log_diag \
  --target_loss huber --target_loss_p huber --target_loss_q huber --target_loss_r mse \
  --epochs 30 --batch_size 64 --weight_decay 0.03 --dropout 0.2 \
  --d_model 128 --tx_layers 2 --tx_heads 4 --seed 42 --device cpu \
  --out_dir runs/paper_model
```

Training saves the best checkpoint and normalization statistics in `runs/paper_model/`.

## Evaluate

```bash
python MPC_src/evaluate_motor_pqr.py \
  --checkpoint runs/paper_model/best.pt \
  --data_root MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_spec_midanchor

python MPC_src/generate_control_plots.py \
  --checkpoint runs/paper_model/best.pt \
  --data_root MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_spec_midanchor
```

The dataset snapshot is ready for training and evaluation. Its manifest records paths to intermediate files used during the original recalibration; those intermediate files are not required to load the final samples.
