"""
Train a load-type PF-identity GNN using the current loadtype dataset
but without the global features:
  - p_sys_balance_kw
  - q_sys_balance_kvar

This keeps the local load-type representation:
  electrical_distance_ohm, m1_p_kw, m1_q_kvar, m2_p_kw, m2_q_kvar,
  m4_p_kw, m4_q_kvar, m5_p_kw, m5_q_kvar, q_cap_kvar, p_pv_kw, q_pv_kvar

The script reuses the existing training pipeline and saves the checkpoint under
models_gnn2/loadtype/.
"""

from __future__ import annotations

import os
import torch

from run_gnn3_best7_train import train_one


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "loadtype")

LOADTYPE_NO_GLOBAL_FEAT = [
    "electrical_distance_ohm",
    "m1_p_kw",
    "m1_q_kvar",
    "m2_p_kw",
    "m2_q_kvar",
    "m4_p_kw",
    "m4_q_kvar",
    "m5_p_kw",
    "m5_q_kvar",
    "q_cap_kvar",
    "p_pv_kw",
    "q_pv_kvar",
]


def main() -> None:
    os.chdir(BASE_DIR)

    cfg_name = "light_emb_h96_phase_onehot_depth3_h112_no_global"
    ckpt_path = train_one(
        block_id=105,
        cfg_name=cfg_name,
        out_dir=DATASET_DIR,
        feature_cols=LOADTYPE_NO_GLOBAL_FEAT,
        target_col="vmag_pu",
        n_emb=16,
        e_emb=8,
        h_dim=112,
        n_layers=3,
        use_norm=False,
        use_phase_onehot=True,
    )

    if ckpt_path is None:
        print("[WARN] Training was skipped or failed; no checkpoint created.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{cfg_name}_best.pt")
    torch.save(ckpt, out_path)
    print(f"[SAVED] No-global Load-type model -> {out_path}")


if __name__ == "__main__":
    main()
