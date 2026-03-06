"""
Train the best PF-identity GNN architecture for the Derived (injection) dataset
and save the checkpoint under models_gnn2/injection/.

Best architecture (from GNN3 exploration):
- Config name: wide_shallow_h160_depth3
- Dataset: datasets_gnn2/injection
- Hyperparameters: n_emb=8, e_emb=4, h=160, L=3, use_norm=False, use_phase_onehot=False
"""

import os
import torch

from run_gnn3_best7_train import train_one, INJECTION_FEAT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "injection")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "injection")


def main() -> None:
    os.chdir(BASE_DIR)

    cfg_name = "wide_shallow_h160_depth3"

    ckpt_path = train_one(
        block_id=102,  # arbitrary ID to keep files distinct from other runs
        cfg_name=cfg_name,
        out_dir=DATASET_DIR,
        feature_cols=INJECTION_FEAT,
        target_col="vmag_pu",
        n_emb=8,
        e_emb=4,
        h_dim=160,
        n_layers=3,
        use_norm=False,
        use_phase_onehot=False,
    )

    if ckpt_path is None:
        print("[WARN] Training was skipped or failed; no checkpoint created.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{cfg_name}_best.pt")
    torch.save(ckpt, out_path)
    print(f"[SAVED] Best Derived (injection) model -> {out_path}")


if __name__ == "__main__":
    main()

