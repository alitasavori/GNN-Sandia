"""
Train the best PF-identity GNN architecture for the Original dataset
and save the checkpoint under models_gnn2/original/.

Best architecture (from GNN3 exploration):
- Config name: light_xwide_emb_depth4
- Dataset: datasets_gnn2/original
- Hyperparameters: n_emb=16, e_emb=8, h=128, L=4, use_norm=False, use_phase_onehot=False
"""

import os
import torch

from run_gnn3_best7_train import train_one, ORIGINAL_FEAT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "original")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "original")


def main() -> None:
    os.chdir(BASE_DIR)

    cfg_name = "light_xwide_emb_depth4"

    ckpt_path = train_one(
        block_id=101,  # arbitrary ID to keep files distinct from other runs
        cfg_name=cfg_name,
        out_dir=DATASET_DIR,
        feature_cols=ORIGINAL_FEAT,
        target_col="vmag_pu",
        n_emb=16,
        e_emb=8,
        h_dim=128,
        n_layers=4,
        use_norm=False,
        use_phase_onehot=False,
        early_stop=False,  # train full 50 epochs, keep best by test RMSE (avoids constant-output collapse on daily profile)
    )

    if ckpt_path is None:
        print("[WARN] Training was skipped or failed; no checkpoint created.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{cfg_name}_best.pt")
    torch.save(ckpt, out_path)
    print(f"[SAVED] Best Original model -> {out_path}")


if __name__ == "__main__":
    main()

