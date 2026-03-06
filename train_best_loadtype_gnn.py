"""
Train the best PF-identity GNN architecture for the Load-type dataset
and save the checkpoint under models_gnn2/loadtype/.

Best architecture (from GNN3 exploration):
- Config name: light_emb_h96_phase_onehot_depth3_h112
- Dataset: datasets_gnn2/loadtype
- Hyperparameters: n_emb=16, e_emb=8, h=112, L=3, use_norm=False, use_phase_onehot=True
"""

import os
import torch

from run_gnn3_best7_train import train_one, LOADTYPE_FEAT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "loadtype")


def main() -> None:
    os.chdir(BASE_DIR)

    cfg_name = "light_emb_h96_phase_onehot_depth3_h112"

    ckpt_path = train_one(
        block_id=103,  # arbitrary ID to keep files distinct from other runs
        cfg_name=cfg_name,
        out_dir=DATASET_DIR,
        feature_cols=LOADTYPE_FEAT,
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
    print(f"[SAVED] Best Load-type model -> {out_path}")


if __name__ == "__main__":
    main()

