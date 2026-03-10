"""
Train the best PF-identity GNN architecture for the Load-type dataset
using the same hyperparameters as the light_emb_h96_depth2 search winner,
but **without** node and edge ID embeddings.

Differences vs the original architecture search winner:
- Same dataset: datasets_gnn2/loadtype
- Same hidden size and depth: h=96, L=2
- No node/edge embeddings: node_emb_dim=0, edge_emb_dim=0
"""

import os
import torch

from run_gnn3_best7_train import train_one, LOADTYPE_FEAT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets_gnn2", "loadtype")
MODELS_DIR = os.path.join(BASE_DIR, "models_gnn2", "loadtype")


def main() -> None:
    os.chdir(BASE_DIR)

    # Config name chosen to match load-type + h=96, depth=2, no embeddings.
    cfg_name = "light_h96_depth2_noemb"

    ckpt_path = train_one(
        block_id=106,  # distinct ID to avoid clobbering existing blocks
        cfg_name=cfg_name,
        out_dir=DATASET_DIR,
        feature_cols=LOADTYPE_FEAT,
        target_col="vmag_pu",
        n_emb=0,          # disable node embeddings
        e_emb=0,          # disable edge embeddings
        h_dim=96,
        n_layers=2,
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
    print(f"[SAVED] Load-type no-embeddings model -> {out_path}")


if __name__ == "__main__":
    main()

