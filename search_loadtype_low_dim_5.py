"""
Search 5 low-dimension (faster) architectures for the Load-type dataset.

Goal:
  Explore whether smaller (lower h_dim / fewer layers / smaller embeddings) models
  can achieve comparable accuracy with lower inference cost.

Notes:
  - Reuses the existing search harness (architecture_search_common.run_architecture_search)
  - Uses 1/3 of the dataset with early stopping (same as the existing loadtype search)

Run:
  python search_loadtype_low_dim_5.py
"""

from __future__ import annotations

from architecture_search_common import run_architecture_search
from run_gnn3_best7_train import LOADTYPE_FEAT


# Smaller candidates than the current "best" families.
# These are intentionally "lean" (lower h_dim, fewer layers, smaller embeddings).
CANDIDATES = [
    # Very small baseline (no embeddings, shallow)
    {"cfg_name": "tiny_noemb_h48_depth2", "n_emb": 0, "e_emb": 0, "h_dim": 48, "n_layers": 2, "use_norm": False, "use_phase_onehot": False},
    # Small with embeddings
    {"cfg_name": "tiny_emb_h48_depth2", "n_emb": 8, "e_emb": 4, "h_dim": 48, "n_layers": 2, "use_norm": False, "use_phase_onehot": False},
    # Slightly wider, still shallow
    {"cfg_name": "small_emb_h64_depth2", "n_emb": 8, "e_emb": 4, "h_dim": 64, "n_layers": 2, "use_norm": False, "use_phase_onehot": False},
    # Small but a bit deeper (tests if depth helps at small width)
    {"cfg_name": "small_emb_h64_depth3", "n_emb": 8, "e_emb": 4, "h_dim": 64, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    # Small with phase one-hot (can help accuracy cheaply)
    {"cfg_name": "small_emb_h64_phase_onehot_depth2", "n_emb": 8, "e_emb": 4, "h_dim": 64, "n_layers": 2, "use_norm": False, "use_phase_onehot": True},
]


def main() -> None:
    run_architecture_search(
        search_name="loadtype_low_dim_5",
        dataset_dir="datasets_gnn2/loadtype",
        models_dir="models_gnn2/loadtype_low_dim_5",
        feature_cols=LOADTYPE_FEAT,
        target_col="vmag_pu",
        candidates=CANDIDATES,
        block_id_start=950,
    )


if __name__ == "__main__":
    main()

