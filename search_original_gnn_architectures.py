"""
Search a small set of promising PF-identity GNN architectures for the Original dataset.

- Uses datasets_gnn2/original
- Uses 1/3 of the valid snapshots
- Uses early stopping for every candidate
- Saves a ranked CSV and copies the best checkpoint into models_gnn2/original/
"""

from __future__ import annotations

from architecture_search_common import run_architecture_search
from run_gnn3_best7_train import ORIGINAL_FEAT


CANDIDATES = [
    {"cfg_name": "light_wide_emb_depth3", "n_emb": 12, "e_emb": 6, "h_dim": 96, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "light_wide_emb_depth4", "n_emb": 12, "e_emb": 6, "h_dim": 96, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "light_xwide_emb_depth4", "n_emb": 16, "e_emb": 8, "h_dim": 128, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "light_xwide_emb_depth5", "n_emb": 16, "e_emb": 8, "h_dim": 128, "n_layers": 5, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "wider_h160_depth4", "n_emb": 16, "e_emb": 8, "h_dim": 160, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
]


def main() -> None:
    run_architecture_search(
        search_name="original",
        dataset_dir="datasets_gnn2/original",
        models_dir="models_gnn2/original",
        feature_cols=ORIGINAL_FEAT,
        target_col="vmag_pu",
        candidates=CANDIDATES,
        block_id_start=710,
    )


if __name__ == "__main__":
    main()
