"""
Search a small set of promising PF-identity GNN architectures for the Load-type dataset.

- Uses datasets_gnn2/loadtype
- Uses 1/3 of the valid snapshots
- Uses early stopping for every candidate
- Saves a ranked CSV and copies the best checkpoint into models_gnn2/loadtype/
"""

from __future__ import annotations

from architecture_search_common import run_architecture_search
from run_gnn3_best7_train import LOADTYPE_FEAT


CANDIDATES = [
    {"cfg_name": "light_emb_h96_depth2", "n_emb": 16, "e_emb": 8, "h_dim": 96, "n_layers": 2, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "light_emb_h96_phase_onehot_depth3", "n_emb": 16, "e_emb": 8, "h_dim": 96, "n_layers": 3, "use_norm": False, "use_phase_onehot": True},
    {"cfg_name": "light_emb_h96_phase_onehot_depth3_h112", "n_emb": 16, "e_emb": 8, "h_dim": 112, "n_layers": 3, "use_norm": False, "use_phase_onehot": True},
    {"cfg_name": "light_emb_h128_phase_onehot_depth3", "n_emb": 16, "e_emb": 8, "h_dim": 128, "n_layers": 3, "use_norm": False, "use_phase_onehot": True},
    {"cfg_name": "light_emb_h112_phase_onehot_depth4", "n_emb": 16, "e_emb": 8, "h_dim": 112, "n_layers": 4, "use_norm": False, "use_phase_onehot": True},
]


def main() -> None:
    run_architecture_search(
        search_name="loadtype",
        dataset_dir="datasets_gnn2/loadtype",
        models_dir="models_gnn2/loadtype",
        feature_cols=LOADTYPE_FEAT,
        target_col="vmag_pu",
        candidates=CANDIDATES,
        block_id_start=910,
    )


if __name__ == "__main__":
    main()
