"""
Search a small set of promising PF-identity GNN architectures for the Injection dataset.

- Uses datasets_gnn2/injection
- Uses 1/3 of the valid snapshots
- Uses early stopping for every candidate
- Saves a ranked CSV and copies the best checkpoint into models_gnn2/injection/
"""

from __future__ import annotations

from architecture_search_common import run_architecture_search
from run_gnn3_best7_train import INJECTION_FEAT


CANDIDATES = [
    {"cfg_name": "deep_h96_depth4", "n_emb": 12, "e_emb": 6, "h_dim": 96, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "wide_shallow_h128_depth3", "n_emb": 8, "e_emb": 4, "h_dim": 128, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "wide_shallow_h160_depth3", "n_emb": 8, "e_emb": 4, "h_dim": 160, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "wide_shallow_h192_depth3", "n_emb": 8, "e_emb": 4, "h_dim": 192, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "deep_h160_depth4", "n_emb": 12, "e_emb": 6, "h_dim": 160, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
]


def main() -> None:
    run_architecture_search(
        search_name="injection",
        dataset_dir="datasets_gnn2/injection",
        models_dir="models_gnn2/injection",
        feature_cols=INJECTION_FEAT,
        target_col="vmag_pu",
        candidates=CANDIDATES,
        block_id_start=810,
    )


if __name__ == "__main__":
    main()
