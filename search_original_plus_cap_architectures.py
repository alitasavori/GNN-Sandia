"""
Search 3 promising PF-identity GNN architectures on the Original+Cap dataset.

- Uses existing datasets_gnn2/original_plus_cap (no dataset generation).
- Uses 1/3 of snapshots and early stopping per candidate.
- Reports MAE/RMSE for each; saves ranked CSV and the best checkpoint to
  gnn2_architecture_search/original_plus_cap/ and models_gnn2/original_plus_cap/.

Run from repo root:
  python search_original_plus_cap_architectures.py
"""

from __future__ import annotations

from architecture_search_common import run_architecture_search

# 5 features: p_load_kw, q_load_kvar, p_pv_kw, q_pv_kvar, q_cap_kvar
FEAT_ORIGINAL_PLUS_CAP = ["p_load_kw", "q_load_kvar", "p_pv_kw", "q_pv_kvar", "q_cap_kvar"]

# 3 promising configs (aligned with best original / injection style)
CANDIDATES = [
    {"cfg_name": "light_wide_emb_depth3", "n_emb": 12, "e_emb": 6, "h_dim": 96, "n_layers": 3, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "light_xwide_emb_depth4", "n_emb": 16, "e_emb": 8, "h_dim": 128, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
    {"cfg_name": "wider_h160_depth4", "n_emb": 16, "e_emb": 8, "h_dim": 160, "n_layers": 4, "use_norm": False, "use_phase_onehot": False},
]


def main() -> None:
    run_architecture_search(
        search_name="original_plus_cap",
        dataset_dir="datasets_gnn2/original_plus_cap",
        models_dir="models_gnn2/original_plus_cap",
        feature_cols=FEAT_ORIGINAL_PLUS_CAP,
        target_col="vmag_pu",
        candidates=CANDIDATES,
        block_id_start=720,
    )


if __name__ == "__main__":
    main()
