from __future__ import annotations

import json
import os
import shutil
from typing import Iterable

import pandas as pd
import torch

import run_gnn3_best7_train as trainer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_ROOT = os.path.join(BASE_DIR, "gnn2_architecture_search")
SEARCH_DATA_FRAC = 1.0 / 3.0


def run_architecture_search(
    *,
    search_name: str,
    dataset_dir: str,
    models_dir: str,
    feature_cols: list[str],
    target_col: str,
    candidates: Iterable[dict],
    block_id_start: int,
) -> str:
    """Train a small candidate set and return the best checkpoint path.

    The shared trainer is reused so the data cleaning, graph construction, and
    checkpoint format stay aligned with the existing train_best_*.py scripts.
    """
    os.chdir(BASE_DIR)

    dataset_dir = os.path.join(BASE_DIR, dataset_dir)
    models_dir = os.path.join(BASE_DIR, models_dir)
    search_dir = os.path.join(SEARCH_ROOT, search_name)
    os.makedirs(search_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    old_output_dir = trainer.OUTPUT_DIR
    old_data_frac = trainer.DATA_FRAC

    candidate_list = list(candidates)
    rows: list[dict] = []
    try:
        trainer.OUTPUT_DIR = search_dir
        trainer.DATA_FRAC = SEARCH_DATA_FRAC

        print("=" * 72)
        print(f"Architecture search: {search_name}")
        print(f"Dataset: {dataset_dir}")
        print(f"Using DATA_FRAC={SEARCH_DATA_FRAC:.6f} and early stopping for every run")
        print("=" * 72)

        for idx, cfg in enumerate(candidate_list):
            block_id = block_id_start + idx
            cfg_name = str(cfg["cfg_name"])
            print(
                f"\n>>> [{idx + 1}/{len(candidate_list)}] {cfg_name} "
                f"(n_emb={cfg['n_emb']}, e_emb={cfg['e_emb']}, h={cfg['h_dim']}, "
                f"L={cfg['n_layers']}, phase_onehot={cfg['use_phase_onehot']})"
            )
            ckpt_path = trainer.train_one(
                block_id=block_id,
                cfg_name=cfg_name,
                out_dir=dataset_dir,
                feature_cols=feature_cols,
                target_col=target_col,
                n_emb=int(cfg["n_emb"]),
                e_emb=int(cfg["e_emb"]),
                h_dim=int(cfg["h_dim"]),
                n_layers=int(cfg["n_layers"]),
                use_norm=bool(cfg.get("use_norm", False)),
                use_phase_onehot=bool(cfg.get("use_phase_onehot", False)),
                early_stop=True,
            )
            if ckpt_path is None:
                rows.append(
                    {
                        "cfg_name": cfg_name,
                        "n_emb": int(cfg["n_emb"]),
                        "e_emb": int(cfg["e_emb"]),
                        "h_dim": int(cfg["h_dim"]),
                        "n_layers": int(cfg["n_layers"]),
                        "use_norm": bool(cfg.get("use_norm", False)),
                        "use_phase_onehot": bool(cfg.get("use_phase_onehot", False)),
                        "best_mae": None,
                        "best_rmse": None,
                        "best_epoch": None,
                        "ckpt_path": None,
                    }
                )
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "cfg_name": cfg_name,
                    "n_emb": int(cfg["n_emb"]),
                    "e_emb": int(cfg["e_emb"]),
                    "h_dim": int(cfg["h_dim"]),
                    "n_layers": int(cfg["n_layers"]),
                    "use_norm": bool(cfg.get("use_norm", False)),
                    "use_phase_onehot": bool(cfg.get("use_phase_onehot", False)),
                    "best_mae": ckpt.get("best_mae"),
                    "best_rmse": ckpt.get("best_rmse"),
                    "best_epoch": ckpt.get("best_epoch"),
                    "ckpt_path": ckpt_path,
                }
            )
    finally:
        trainer.OUTPUT_DIR = old_output_dir
        trainer.DATA_FRAC = old_data_frac

    df = pd.DataFrame(rows)
    if df.empty or df["best_rmse"].notna().sum() == 0:
        raise RuntimeError(f"No successful runs were produced for search '{search_name}'.")

    df = df.sort_values(["best_rmse", "best_mae", "cfg_name"], na_position="last").reset_index(drop=True)
    csv_path = os.path.join(search_dir, "search_results.csv")
    df.to_csv(csv_path, index=False)

    best = df.iloc[0]
    best_src = str(best["ckpt_path"])
    best_cfg_name = str(best["cfg_name"])
    best_search_ckpt = os.path.join(search_dir, "best.pt")
    best_models_ckpt = os.path.join(models_dir, f"{best_cfg_name}_search_best.pt")
    shutil.copy2(best_src, best_search_ckpt)
    shutil.copy2(best_src, best_models_ckpt)

    summary = {
        "search_name": search_name,
        "dataset_dir": dataset_dir,
        "data_frac": SEARCH_DATA_FRAC,
        "best_cfg_name": best_cfg_name,
        "best_mae": None if pd.isna(best["best_mae"]) else float(best["best_mae"]),
        "best_rmse": None if pd.isna(best["best_rmse"]) else float(best["best_rmse"]),
        "best_epoch": None if pd.isna(best["best_epoch"]) else int(best["best_epoch"]),
        "best_search_checkpoint": best_search_ckpt,
        "best_models_checkpoint": best_models_ckpt,
        "results_csv": csv_path,
    }
    summary_path = os.path.join(search_dir, "best_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Best architecture for {search_name}: {best_cfg_name}")
    print(
        "  RMSE={:.6f} | MAE={:.6f} | epoch={}".format(
            float(summary["best_rmse"]),
            float(summary["best_mae"]),
            summary["best_epoch"],
        )
    )
    print(f"  Results CSV -> {csv_path}")
    print(f"  Best checkpoint (search dir) -> {best_search_ckpt}")
    print(f"  Best checkpoint (models dir) -> {best_models_ckpt}")
    print("=" * 72)
    return best_models_ckpt
