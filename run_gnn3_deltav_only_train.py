"""
Temporary: Train and save only the two last models (Block 6 & 7) with the new Delta-V datasets.
Same logic as run_gnn3_best7_train.py but only blocks 6 (gnn_samples_deltav_full) and 7 (gnn_samples_deltav_5x_full).
Run after regenerating datasets with run_deltav_dataset.py and run_deltav_5x_dataset.py.
"""
import os

# Ensure we use the same module as run_gnn3_best7_train
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Import and run only blocks 6 and 7 from the main trainer
from run_gnn3_best7_train import (
    MODELS,
    OUTPUT_DIR,
    train_one,
)

def main():
    print("=" * 70)
    print("GNN3 DELTA-V ONLY: Train blocks 6 & 7 on new datasets")
    print("=" * 70)
    for tup in MODELS:
        block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph = tup
        if block_id not in (6, 7):
            continue
        print(f"\n>>> Block {block_id}: {name} + {out_dir} (target={target})")
        train_one(block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph)
    print("\n" + "=" * 70)
    print("Done. Checkpoints in", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
