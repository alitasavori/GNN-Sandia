"""
Temporary: Train and save only the Delta-V model (Block 6) with the Delta-V dataset.
Same logic as run_gnn3_best7_train.py but restricted to block 6 (datasets_gnn2/deltav).
Run after regenerating the dataset with run_deltav_dataset.py.
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
    print("GNN3 DELTA-V ONLY: Train block 6 on Delta-V dataset")
    print("=" * 70)
    for tup in MODELS:
        block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph = tup
        if block_id != 6:
            continue
        print(f"\n>>> Block {block_id}: {name} + {out_dir} (target={target})")
        train_one(block_id, name, out_dir, feat, target, n_emb, e_emb, h_dim, n_layers, use_norm, use_ph)
    print("\n" + "=" * 70)
    print("Done. Checkpoints in", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
