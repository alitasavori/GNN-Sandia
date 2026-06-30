# Running the DA-GPS daily compare on Google Colab (CUDA GPU)

This makes `nonunique.ipynb` cell 2 (`mode="da_gps_daily_compare"` — DA-GPS GNN vs
OpenDSS native daily QSTS) runnable on a Colab GPU.

## Quick start

1. Open **`nonunique_colab.ipynb`** in Google Colab.
2. `Runtime > Change runtime type > GPU`.
3. Run the cells top to bottom:
   - **Clone** the repo into `/content/GNN2`.
   - **Install** `torch_geometric` + `opendssdirect.py` (Colab already has CUDA `torch`).
   - **Set env** `GNN2_REPO_ROOT=/content/GNN2` and `GNN_TORCH_COMPILE=0`.
   - **(No upload needed)** the slim DA-GPS cache ships in the repo; section 4 of
     the notebook just verifies it is present.
   - **Run** the compare (a `npts=12` smoke first, then the full 288-point day).

The GNN auto-detects CUDA and runs on the GPU automatically.

## What is in git vs what you must upload

**Shipped in the repo (cloned automatically):**
- All code modules in the import closure (the `nonunique_*` entry points,
  `run_da_gps_daily_opendss_compare.py`, the `train_*`/`compare_*`/`run_*` helpers).
- Grid folder `8500 nodes with solar unbalanced/` (DSS + shape/irradiance CSVs).
- Reference profiles `a representativ days/load_day_004.csv`, `irr_day_004.csv`,
  `battery_arbitrage_der_injection.csv`.
- Edge catalog `datasets_gnn2_from pc/gnn_edges_phase_static.csv`.
- MV↔sx mapping `8500-node/mv_x_sx_node_mapping_8500.csv`.
- Model checkpoint (small, ~9 MB) in
  `gnn2_architecture_search/attention checkpoints/da_gps_chunked_l4_mvagg_gine_metaaux_regce_20260516_225149_CCE/`:
  `training_last.pt`, `x_mean.pt`, `x_std.pt`, `y_mean.pt`, `y_std.pt`,
  `reg_mean.pt`, `reg_std.pt`, `reg_class_values.pt`, `reg_class_tables.json`,
  `pv_mean.pt`, `pv_std.pt`.
- **The DA-GPS tensor cache** — the *slim* single-sample build
  `datasets_gnn2_from pc/run_001_ref0_slim__full__nobess__regce__mauxb7bd1d58.pt`
  (~0.28 MB). This is the default `DA_GPS_CACHE_PT` and is bit-identical to the
  old 359 MB full cache for daily-compare inference (which only reads
  `ref_sample_index=0`). **Nothing to upload.**

**Optional / legacy (NOT required for the daily compare):**
- The full multi-sample cache (~359 MB):
  `datasets_gnn2_from pc/run_001_scen_0000_0049_seed_20420233__full__nobess__regce__mauxb7bd1d58.pt`.
  Only needed if you want all samples (e.g. training/eval). To fetch it: upload to
  Google Drive, Share > "Anyone with the link", copy the file ID from
  `https://drive.google.com/file/d/<FILE_ID>/view`, and paste it into the
  `CACHE_DRIVE_FILE_ID` variable in `nonunique_colab.ipynb` (section 4). Leaving it
  empty (`""`) skips the download. A commented mount-Drive cell is also provided.

To regenerate the slim cache from a full cache locally:
`python make_slim_da_gps_cache.py` (defaults read the full cache and write the slim
one; use `--in/--out/--ref` to override).

## Notes / manual steps
- **Private repo:** if `alitasavori/GNN-Sandia` is private, replace the clone URL with a
  token form: `https://<USER>:<TOKEN>@github.com/alitasavori/GNN-Sandia.git`.
- **PyG install:** this model uses only `torch_geometric` core (`GINEConv`, `Data`,
  `to_dense_batch`), so no `torch-scatter`/`torch-sparse` wheels are required.
- **Local Windows runs are unaffected:** paths fall back to the module directory when
  `GNN2_REPO_ROOT` is unset.
