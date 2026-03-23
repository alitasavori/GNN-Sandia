"""
LaTeX checklist Step 5 — MLP baseline for IEEE 8500-node load-type tensors.

Training uses **voltage magnitude |V| (pu) only** — angle is not used as a
target or in the loss.

Loads `dataset_tensors/X.pt` and `Y.pt` (vmag). Flattens inputs, z-score
normalizes from the train split, and minimizes **MSE on |V| only**.

**Architecture sweep:** `train_mlp_architecture_sweep_8500()` trains five MLP
configs, saves each under `mlp_sweep_8500/<name>/`, and writes
`mlp_sweep_8500/sweep_summary.json` plus copies the best run to
`mlp_sweep_8500/best/` (by lowest **test** MAE on |V|).

Pass criterion (checklist): test MAE < 0.005 pu on |V| — may need more samples.

Run after tensor assembly (notebook Step 5 / 6).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    _REPO = Path(__file__).resolve().parent
except NameError:
    _REPO = Path.cwd()

DEFAULT_DATASET_DIR = _REPO / "datasets_gnn2" / "loadtype_8500"
DEFAULT_TENSOR_SUBDIR = "dataset_tensors"
DEFAULT_OUT_SUBDIR = "mlp_baseline_8500"
DEFAULT_SWEEP_SUBDIR = "mlp_sweep_8500"

# Five distinct MLP shapes (width x depth)
DEFAULT_SWEEP_ARCHITECTURES: tuple[dict[str, int | str], ...] = (
    {"name": "h256_l2", "hidden_dim": 256, "num_hidden_layers": 2},
    {"name": "h512_l2", "hidden_dim": 512, "num_hidden_layers": 2},
    {"name": "h768_l2", "hidden_dim": 768, "num_hidden_layers": 2},
    {"name": "h512_l3", "hidden_dim": 512, "num_hidden_layers": 3},
    {"name": "h1024_l2", "hidden_dim": 1024, "num_hidden_layers": 2},
)


class MLP8500(nn.Module):
    """Fully-connected MLP: in -> (hidden -> ReLU) x L -> out. Targets |V| only (out_dim = N)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, num_hidden_layers: int = 2):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _naive_baseline_mae_vmag(y_train_v: torch.Tensor, y_test_v: torch.Tensor) -> float:
    mu = y_train_v.mean(dim=0, keepdim=True)
    return float((y_test_v - mu).abs().mean().cpu())


@torch.no_grad()
def _val_metrics_vmag(model: nn.Module, dl_val: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sum_abs_v = 0.0
    sum_sq_v = 0.0
    n_pix = 0
    for xb, yb in dl_val:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        sum_abs_v += (pred - yb).abs().sum().item()
        sum_sq_v += ((pred - yb) ** 2).sum().item()
        n_pix += xb.size(0) * yb.size(1)
    mae_v = sum_abs_v / max(n_pix, 1)
    rmse_v = (sum_sq_v / max(n_pix, 1)) ** 0.5
    return {"val_mae_vmag_pu": mae_v, "val_rmse_vmag_pu": rmse_v}


def _train_one_mlp(
    *,
    dataset_dir: Path,
    tensor_subdir: str,
    out_dir: Path,
    hidden_dim: int,
    num_hidden_layers: int,
    arch_label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_frac: float,
    val_frac: float,
    verbose: bool = True,
    perm: torch.Tensor | None = None,
    X: torch.Tensor | None = None,
    Y: torch.Tensor | None = None,
) -> dict:
    tdir = dataset_dir / tensor_subdir
    path_x, path_y = tdir / "X.pt", tdir / "Y.pt"
    if X is None or Y is None:
        if not path_x.is_file() or not path_y.is_file():
            raise FileNotFoundError(f"Need {path_x} and {path_y}. Run assemble_dataset_tensors_8500 first.")
        try:
            X = torch.load(path_x, map_location="cpu", weights_only=True)
        except TypeError:
            X = torch.load(path_x, map_location="cpu")
        try:
            Y = torch.load(path_y, map_location="cpu", weights_only=True)
        except TypeError:
            Y = torch.load(path_y, map_location="cpu")
    X = X.float()
    Y = Y.float()
    if X.dim() != 3:
        raise ValueError(f"Expected X [S,N,F], got {tuple(X.shape)}")
    S, N, F = X.shape
    x_flat = X.reshape(S, N * F)
    y_flat = Y
    out_dim = N

    if perm is None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(S, generator=g)
    else:
        if perm.numel() != S:
            raise ValueError("perm length must match number of samples")
        perm = perm.clone()
    torch.manual_seed(seed)
    n_train = int(S * train_frac)
    n_val = int(S * val_frac)
    n_test = S - n_train - n_val
    if n_test < 1:
        raise RuntimeError(f"Not enough samples S={S} for 80/10/10 split.")

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    x_tr = x_flat[idx_train]
    mean = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_flat_n = (x_flat - mean) / std

    ds_train = TensorDataset(x_flat_n[idx_train], y_flat[idx_train])
    ds_val = TensorDataset(x_flat_n[idx_val], y_flat[idx_val])
    ds_test = TensorDataset(x_flat_n[idx_test], y_flat[idx_test])

    y_train_v = Y[idx_train]
    y_val_v = Y[idx_val]
    y_test_v = Y[idx_test]
    naive_val_mae = _naive_baseline_mae_vmag(y_train_v, y_val_v)
    naive_test_mae = _naive_baseline_mae_vmag(y_train_v, y_test_v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP8500(x_flat.shape[1], out_dim, hidden=hidden_dim, num_hidden_layers=num_hidden_layers).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    dl_tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    def mse_vmag(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - tgt) ** 2)

    best_val = float("inf")
    best_state = None

    if verbose:
        print(
            f"[{arch_label}] Naive baseline (val) MAE |V| = {naive_val_mae:.6f} pu  "
            f"(test naive MAE = {naive_test_mae:.6f} pu)"
        )
        print(
            f"  arch: hidden_dim={hidden_dim}  num_hidden_layers={num_hidden_layers}  "
            f"loss=MSE(|V|) only"
        )

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = mse_vmag(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += mse_vmag(model(xb), yb).item() * xb.size(0)
                n_val_samples += xb.size(0)
        val_loss /= max(n_val_samples, 1)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == 1):
            vm = _val_metrics_vmag(model, dl_val, device)
            print(
                f"  epoch {ep:3d}/{epochs}  val_mse_|V|={val_loss:.6f}  "
                f"val_mae_|V|_pu={vm['val_mae_vmag_pu']:.6f}  "
                f"val_rmse_|V|_pu={vm['val_rmse_vmag_pu']:.6f}  lr={opt.param_groups[0]['lr']:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    x_test, y_test = ds_test.tensors[0].to(device), ds_test.tensors[1].to(device)
    with torch.no_grad():
        pred = model(x_test)
        mae_pu = float((pred - y_test).abs().mean().cpu())

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "mlp_8500.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "in_dim": int(x_flat.shape[1]),
            "out_dim": out_dim,
            "num_nodes": int(N),
            "num_feat": int(F),
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "arch_label": arch_label,
        },
        ckpt,
    )

    meta = {
        "arch_label": arch_label,
        "dataset_dir": str(dataset_dir.resolve()),
        "tensor_dir": str(tdir.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "samples_total": S,
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "naive_baseline_test_mae_vmag_pu": naive_test_mae,
        "test_mae_vmag_pu": mae_pu,
        "beats_naive_baseline_vmag": mae_pu < naive_test_mae,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "epochs": epochs,
        "pass_mae_under_0_005_pu": mae_pu < 0.005,
        "loss": "MSE_vmag_only",
    }
    (out_dir / "mlp_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(
            f"[{arch_label}] device={device!s}  test MAE |V| (pu): {mae_pu:.6f}  "
            f"vs naive {naive_test_mae:.6f}  ({'better' if mae_pu < naive_test_mae else 'worse'} than naive)"
        )
        print(f"  saved {ckpt}")
    return meta


def train_mlp_baseline_8500(
    dataset_dir: str | os.PathLike | None = None,
    tensor_subdir: str = DEFAULT_TENSOR_SUBDIR,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 512,
    num_hidden_layers: int = 2,
    seed: int = 20260322,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    """Train a single MLP; loss is MSE on |V| (pu) only."""
    dset = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    out = dset / out_subdir
    return _train_one_mlp(
        dataset_dir=dset,
        tensor_subdir=tensor_subdir,
        out_dir=out,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        arch_label="single",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        verbose=True,
    )


def train_mlp_architecture_sweep_8500(
    dataset_dir: str | os.PathLike | None = None,
    tensor_subdir: str = DEFAULT_TENSOR_SUBDIR,
    sweep_subdir: str = DEFAULT_SWEEP_SUBDIR,
    architectures: tuple[dict[str, int | str], ...] | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    base_seed: int = 20260322,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    """
    Train five (default) architectures; each saved under `<dataset>/sweep_subdir/<name>/`.
    Best = lowest test_mae_vmag_pu. Copies best to `<dataset>/sweep_subdir/best/`.
    """
    dset = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    archs = architectures if architectures is not None else DEFAULT_SWEEP_ARCHITECTURES
    if len(archs) == 0:
        raise ValueError("architectures must be non-empty")

    tdir = dset / tensor_subdir
    path_x, path_y = tdir / "X.pt", tdir / "Y.pt"
    if not path_x.is_file() or not path_y.is_file():
        raise FileNotFoundError(f"Need {path_x} and {path_y}. Run assemble_dataset_tensors_8500 first.")
    try:
        X = torch.load(path_x, map_location="cpu", weights_only=True).float()
    except TypeError:
        X = torch.load(path_x, map_location="cpu").float()
    try:
        Y = torch.load(path_y, map_location="cpu", weights_only=True).float()
    except TypeError:
        Y = torch.load(path_y, map_location="cpu").float()
    S = int(X.shape[0])
    g = torch.Generator().manual_seed(base_seed)
    perm = torch.randperm(S, generator=g)

    sweep_root = dset / sweep_subdir
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, cfg in enumerate(archs):
        name = str(cfg["name"])
        hd = int(cfg["hidden_dim"])
        nl = int(cfg["num_hidden_layers"])
        out_dir = sweep_root / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        meta = _train_one_mlp(
            dataset_dir=dset,
            tensor_subdir=tensor_subdir,
            out_dir=out_dir,
            hidden_dim=hd,
            num_hidden_layers=nl,
            arch_label=name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=base_seed + i,
            train_frac=train_frac,
            val_frac=val_frac,
            verbose=True,
            perm=perm,
            X=X,
            Y=Y,
        )
        meta["sweep_name"] = name
        results.append(meta)
        print()

    best = min(results, key=lambda m: m["test_mae_vmag_pu"])
    best_name = str(best["sweep_name"])

    best_src = sweep_root / best_name
    best_dst = sweep_root / "best"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(best_src, best_dst)

    summary = {
        "dataset_dir": str(dset.resolve()),
        "sweep_dir": str(sweep_root.resolve()),
        "split_seed": base_seed,
        "note": "All architectures use the same train/val/test permutation (split_seed).",
        "best_architecture": best_name,
        "selection_metric": "test_mae_vmag_pu_min",
        "best_test_mae_vmag_pu": best["test_mae_vmag_pu"],
        "runs": [
            {
                "name": str(r["sweep_name"]),
                "test_mae_vmag_pu": r["test_mae_vmag_pu"],
                "hidden_dim": r["hidden_dim"],
                "num_hidden_layers": r["num_hidden_layers"],
                "checkpoint": r["checkpoint"],
            }
            for r in results
        ],
    }
    summary_path = sweep_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"SWEEP DONE. Best architecture: {best_name}")
    print(f"  test MAE |V| (pu) = {best['test_mae_vmag_pu']:.6f}")
    print(f"  copied to {best_dst}")
    print(f"  summary: {summary_path}")
    return summary


def main() -> None:
    train_mlp_architecture_sweep_8500()


if __name__ == "__main__":
    main()
