"""Shared helpers for daily compare scripts: CUDA opts, compile, CUDA Graphs, batched inference."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def _default_gnn_torch_compile() -> str:
    """Default compile **off** (CPU and GPU). Set ``GNN_TORCH_COMPILE=1`` to opt in."""
    return "0"


def read_gnn_batch_steps(explicit: int | None = None) -> int:
    """Steps per batched GNN forward (``GNN_BATCH_STEPS`` env or kwarg; default 1 = streaming)."""
    if explicit is not None:
        return max(1, int(explicit))
    return _env_int("GNN_BATCH_STEPS", 1)


def read_gnn_defer_d2h(explicit: bool | None = None) -> bool:
    """Accumulate outputs on device; copy to host once at end (default on CUDA)."""
    if explicit is not None:
        return bool(explicit)
    if os.environ.get("GNN_DEFER_D2H", "").strip():
        return _env_flag("GNN_DEFER_D2H", "0")
    return torch.cuda.is_available()


def read_gnn_cuda_graphs(explicit: bool | None = None, *, device: torch.device) -> bool:
    """CUDA Graph capture for fixed-topology normalize→forward→denorm (CUDA only)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("GNN_CUDA_GRAPHS", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def configure_cuda_inference(device) -> None:
    """Enable safe CUDA matmul/conv speedups for sequential batch-1 inference."""
    if not isinstance(device, torch.device):
        device = torch.device(str(device))
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    tf32_off = os.environ.get("GNN_TF32", "").strip().lower() in ("0", "false", "no", "off")
    if tf32_off:
        torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        print(
            "[GNN] CUDA inference opts: TF32 disabled (GNN_TF32=0, float32_matmul_precision=highest)",
            flush=True,
        )
    else:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        print(
            "[GNN] CUDA inference opts: TF32 matmul (float32_matmul_precision=high), cudnn.benchmark=True",
            flush=True,
        )
    torch.backends.cudnn.benchmark = True


class _CompileOrEager(nn.Module):
    """
    ``torch.compile`` can fail lazily on the first real forward (e.g. Windows without MSVC
    for Inductor). On failure, fall back to eager ``net`` for that call and all later calls.

    On **Windows**, the default Inductor backend invokes ``cl.exe`` for CPU codegen; we use
    ``backend="eager"`` (Dynamo + eager ops, no Inductor C++ build) so compare runs without
    Visual Studio Build Tools.
    """

    def __init__(self, net: nn.Module, label: str, *, compile_mode: str | None = None):
        super().__init__()
        self.net = net
        self._label = label
        if sys.platform == "win32":
            self._compiled = torch.compile(net, backend="eager")  # type: ignore[assignment]
        elif compile_mode:
            self._compiled = torch.compile(net, mode=compile_mode)  # type: ignore[assignment]
        else:
            self._compiled = torch.compile(net)  # type: ignore[assignment]
        self._use_compile = True

    def forward(self, *args, **kwargs):
        if not self._use_compile:
            return self.net(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)  # type: ignore[misc]
        except Exception as e:
            self._use_compile = False
            print(
                f"[{self._label}] torch.compile failed on first forward; using eager mode. "
                f"({type(e).__name__}: {e})  "
                "Set GNN_TORCH_COMPILE=0 to skip compile attempts.",
                flush=True,
            )
            return self.net(*args, **kwargs)


def maybe_torch_compile(model: nn.Module, *, label: str = "GNN", device: torch.device | None = None) -> nn.Module:
    """
    Wrap ``model`` with ``torch.compile`` when PyTorch supports it and
    ``GNN_TORCH_COMPILE`` is not ``0``/``false``.

    Default: **off** everywhere. Set ``GNN_TORCH_COMPILE=1`` to opt in (Inductor on Linux/macOS
    with ``mode='reduce-overhead'`` on CUDA; on Windows uses Dynamo+eager backend).

    First few forwards after compile can be slow (graph capture); timing prints
    still include that overhead unless you add a separate warmup loop.
    """
    v = os.environ.get("GNN_TORCH_COMPILE", _default_gnn_torch_compile()).strip().lower()
    if v in ("0", "false", "no", "off"):
        print(f"[{label}] torch.compile disabled (GNN_TORCH_COMPILE={v!r})")
        return model
    if not hasattr(torch, "compile"):
        print(f"[{label}] torch.compile not available (PyTorch version)")
        return model

    compile_mode: str | None = None
    if device is not None and device.type == "cuda" and sys.platform != "win32":
        compile_mode = "reduce-overhead"

    try:
        out = _CompileOrEager(model, label, compile_mode=compile_mode)
        mode_note = (
            "Windows: Dynamo+eager backend, no MSVC; "
            if sys.platform == "win32"
            else (f"mode={compile_mode!r}; " if compile_mode else "Inductor; ")
        )
        print(
            f"[{label}] torch.compile enabled ({mode_note}"
            "falls back to plain eager on failure) — set GNN_TORCH_COMPILE=0 to disable",
            flush=True,
        )
        return out
    except Exception as e:
        print(f"[{label}] torch.compile skipped: {e}")
        return model


def _block_diag_edge_index(edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    if batch_size <= 1:
        return edge_index
    parts = [edge_index + k * int(n_nodes) for k in range(int(batch_size))]
    return torch.cat(parts, dim=1)


def _block_diag_edge_attr(edge_attr: torch.Tensor, edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    if batch_size <= 1:
        return edge_attr
    n_edges = int(edge_index.shape[1])
    parts = [edge_attr for _ in range(int(batch_size))]
    return torch.cat(parts, dim=0)


@dataclass
class DailyGnnForwardOut:
    pred_mag: torch.Tensor
    pred_ang_deg: torch.Tensor
    cap_act: torch.Tensor | None
    reg_tap: torch.Tensor | None
    pv_dn: torch.Tensor | None


class DailyGnnInferenceRunner:
    """
    Deployment-oriented DA-GPS daily inference: optional CUDA Graphs, batched steps, deferred D2H.

    Static graph topology and normalization tensors are fixed at setup. Per timestep only
    dynamic node features (P/Q, irradiance, DER scalars) change — applied on host then copied
    to ``x_t_dev`` before forward.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        n_nodes: int,
        n_feat: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        x_mean_d: torch.Tensor,
        x_std_d: torch.Tensor,
        y_mean_d: torch.Tensor,
        y_std_d: torch.Tensor,
        reg_mean_d: torch.Tensor | None,
        reg_std_d: torch.Tensor | None,
        pv_mean_d: torch.Tensor | None,
        pv_std_d: torch.Tensor | None,
        reg_loss_mode: str,
        reg_class_values: torch.Tensor | None,
        n_cap: int,
        n_reg: int,
        n_pv: int,
        batch_steps: int = 1,
        defer_d2h: bool | None = None,
        use_cuda_graphs: bool | None = None,
        scatter_li: torch.Tensor | None = None,
        scatter_j: torch.Tensor | None = None,
        n_scatter_cols: int = 0,
    ):
        self.model = model
        self.device = device
        self.n_nodes = int(n_nodes)
        self.n_feat = int(n_feat)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x_mean_d = x_mean_d
        self.x_std_d = x_std_d
        self.y_mean_d = y_mean_d
        self.y_std_d = y_std_d
        self.reg_mean_d = reg_mean_d
        self.reg_std_d = reg_std_d
        self.pv_mean_d = pv_mean_d
        self.pv_std_d = pv_std_d
        self.reg_loss_mode = str(reg_loss_mode)
        self.reg_class_values = reg_class_values
        self.n_cap = int(n_cap)
        self.n_reg = int(n_reg)
        self.n_pv = int(n_pv)
        self.batch_steps = max(1, int(batch_steps))
        self.defer_d2h = read_gnn_defer_d2h(defer_d2h) and device.type == "cuda"
        self.use_cuda_graphs = read_gnn_cuda_graphs(use_cuda_graphs, device=device)
        self.scatter_li = scatter_li
        self.scatter_j = scatter_j
        self.n_scatter_cols = int(n_scatter_cols)

        self.x_t_dev = torch.empty((self.n_nodes, self.n_feat), dtype=torch.float32, device=device)
        self._cuda_graph = None
        self._graph_out: dict[str, torch.Tensor | None] = {}
        self._batched_ei: dict[int, torch.Tensor] = {1: edge_index}
        self._batched_ea: dict[int, torch.Tensor] = {1: edge_attr}

        self._cap_buf: torch.Tensor | None = None
        self._reg_buf: torch.Tensor | None = None
        self._meta_buf: torch.Tensor | None = None
        self._vmag_buf: torch.Tensor | None = None
        self._vang_buf: torch.Tensor | None = None
        self._vmag_scatter_buf: torch.Tensor | None = None
        self._vang_scatter_buf: torch.Tensor | None = None

    def setup(self) -> float:
        """Warmup + optional CUDA Graph capture. Returns wall seconds for one-time setup."""
        import time

        t0 = time.perf_counter()
        if self.batch_steps > 1:
            self._batched_ei[self.batch_steps] = _block_diag_edge_index(
                self.edge_index, self.batch_steps, self.n_nodes
            )
            self._batched_ea[self.batch_steps] = _block_diag_edge_attr(
                self.edge_attr, self.edge_index, self.batch_steps, self.n_nodes
            )

        if self.use_cuda_graphs and self.batch_steps == 1 and not _env_flag("GNN_TORCH_COMPILE", "0"):
            try:
                self._capture_cuda_graph()
                print("[GNN] CUDA Graph capture OK (normalize→model→denorm); replay per step after x_t_dev update", flush=True)
            except Exception as e:
                self.use_cuda_graphs = False
                self._cuda_graph = None
                print(f"[GNN] CUDA Graph capture skipped ({type(e).__name__}: {e}); using eager forward", flush=True)
        elif self.use_cuda_graphs and _env_flag("GNN_TORCH_COMPILE", "0"):
            self.use_cuda_graphs = False
            print("[GNN] CUDA Graphs disabled when GNN_TORCH_COMPILE=1 (incompatible)", flush=True)

        # Eager warmup (also primes compile/cudnn)
        self.x_t_dev.zero_()
        with torch.no_grad():
            if self.batch_steps == 1:
                self._forward_eager_single(sync=True)
            else:
                xb = self.x_t_dev.repeat(self.batch_steps, 1)
                self._forward_eager_batch(xb, batch_size=self.batch_steps, sync=True)

        return time.perf_counter() - t0

    def _capture_cuda_graph(self) -> None:
        assert self.device.type == "cuda"
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._forward_eager_single(sync=False)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        self._graph_out = {}
        with torch.cuda.graph(g):
            x_n = (self.x_t_dev - self.x_mean_d) / self.x_std_d
            data = Data(x=x_n, edge_index=self.edge_index, edge_attr=self.edge_attr)
            out_m = self.model(data)
            volt_n = out_m[0]
            cap_log_b = out_m[1] if len(out_m) > 1 else None
            reg_pred_b = out_m[2] if len(out_m) > 2 else None
            pv_pred_b = out_m[3] if len(out_m) > 3 else None
            v_flat = volt_n.view(1, -1) * self.y_std_d + self.y_mean_d
            pred_ri = v_flat.view(self.n_nodes, 2)
            pred_mag = torch.sqrt(pred_ri[:, 0] ** 2 + pred_ri[:, 1] ** 2 + 1e-12)
            pred_ang = torch.atan2(pred_ri[:, 1], pred_ri[:, 0])
            self._graph_out = {
                "pred_mag": pred_mag,
                "pred_ang": pred_ang,
                "cap_log_b": cap_log_b,
                "reg_pred_b": reg_pred_b,
                "pv_pred_b": pv_pred_b,
            }
        self._cuda_graph = g

    def _denorm_aux(
        self,
        cap_log_b: torch.Tensor | None,
        reg_pred_b: torch.Tensor | None,
        pv_pred_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        cap_act = None
        reg_tap = None
        pv_dn = None
        if cap_log_b is not None and self.n_cap > 0:
            cap_act = torch.sigmoid(cap_log_b.view(-1))
        if reg_pred_b is not None and self.n_reg > 0:
            if self.reg_loss_mode == "ce" and self.reg_class_values is not None:
                from train_da_gps_multitask_complex_voltage_gine import _reg_indices_to_tap_pu

                pred_idx = reg_pred_b.view(1, -1).long()
                tap_pu, _ = _reg_indices_to_tap_pu(pred_idx, pred_idx, self.reg_class_values)
                reg_tap = tap_pu.view(-1)
            elif self.reg_mean_d is not None and self.reg_std_d is not None:
                reg_tap = (reg_pred_b.view(1, -1) * self.reg_std_d + self.reg_mean_d).view(-1)
        if pv_pred_b is not None and self.pv_mean_d is not None and self.pv_std_d is not None and self.n_pv > 0:
            pv_dn = (pv_pred_b.view(1, -1) * self.pv_std_d + self.pv_mean_d).view(-1)
        return cap_act, reg_tap, pv_dn

    def _forward_eager_single(self, *, sync: bool) -> DailyGnnForwardOut:
        x_n = (self.x_t_dev - self.x_mean_d) / self.x_std_d
        data = Data(x=x_n, edge_index=self.edge_index, edge_attr=self.edge_attr)
        out_m = self.model(data)
        volt_n = out_m[0]
        cap_log_b = out_m[1] if len(out_m) > 1 else None
        reg_pred_b = out_m[2] if len(out_m) > 2 else None
        pv_pred_b = out_m[3] if len(out_m) > 3 else None
        if sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        v_flat = volt_n.view(1, -1) * self.y_std_d + self.y_mean_d
        pred_ri = v_flat.view(self.n_nodes, 2)
        pred_mag = torch.sqrt(pred_ri[:, 0] ** 2 + pred_ri[:, 1] ** 2 + 1e-12)
        pred_ang = torch.atan2(pred_ri[:, 1], pred_ri[:, 0])
        cap_act, reg_tap, pv_dn = self._denorm_aux(cap_log_b, reg_pred_b, pv_pred_b)
        return DailyGnnForwardOut(
            pred_mag=pred_mag,
            pred_ang_deg=torch.rad2deg(pred_ang),
            cap_act=cap_act,
            reg_tap=reg_tap,
            pv_dn=pv_dn,
        )

    def _forward_eager_batch(self, x_batch: torch.Tensor, *, batch_size: int, sync: bool) -> list[DailyGnnForwardOut]:
        ei = self._batched_ei[batch_size]
        ea = self._batched_ea[batch_size]
        x_n = (x_batch - self.x_mean_d) / self.x_std_d
        bptr = torch.arange(batch_size, device=self.device, dtype=torch.long).repeat_interleave(self.n_nodes)
        data = Data(x=x_n, edge_index=ei, edge_attr=ea, batch=bptr, num_graphs=batch_size)
        out_m = self.model(data)
        volt_n = out_m[0]
        cap_log_b = out_m[1] if len(out_m) > 1 else None
        reg_pred_b = out_m[2] if len(out_m) > 2 else None
        pv_pred_b = out_m[3] if len(out_m) > 3 else None
        if sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        v_flat = volt_n.view(batch_size, -1) * self.y_std_d + self.y_mean_d
        pred_ri = v_flat.view(batch_size, self.n_nodes, 2)
        pred_mag = torch.sqrt(pred_ri[..., 0] ** 2 + pred_ri[..., 1] ** 2 + 1e-12)
        pred_ang = torch.rad2deg(torch.atan2(pred_ri[..., 1], pred_ri[..., 0]))
        outs: list[DailyGnnForwardOut] = []
        for k in range(batch_size):
            cap_k = cap_log_b[k : k + 1] if cap_log_b is not None and self.n_cap > 0 else None
            reg_k = reg_pred_b[k : k + 1] if reg_pred_b is not None and self.n_reg > 0 else None
            pv_k = pv_pred_b[k : k + 1] if pv_pred_b is not None and self.n_pv > 0 else None
            cap_act, reg_tap, pv_dn = self._denorm_aux(cap_k, reg_k, pv_k)
            outs.append(
                DailyGnnForwardOut(
                    pred_mag=pred_mag[k],
                    pred_ang_deg=pred_ang[k],
                    cap_act=cap_act,
                    reg_tap=reg_tap,
                    pv_dn=pv_dn,
                )
            )
        return outs

    def copy_host_features(self, x_torch_host: torch.Tensor) -> None:
        self.x_t_dev.copy_(x_torch_host, non_blocking=(self.device.type == "cuda"))

    def forward_single(self, *, sync_forward: bool = True) -> DailyGnnForwardOut:
        with torch.no_grad():
            if self.use_cuda_graphs and self._cuda_graph is not None:
                self._cuda_graph.replay()
                if sync_forward and self.device.type == "cuda":
                    torch.cuda.synchronize()
                go = self._graph_out
                cap_act, reg_tap, pv_dn = self._denorm_aux(go.get("cap_log_b"), go.get("reg_pred_b"), go.get("pv_pred_b"))
                return DailyGnnForwardOut(
                    pred_mag=go["pred_mag"],  # type: ignore[index]
                    pred_ang_deg=torch.rad2deg(go["pred_ang"]),  # type: ignore[index]
                    cap_act=cap_act,
                    reg_tap=reg_tap,
                    pv_dn=pv_dn,
                )
            return self._forward_eager_single(sync=sync_forward)

    def forward_batch(self, x_hosts: list[torch.Tensor], *, sync_forward: bool = True) -> list[DailyGnnForwardOut]:
        with torch.no_grad():
            b = len(x_hosts)
            if b == 1:
                self.copy_host_features(x_hosts[0])
                return [self.forward_single(sync_forward=sync_forward)]
            xb = torch.empty((b * self.n_nodes, self.n_feat), dtype=torch.float32, device=self.device)
            for k, xh in enumerate(x_hosts):
                sl = slice(k * self.n_nodes, (k + 1) * self.n_nodes)
                xb[sl].copy_(xh, non_blocking=(self.device.type == "cuda"))
            return self._forward_eager_batch(xb, batch_size=b, sync=sync_forward)

    def alloc_deferred_buffers(self, n_steps: int) -> None:
        if not self.defer_d2h:
            return
        self._vmag_buf = torch.empty((n_steps, self.n_nodes), dtype=torch.float32, device=self.device)
        self._vang_buf = torch.empty((n_steps, self.n_nodes), dtype=torch.float32, device=self.device)
        if self.n_cap > 0:
            self._cap_buf = torch.empty((n_steps, self.n_cap), dtype=torch.float32, device=self.device)
        if self.n_reg > 0:
            self._reg_buf = torch.empty((n_steps, self.n_reg), dtype=torch.float32, device=self.device)
        if self.n_pv > 0:
            self._meta_buf = torch.empty((n_steps, self.n_pv), dtype=torch.float32, device=self.device)
        if self.scatter_li is not None and self.scatter_j is not None and self.n_scatter_cols > 0:
            self._vmag_scatter_buf = torch.empty((n_steps, self.n_scatter_cols), dtype=torch.float32, device=self.device)
            self._vang_scatter_buf = torch.empty((n_steps, self.n_scatter_cols), dtype=torch.float32, device=self.device)

    def store_step(self, step_i: int, out: DailyGnnForwardOut) -> None:
        if not self.defer_d2h:
            return
        assert self._vmag_buf is not None and self._vang_buf is not None
        self._vmag_buf[step_i].copy_(out.pred_mag)
        self._vang_buf[step_i].copy_(out.pred_ang_deg)
        if out.cap_act is not None and self._cap_buf is not None:
            n = min(self.n_cap, int(out.cap_act.numel()))
            self._cap_buf[step_i, :n] = out.cap_act[:n]
        if out.reg_tap is not None and self._reg_buf is not None:
            n = min(self.n_reg, int(out.reg_tap.numel()))
            self._reg_buf[step_i, :n] = out.reg_tap[:n]
        if out.pv_dn is not None and self._meta_buf is not None:
            n = min(self.n_pv, int(out.pv_dn.numel()))
            self._meta_buf[step_i, :n] = out.pv_dn[:n]
        if self.scatter_li is not None and self._vmag_scatter_buf is not None and self._vang_scatter_buf is not None:
            self._vmag_scatter_buf[step_i] = out.pred_mag.index_select(0, self.scatter_li)
            self._vang_scatter_buf[step_i] = out.pred_ang_deg.index_select(0, self.scatter_li)

    def finalize_deferred(
        self,
        v_gnn: Any,
        va_gnn: Any,
        cap_gnn_prob: Any,
        reg_gnn_tap: Any,
        meta_gnn: Any,
        scatter_j_np: Any,
    ) -> None:
        """Single (or chunked) D2H copy after all steps."""
        import numpy as np

        if not self.defer_d2h:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        assert self._vmag_buf is not None and self._vang_buf is not None
        if self._vmag_scatter_buf is not None and self._vang_scatter_buf is not None and scatter_j_np is not None:
            vm = self._vmag_scatter_buf.detach().cpu().numpy()
            va = self._vang_scatter_buf.detach().cpu().numpy()
            v_gnn[:, scatter_j_np] = vm
            va_gnn[:, scatter_j_np] = va
        else:
            vm = self._vmag_buf.detach().cpu().numpy()
            va = self._vang_buf.detach().cpu().numpy()
            n_cols = min(v_gnn.shape[1], vm.shape[1])
            v_gnn[:, :n_cols] = vm[:, :n_cols]
            va_gnn[:, :n_cols] = va[:, :n_cols]
        if self._cap_buf is not None and cap_gnn_prob is not None:
            cap_gnn_prob[:, :] = self._cap_buf.detach().cpu().numpy()
        if self._reg_buf is not None and reg_gnn_tap is not None:
            reg_gnn_tap[:, :] = self._reg_buf.detach().cpu().numpy()
        if self._meta_buf is not None and meta_gnn is not None:
            meta_gnn[:, :] = self._meta_buf.detach().cpu().numpy()

    def write_step_host(
        self,
        step_i: int,
        out: DailyGnnForwardOut,
        *,
        v_gnn: Any,
        va_gnn: Any,
        cap_gnn_prob: Any,
        reg_gnn_tap: Any,
        meta_gnn: Any,
        scatter_j_np: Any,
        scatter_li_np: Any,
        n_cap_plot: int,
        n_reg_plot: int,
        n_pv_plot: int,
    ) -> None:
        """Streaming D2H path when defer_d2h is off."""
        import numpy as np

        pred_np = out.pred_mag.detach().cpu().numpy().astype(np.float32)
        pred_ang_deg = out.pred_ang_deg.detach().cpu().numpy().astype(np.float32)
        if scatter_li_np is not None and scatter_j_np is not None:
            v_gnn[step_i, scatter_j_np] = pred_np[scatter_li_np]
            va_gnn[step_i, scatter_j_np] = pred_ang_deg[scatter_li_np]
        else:
            n_cols = min(v_gnn.shape[1], pred_np.shape[0])
            v_gnn[step_i, :n_cols] = pred_np[:n_cols]
            va_gnn[step_i, :n_cols] = pred_ang_deg[:n_cols]
        if cap_gnn_prob is not None and out.cap_act is not None and n_cap_plot > 0:
            cap_act = out.cap_act.detach().cpu().numpy().reshape(-1)
            for jc in range(n_cap_plot):
                cap_gnn_prob[step_i, jc] = float(cap_act[jc]) if jc < int(cap_act.shape[0]) else np.nan
        if reg_gnn_tap is not None and out.reg_tap is not None and n_reg_plot > 0:
            reg_dn = out.reg_tap.detach().cpu().numpy().reshape(-1)
            for jr in range(n_reg_plot):
                reg_gnn_tap[step_i, jr] = float(reg_dn[jr]) if jr < int(reg_dn.shape[0]) else np.nan
        if meta_gnn is not None and out.pv_dn is not None and n_pv_plot > 0:
            pv_dn = out.pv_dn.detach().cpu().numpy().reshape(-1)
            for jm in range(n_pv_plot):
                meta_gnn[step_i, jm] = float(pv_dn[jm]) if jm < int(pv_dn.shape[0]) else np.nan


def build_scatter_indices(
    node_order: list[str],
    node_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, "np.ndarray", "np.ndarray"]:
    """Precompute gather/scatter indices (replaces per-step Python node loop)."""
    import numpy as np

    scatter_li: list[int] = []
    scatter_j: list[int] = []
    for li, name in enumerate(node_order):
        nk = str(name).strip().lower()
        j = node_to_idx.get(nk)
        if j is not None:
            scatter_li.append(int(li))
            scatter_j.append(int(j))
    li_t = torch.tensor(scatter_li, dtype=torch.long, device=device)
    li_np = np.asarray(scatter_li, dtype=np.int64)
    j_np = np.asarray(scatter_j, dtype=np.int32)
    return li_t, j_np, li_np
