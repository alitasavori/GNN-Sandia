"""PowerFlowMultiNet — oracle device-state baseline (GENConv / DeeperGCN).

Aligned with arXiv:2403.00892v3 where practical — NOT a full paper reproduction.
  - node / edge / state MLPs → L× GENConv (powermean, learn_p, msg_norm, learn_msg_scale)
  - residual DeepGCNLayer (plain then res+)
  - node features: P/Q per phase only (paper II-A)
  - caps/switches: state MLP only; concat with graph embedding for substation P/Q (Fig. 2)
  - bus V/φ readout from node embedding ``h`` (no state concat — paper Fig. 2 ties
    state concat to the substation FC)

Implementation choices (paper silent on exact sizes):
  - hidden=128, num_layers=12 (unified across ieee34 / 906 / 8500; paper cases are
    IEEE 13 / 123 / 906 — not ieee34 or 8500)
  - voltage head (paper): Linear(h → 6); substation head: [pool(h) ‖ g_s] → Linear → 6

Legacy checkpoints (trained before paper-head alignment) use
``voltage_head_mode="state_concat_mlp"``: V/φ head sees ``[h ‖ g_s]`` via a small MLP.
Method A / eval infer this from state_dict keys.

Oracle framing: settled regulator taps and capacitor states are *inputs*
(not predicted — contrast with DA-GPS). Joint MSE on V/φ + substation P/Q
is controlled by ``lambda_sub`` in the trainer (default 1.0).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch_geometric.nn import DeepGCNLayer, GENConv, LayerNorm, global_mean_pool


def _mlp2(in_dim: int, hidden: int, out_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def infer_voltage_head_mode(state: Mapping[str, Any] | None) -> str:
    """Return ``state_concat_mlp`` or ``linear`` from checkpoint keys."""
    if not state:
        return "linear"
    keys = set(state.keys())
    if any(k.startswith("voltage_head.0.") for k in keys) or "voltage_head.0.weight" in keys:
        return "state_concat_mlp"
    return "linear"


class PowerFlowMultiNet(nn.Module):
    """Physical-bus multigraph GENConv model with oracle tap/cap inputs."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        state_dim: int,
        *,
        hidden: int = 128,
        num_layers: int = 12,
        dropout: float = 0.1,
        predict_substation: bool = True,
        voltage_head_mode: str = "linear",
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.predict_substation = bool(predict_substation)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.state_dim = int(state_dim)
        mode = str(voltage_head_mode or "linear").strip().lower()
        if mode in ("legacy", "state_concat", "mlp", "state_concat_mlp"):
            mode = "state_concat_mlp"
        elif mode in ("linear", "paper"):
            mode = "linear"
        else:
            raise ValueError(f"Unknown voltage_head_mode={voltage_head_mode!r}")
        self.voltage_head_mode = mode

        self.node_encoder = _mlp2(node_dim, hidden, hidden, dropout=dropout)
        self.edge_encoder = _mlp2(edge_dim, hidden, hidden, dropout=dropout)
        # State vector may be empty on some feeders; keep a tiny dummy dim of 1.
        state_in = max(1, int(state_dim))
        self.state_encoder = nn.Sequential(
            nn.Linear(state_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            conv = GENConv(
                hidden,
                hidden,
                aggr="powermean",
                learn_p=True,
                msg_norm=True,
                learn_msg_scale=True,
                norm="layer",
                num_layers=2,
            )
            norm = LayerNorm(hidden, affine=True)
            act = nn.ReLU(inplace=True)
            # First layer: plain; subsequent: residual res+ (DeeperGCN style).
            block = "res+" if i > 0 else "plain"
            self.layers.append(
                DeepGCNLayer(conv, norm, act, block=block, dropout=dropout, ckpt_grad=False)
            )

        self.substation_head: nn.Module | None
        if mode == "state_concat_mlp":
            # Legacy: V/φ from [h || g_s]; MLP heads (matches early oracle ckpts).
            self.voltage_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 6),
            )
            if self.predict_substation:
                self.substation_head = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 6),
                )
            else:
                self.substation_head = None
        else:
            # Paper: bus V/φ from node embedding only; Linear substation head.
            self.voltage_head = nn.Linear(hidden, 6)
            if self.predict_substation:
                self.substation_head = nn.Linear(hidden * 2, 6)
            else:
                self.substation_head = None

    def _encode_state(self, device_state: torch.Tensor) -> torch.Tensor:
        if device_state.numel() == 0 or self.state_dim <= 0:
            # [B, 1] zeros → state MLP with state_in=1
            b = device_state.size(0) if device_state.dim() >= 1 and device_state.size(0) > 0 else 1
            device_state = device_state.new_zeros((b, 1))
        elif device_state.size(-1) == 0:
            device_state = device_state.new_zeros((*device_state.shape[:-1], 1))
        return self.state_encoder(device_state)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        device_state: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(y_voltage [N,6], y_substation [B,6] | None)``."""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)

        y_sub = None
        if self.voltage_head_mode == "state_concat_mlp":
            g_s = self._encode_state(device_state)
            if g_s.dim() == 1:
                g_s = g_s.unsqueeze(0)
            g_nodes = g_s[batch]
            y_v = self.voltage_head(torch.cat([h, g_nodes], dim=-1))
            if self.substation_head is not None:
                g_graph = global_mean_pool(h, batch)
                y_sub = self.substation_head(torch.cat([g_graph, g_s], dim=-1))
        else:
            y_v = self.voltage_head(h)
            if self.substation_head is not None:
                g_s = self._encode_state(device_state)
                if g_s.dim() == 1:
                    g_s = g_s.unsqueeze(0)
                g_graph = global_mean_pool(h, batch)
                y_sub = self.substation_head(torch.cat([g_graph, g_s], dim=-1))
        return y_v, y_sub


__all__ = ["PowerFlowMultiNet", "infer_voltage_head_mode"]
