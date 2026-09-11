"""Fast online Log(v) 3LPF solver: init once, cheap repeated sequential solves.

The intended regime here is sequential scenarios (one PF at a time):
  one expensive initialization + many single-RHS solves on a fixed feeder.

The stock ``rank_k_correction_solve`` path re-runs ``get_loads`` (DataFrame /
Python loops) and dictionary voltage mapping every snapshot — that dominates
wall-clock (e.g. ~170 ms of ~230 ms on IEEE 8500).

This module:
  * precomputes the const-P injection map and fixed ZIP/cap contributions;
  * reuses the sparse factorization of A (and Woodbury constants when k>0);
  * provides a dedicated k=0 path and optional dense H = A^{-1} G for single RHS;
  * returns contiguous NumPy arrays (no bus-name dicts) in the hot path;
  * benchmarks sequential single-snapshot latency vs OpenDSS (no multi-RHS batch).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class StageTimes:
    rhs: float = 0.0
    base_solve: float = 0.0
    woodbury: float = 0.0
    recover: float = 0.0

    @property
    def total(self) -> float:
        return self.rhs + self.base_solve + self.woodbury + self.recover


class FastLogvSolver:
    """Precomputed online Log(v) solver for repeated sequential scenarios.

    Online inputs are load (and optional PV-as-load) active/reactive power in kW/kvar,
    ordered as the const-P loads in ``case.loads`` (model==1), shape ``(n_cp,)``.
    Primary API is ``solve(P, Q)`` — one scenario at a time.
    """

    def __init__(self, case, *, build_dense_H: bool | None = None, dtype=np.float64):
        import logv3lpf.linpf as linpf

        self.dtype = np.dtype(dtype)
        self.Nn = int(case.Nn)
        self.mask = np.asarray(case.mask, dtype=np.int64)
        self.n_sys = int(len(self.mask))
        self.n_unk = self.n_sys // 2  # log|V| unknowns (= angle unknowns)
        self.solve_A = case._solve_A
        self.sdrop = np.asarray(case.sdrop, dtype=self.dtype).reshape(-1)
        self.sshunt = np.asarray(case.sshunt, dtype=self.dtype).reshape(-1)
        self.fixed_bias = self.sdrop + self.sshunt  # subtracted from RHS

        # ---- const-P injection map (and freeze ZIP/cap L, s_fixed) ----
        linpf.get_loads(case)
        self._build_const_p_map(case)
        self._build_fixed_zip_cap(case)

        # RHS map: b = G_p @ P + G_q @ Q + b0  (length n_sys)
        # G_* are sparse (n_sys x n_cp)
        self.G_p, self.G_q, self.b0 = self._build_rhs_maps(case)
        self._use_fd_s = False
        self._ds_dP = self._ds_dQ = None
        if self._has_delta_cp:
            # Delta const-P in get_loads expands via Omega; simple wye column maps
            # are wrong on IEEE 34. Build an exact affine s(P,Q) by FD once.
            self._build_fd_const_p_maps(case)
            self._use_fd_s = True
            self.G_p, self.G_q, self.b0 = self._rhs_maps_from_ds(case)

        k = int(case.U.shape[1]) if getattr(case, "U", None) is not None else 0
        self.k = k
        self.use_woodbury = k > 0 and not isinstance(getattr(case, "Lu", None), list)

        if self.use_woodbury:
            self.AinvU = np.asarray(case.AinvU, dtype=self.dtype)
            self.VAinv = np.asarray(case.VAinv, dtype=self.dtype)
            self.VAinvU = np.asarray(case.VAinvU, dtype=self.dtype)
            self.L = self._assemble_L(case)
            self.M = np.asarray(np.eye(self.L.shape[0]) + self.L @ self.VAinvU, dtype=self.dtype)
            # Factor small Woodbury matrix once (L fixed when only const-P P/Q change).
            self._M_lu = None
            try:
                self._M_lu = np.linalg.lu_factor(self.M)
            except Exception:
                self._M_lu = None
        else:
            self.AinvU = self.VAinv = self.VAinvU = self.L = self.M = None
            self._M_lu = None
            self.k = 0
            self.use_woodbury = False

        # Optional dense H: x = H_p @ P + H_q @ Q + h  (includes Woodbury when k>0).
        # This is the sub-ms online path on ieee34-sized feeders; without it, each
        # solve pays a Python Woodbury stack (~1–3 ms) that looks like "slow Fast".
        if build_dense_H is None:
            build_dense_H = (self.n_cp <= 512) and (
                self.n_sys * self.n_cp * 2 * 8 < 2_000_000_000
            )
        self._want_dense_H = bool(build_dense_H)
        self.H_p = self.H_q = self.h = None
        self.has_H = False
        if self._want_dense_H:
            self._build_dense_H()

        # Nominal P,Q for convenience
        self.P0 = self._P0.copy()
        self.Q0 = self._Q0.copy()

        # Output buffers
        self._x = np.zeros(self.n_sys, dtype=self.dtype)
        self._vm = np.zeros(self.n_unk, dtype=self.dtype)
        self._va = np.zeros(self.n_unk, dtype=self.dtype)
        self._b = np.zeros(self.n_sys, dtype=self.dtype)

        # Reference magnitudes (for full bus-phase recovery if needed)
        self.ref_vm = float(getattr(case, "refvm", 1.0))
        self.ref_idx_real = self._ref_real_indices(case)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_const_p_map(self, case) -> None:
        loads = case.loads
        cp_idx = [i for i in range(len(loads)) if int(loads.model[i]) == 1]
        self.cp_idx = np.asarray(cp_idx, dtype=np.int64)
        self.n_cp = int(len(cp_idx))

        # EloadS columns follow ALL loads in order (then capacitors). Const-P
        # slots are not necessarily a contiguous prefix — track absolute columns.
        coeffs = []
        slot_cols = []
        slot_to_cp = []
        P0 = []
        Q0 = []
        col = 0
        cp_local_of_load = {int(i): loc for loc, i in enumerate(cp_idx)}
        for i in range(len(loads)):
            nph = int(loads.nphases[i])
            if int(loads.model[i]) != 1:
                col += nph
                continue
            local = cp_local_of_load[i]
            bus = loads.bus[i]
            base = float(case.base[bus]["VABase"])
            isdelta = bool(loads.isdelta[i])
            P0.append(float(loads.kW[i]))
            Q0.append(float(loads.kvar[i]))
            if isdelta:
                for _ in range(nph):
                    coeffs.append(np.nan + 1j * np.nan)
                    slot_cols.append(col)
                    slot_to_cp.append(local)
                    col += 1
            else:
                c = -(1000.0) / (nph * base)
                for _ in range(nph):
                    coeffs.append(c + 0.0j)
                    slot_cols.append(col)
                    slot_to_cp.append(local)
                    col += 1

        self._cp_coeffs = np.asarray(coeffs, dtype=np.complex128)
        self._slot_cols = np.asarray(slot_cols, dtype=np.int64)
        self._slot_to_cp = np.asarray(slot_to_cp, dtype=np.int64)
        self._P0 = np.asarray(P0, dtype=self.dtype)
        self._Q0 = np.asarray(Q0, dtype=self.dtype)
        self._has_delta_cp = bool(np.any(~np.isfinite(self._cp_coeffs.real)))
        self.n_cp_slots = int(len(coeffs))
        self.EloadS = case.EloadS.tocsr().astype(self.dtype)
        self._n_load_phase_cols = col

    def _build_fixed_zip_cap(self, case) -> None:
        """Freeze s and L contributions from non-const-P loads and capacitors."""
        self._s_nominal = np.asarray(case.s, dtype=np.complex128).reshape(-1)
        s_fixed = self._s_nominal.copy()
        for j, col in enumerate(self._slot_cols):
            if np.isfinite(self._cp_coeffs[j].real):
                s_fixed[int(col)] = 0.0
        self._s_fixed = s_fixed

        Lu, Lth = case.Lu, case.Lθ
        if isinstance(Lu, list) or Lu is None or (hasattr(Lu, "shape") and Lu.shape[0] == 0):
            self._L_fixed = None
        else:
            self._L_fixed = sp.bmat(
                [[Lu.real, Lth.real], [Lu.imag, Lth.imag]]
            ).tocsr().astype(self.dtype)

    def _assemble_L(self, case):
        if self._L_fixed is not None:
            return self._L_fixed.toarray() if sp.issparse(self._L_fixed) else np.asarray(self._L_fixed)
        Lu, Lth = case.Lu, case.Lθ
        L = sp.bmat([[Lu.real, Lth.real], [Lu.imag, Lth.imag]])
        return L.toarray().astype(self.dtype)

    def _build_rhs_maps(self, case):
        """Build sparse G_p, G_q and constant b0 so b = G_p@P + G_q@Q + b0."""
        n_slots_all = int(self.EloadS.shape[1])
        Cp = sp.lil_matrix((n_slots_all, self.n_cp), dtype=self.dtype)
        Cq = sp.lil_matrix((n_slots_all, self.n_cp), dtype=self.dtype)
        for j in range(self.n_cp_slots):
            c = self._cp_coeffs[j]
            if not np.isfinite(c.real):
                continue
            cp = int(self._slot_to_cp[j])
            col = int(self._slot_cols[j])
            Cp[col, cp] = float(c.real)
            Cq[col, cp] = float(c.real)

        Cp = Cp.tocsr()
        Cq = Cq.tocsr()
        Ep = (self.EloadS @ Cp).tocsr()
        Eq = (self.EloadS @ Cq).tocsr()

        Z = sp.csr_matrix((self.Nn, self.n_cp), dtype=self.dtype)
        G_p_full = sp.vstack([Ep, Z], format="csr")
        G_q_full = sp.vstack([Z, Eq], format="csr")
        G_p = G_p_full[self.mask, :]
        G_q = G_q_full[self.mask, :]

        s_fix = self._s_fixed.reshape(-1, 1)
        inj_f = self.EloadS @ s_fix
        b_full_f = np.vstack([inj_f.real, inj_f.imag]).reshape(-1)
        b0 = np.asarray(b_full_f[self.mask], dtype=self.dtype) - self.fixed_bias
        return G_p.astype(self.dtype), G_q.astype(self.dtype), np.asarray(b0, dtype=self.dtype)


    def _build_fd_const_p_maps(self, case) -> None:
        """Exact ds/dP, ds/dQ for const-P loads via one-sided FD on get_loads."""
        import logv3lpf.linpf as linpf

        linpf.get_loads(case)
        s0 = np.asarray(case.s, dtype=np.complex128).reshape(-1)
        n_s = s0.size
        dP = np.zeros((n_s, self.n_cp), dtype=np.complex128)
        dQ = np.zeros((n_s, self.n_cp), dtype=np.complex128)
        eps = 1.0
        for local, i in enumerate(self.cp_idx):
            i = int(i)
            p_nom = float(case.loads.at[i, "kW"])
            q_nom = float(case.loads.at[i, "kvar"])
            case.loads.at[i, "kW"] = p_nom + eps
            linpf.get_loads(case)
            dP[:, local] = (np.asarray(case.s, dtype=np.complex128).reshape(-1) - s0) / eps
            case.loads.at[i, "kW"] = p_nom
            case.loads.at[i, "kvar"] = q_nom + eps
            linpf.get_loads(case)
            dQ[:, local] = (np.asarray(case.s, dtype=np.complex128).reshape(-1) - s0) / eps
            case.loads.at[i, "kvar"] = q_nom
        linpf.get_loads(case)
        self._s0_fd = s0
        self._ds_dP = dP
        self._ds_dQ = dQ

    def _rhs_maps_from_ds(self, case):
        """Build G_p/G_q/b0 from complex ds/dP, ds/dQ maps (affine in P,Q)."""
        dPr, dPi = np.real(self._ds_dP), np.imag(self._ds_dP)
        dQr, dQi = np.real(self._ds_dQ), np.imag(self._ds_dQ)
        Ep_P = np.asarray(self.EloadS @ dPr)
        Ei_P = np.asarray(self.EloadS @ dPi)
        Ep_Q = np.asarray(self.EloadS @ dQr)
        Ei_Q = np.asarray(self.EloadS @ dQi)
        Gp_full = np.vstack([Ep_P, Ei_P])
        Gq_full = np.vstack([Ep_Q, Ei_Q])
        G_p = sp.csr_matrix(Gp_full[self.mask, :], dtype=self.dtype)
        G_q = sp.csr_matrix(Gq_full[self.mask, :], dtype=self.dtype)
        inj = self.EloadS @ self._s0_fd.reshape(-1, 1)
        b_full = np.vstack([inj.real, inj.imag]).reshape(-1)
        b0 = np.asarray(b_full[self.mask], dtype=self.dtype) - self.fixed_bias
        b0 = b0 - np.asarray(G_p @ self._P0 + G_q @ self._Q0, dtype=self.dtype).reshape(-1)
        return G_p, G_q, b0

    def _apply_A_inv(self, b: np.ndarray) -> np.ndarray:
        """Apply ``A^{-1}`` or full Woodbury ``(A+ULV)^{-1}`` to one RHS."""
        b = np.asarray(b, dtype=self.dtype).reshape(-1)
        x0 = np.asarray(self.solve_A(b), dtype=self.dtype).reshape(-1)
        if not self.use_woodbury:
            return x0
        rhs_small = self.L @ (self.VAinv @ b.reshape(-1, 1))
        if self._M_lu is not None:
            w = np.linalg.lu_solve(self._M_lu, rhs_small)
        else:
            w = np.linalg.solve(self.M, rhs_small)
        return x0 - (self.AinvU @ w).reshape(-1)

    def _build_dense_H(self) -> None:
        """Precompute x = H_p P + H_q Q + h via factorized solves of G columns.

        Each column uses the full online operator (sparse factor + Woodbury when
        ``k>0``), so the hot path is only two dense matvecs + ``exp``.
        """
        Hp = np.zeros((self.n_sys, self.n_cp), dtype=self.dtype)
        Hq = np.zeros((self.n_sys, self.n_cp), dtype=self.dtype)
        Gp = self.G_p.tocsc()
        Gq = self.G_q.tocsc()
        for j in range(self.n_cp):
            Hp[:, j] = self._apply_A_inv(Gp[:, j].toarray().ravel())
            Hq[:, j] = self._apply_A_inv(Gq[:, j].toarray().ravel())
        self.h = self._apply_A_inv(self.b0)
        self.H_p, self.H_q = Hp, Hq
        self.has_H = True

    def _ref_real_indices(self, case):
        src = case.sourcebus
        if src not in case.bus_phases:
            for b in case.bus_phases:
                if str(b).lower() == str(src).lower():
                    src = b
                    break
        return [
            case.node_from_name[f"{src}.{ph}"]
            for ph in case.bus_phases[src]
        ]

    # ------------------------------------------------------------------
    # refresh after A / ZIP / cap changes (no FD re-init)
    # ------------------------------------------------------------------
    def refresh_factorization(self, case) -> None:
        """Pull new ``A`` factor / Woodbury mats / shunt bias from ``case``.

        Call after ``calculate_base_matrices`` (tap bake). Does **not** re-run
        ``get_loads`` or the expensive delta-const-P FD maps.
        """
        self.solve_A = case._solve_A
        self.mask = np.asarray(case.mask, dtype=np.int64)
        self.n_sys = int(len(self.mask))
        self.n_unk = self.n_sys // 2
        self.sdrop = np.asarray(case.sdrop, dtype=self.dtype).reshape(-1)
        self.sshunt = np.asarray(case.sshunt, dtype=self.dtype).reshape(-1)
        self.fixed_bias = self.sdrop + self.sshunt

        k = int(case.U.shape[1]) if getattr(case, "U", None) is not None else 0
        self.use_woodbury = k > 0 and not isinstance(getattr(case, "Lu", None), list)
        if self.use_woodbury:
            self.k = k
            self.AinvU = np.asarray(case.AinvU, dtype=self.dtype)
            self.VAinv = np.asarray(case.VAinv, dtype=self.dtype)
            self.VAinvU = np.asarray(case.VAinvU, dtype=self.dtype)
            self.L = self._assemble_L(case)
            self.M = np.asarray(
                np.eye(self.L.shape[0]) + self.L @ self.VAinvU, dtype=self.dtype
            )
            try:
                self._M_lu = np.linalg.lu_factor(self.M)
            except Exception:
                self._M_lu = None
        else:
            self.k = 0
            self.use_woodbury = False
            self.AinvU = self.VAinv = self.VAinvU = self.L = self.M = None
            self._M_lu = None

        self._recompute_b0()
        # Rebuild dense H after factor/ZIP refresh (H includes Woodbury).
        # Never leave a stale H that still points at the previous tap/cap state.
        if getattr(self, "_want_dense_H", False):
            self._build_dense_H()
        else:
            self.has_H = False
            self.H_p = self.H_q = self.h = None
        # Keep output buffers sized correctly if mask length ever changes.
        if self._x.shape[0] != self.n_sys:
            self._x = np.zeros(self.n_sys, dtype=self.dtype)
            self._b = np.zeros(self.n_sys, dtype=self.dtype)
            self._vm = np.zeros(self.n_unk, dtype=self.dtype)
            self._va = np.zeros(self.n_unk, dtype=self.dtype)

    def refresh_zip_cap_and_factor(self, case) -> None:
        """Update frozen ZIP/cap terms + factorization after cap/tap changes.

        Runs ``get_loads`` once (tens of ms). Never re-runs FD const-P maps.
        Must stay **outside** the Fast online-solve timer.
        """
        import logv3lpf.linpf as linpf

        if self._use_fd_s:
            # Refresh s0 at nominal P0,Q0 so cap/ZIP changes enter b0; keep ds/dP.
            saved_p = [float(case.loads.kW[int(i)]) for i in self.cp_idx]
            saved_q = [float(case.loads.kvar[int(i)]) for i in self.cp_idx]
            for loc, i in enumerate(self.cp_idx):
                case.loads.at[int(i), "kW"] = float(self._P0[loc])
                case.loads.at[int(i), "kvar"] = float(self._Q0[loc])
            linpf.get_loads(case)
            self._s0_fd = np.asarray(case.s, dtype=np.complex128).reshape(-1)
            for loc, i in enumerate(self.cp_idx):
                case.loads.at[int(i), "kW"] = saved_p[loc]
                case.loads.at[int(i), "kvar"] = saved_q[loc]
            linpf.get_loads(case)
            self._build_fixed_zip_cap(case)
            self.G_p, self.G_q, self.b0 = self._rhs_maps_from_ds(case)
        else:
            linpf.get_loads(case)
            self._build_fixed_zip_cap(case)
            self.G_p, self.G_q, self.b0 = self._build_rhs_maps(case)

        self.refresh_factorization(case)

    def _recompute_b0(self) -> None:
        if self._use_fd_s:
            inj = self.EloadS @ self._s0_fd.reshape(-1, 1)
            b_full = np.vstack([inj.real, inj.imag]).reshape(-1)
            b0 = np.asarray(b_full[self.mask], dtype=self.dtype) - self.fixed_bias
            b0 = b0 - np.asarray(
                self.G_p @ self._P0 + self.G_q @ self._Q0, dtype=self.dtype
            ).reshape(-1)
            self.b0 = b0
        else:
            s_fix = self._s_fixed.reshape(-1, 1)
            inj_f = self.EloadS @ s_fix
            b_full_f = np.vstack([inj_f.real, inj_f.imag]).reshape(-1)
            self.b0 = (
                np.asarray(b_full_f[self.mask], dtype=self.dtype) - self.fixed_bias
            )

    # ------------------------------------------------------------------
    # online API
    # ------------------------------------------------------------------
    def build_rhs(self, P: np.ndarray | None = None, Q: np.ndarray | None = None) -> np.ndarray:
        P = self.P0 if P is None else np.asarray(P, dtype=self.dtype).reshape(-1)
        Q = self.Q0 if Q is None else np.asarray(Q, dtype=self.dtype).reshape(-1)
        if P.shape[0] != self.n_cp or Q.shape[0] != self.n_cp:
            raise ValueError(f"P,Q must have length n_cp={self.n_cp}")
        b = self.G_p @ P + self.G_q @ Q + self.b0
        return np.asarray(b, dtype=self.dtype).reshape(-1)

    def solve(
        self,
        P: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        *,
        profile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, StageTimes]:
        """Single-scenario online solve. Returns (vm, va) over reduced unknowns.

        Hot path is dense ``H`` when available (no ``get_loads``, no refactor).
        """
        st = StageTimes()
        if self.has_H:
            # Sub-ms path: skip explicit RHS; x = H_p P + H_q Q + h
            t0 = time.perf_counter()
            Pv = self.P0 if P is None else np.asarray(P, dtype=self.dtype).reshape(-1)
            Qv = self.Q0 if Q is None else np.asarray(Q, dtype=self.dtype).reshape(-1)
            if Pv.shape[0] != self.n_cp or Qv.shape[0] != self.n_cp:
                raise ValueError(f"P,Q must have length n_cp={self.n_cp}")
            x = self.H_p @ Pv + self.H_q @ Qv + self.h
            st.base_solve = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            b = self.build_rhs(P, Q)
            st.rhs = time.perf_counter() - t0
            t1 = time.perf_counter()
            x0 = np.asarray(self.solve_A(b), dtype=self.dtype).reshape(-1)
            st.base_solve = time.perf_counter() - t1
            if self.use_woodbury:
                t2 = time.perf_counter()
                rhs_small = self.L @ (self.VAinv @ b.reshape(-1, 1))
                if self._M_lu is not None:
                    w = np.linalg.lu_solve(self._M_lu, rhs_small)
                else:
                    w = np.linalg.solve(self.M, rhs_small)
                x = x0 - (self.AinvU @ w).reshape(-1)
                st.woodbury = time.perf_counter() - t2
            else:
                x = x0

        t3 = time.perf_counter()
        half = self.n_unk
        vm = np.exp(x[:half])
        va = x[half:]
        st.recover = time.perf_counter() - t3
        if profile:
            return vm, va, st
        return vm, va

    def solve_batch(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        *,
        profile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Batch solve. P,Q shape (n_cp, M). Returns vm, va each (n_unk, M)."""
        P = np.asarray(P, dtype=self.dtype)
        Q = np.asarray(Q, dtype=self.dtype)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
            Q = Q.reshape(-1, 1)
        if P.shape[0] != self.n_cp or Q.shape[0] != self.n_cp:
            raise ValueError(f"P,Q must be (n_cp, M) with n_cp={self.n_cp}")
        M = int(P.shape[1])
        times: dict[str, float] = {}

        t0 = time.perf_counter()
        if self.has_H:
            times["rhs"] = 0.0
            t1 = time.perf_counter()
            X = self.H_p @ P + self.H_q @ Q + self.h.reshape(-1, 1)
            times["solve"] = time.perf_counter() - t1
        else:
            B = self.G_p @ P + self.G_q @ Q + self.b0.reshape(-1, 1)
            times["rhs"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            # Multi-RHS via column loop on cached factorization (no refactor).
            X0 = np.empty((self.n_sys, M), dtype=self.dtype)
            for j in range(M):
                X0[:, j] = np.asarray(self.solve_A(B[:, j]), dtype=self.dtype)
            times["base_solve"] = time.perf_counter() - t1
            if self.use_woodbury:
                t2 = time.perf_counter()
                # rhs_small: k x M
                tmp = self.VAinv @ B
                rhs_small = self.L @ tmp
                if self._M_lu is not None:
                    W = np.linalg.lu_solve(self._M_lu, rhs_small)
                else:
                    W = np.linalg.solve(self.M, rhs_small)
                X = X0 - self.AinvU @ W
                times["woodbury"] = time.perf_counter() - t2
            else:
                X = X0
                times["solve"] = times.pop("base_solve", 0.0)

        t3 = time.perf_counter()
        half = self.n_unk
        vm = np.exp(X[:half, :])
        va = X[half:, :]
        times["recover"] = time.perf_counter() - t3
        times["total"] = float(sum(times.values()))
        if profile:
            return vm, va, times
        return vm, va

    def scale_loads(self, scale: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (P,Q) = scale * nominal. ``scale`` scalar or (n_cp,) / (M,) broadcasting."""
        s = np.asarray(scale, dtype=self.dtype)
        if s.ndim == 0:
            return self.P0 * float(s), self.Q0 * float(s)
        if s.ndim == 1 and s.shape[0] == self.n_cp:
            return self.P0 * s, self.Q0 * s
        # treat as batch of global scales
        P = self.P0.reshape(-1, 1) * s.reshape(1, -1)
        Q = self.Q0.reshape(-1, 1) * s.reshape(1, -1)
        return P, Q


def _time_stats(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, float)
    return {
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "n": float(a.size),
    }


def benchmark_fast_vs_opendss(
    repo=None,
    *,
    feeder: str = "906",
    control_mode: str = "off",
    n_warmup: int = 10,
    n_latency: int = 200,
    n_sequential: int | None = None,
    build_dense_H: bool | None = None,
) -> dict[str, Any]:
    """Compare FastLogvSolver to OpenDSS with sequential (one-at-a-time) solves only.

    No multi-RHS batch inference for Log(v) or OpenDSS. Each timed scenario is:
      update loads → one PF solve → next scenario.
    """
    import os
    from pathlib import Path

    import opendssdirect as dss

    from logv3lpf_daily_demo import (
        FEEDERS,
        _quiet,
        compile_cmd,
        ensure_logv3lpf,
        resolve_feeder,
    )
    import logv3lpf
    import logv3lpf.linpf as linpf
    from logv3lpf.DSSParser import DSScase

    repo = Path(repo or Path.cwd()).resolve()
    ensure_logv3lpf(repo)
    key = resolve_feeder(feeder)
    meta = FEEDERS[key]
    dss_path = Path(meta["dss"](repo)).resolve()
    n_seq = int(n_sequential) if n_sequential is not None else int(n_latency)

    os.chdir(repo)
    t_init0 = time.perf_counter()
    with _quiet():
        case = logv3lpf.case(
            compile_cmd(dss_path),
            meta["sourcebus"],
            float(meta["refvm"]),
            0,
        )
        try:
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Set ControlMode=OFF")
            dss.Solution.Solve()
            DSScase.process_openDSS_solution(case)
        except Exception:
            pass
    solver = FastLogvSolver(case, build_dense_H=build_dense_H)
    t_init = time.perf_counter() - t_init0

    # Accuracy check vs stock path at nominal
    vm_f, va_f = solver.solve()
    linpf.rank_k_correction_solve(case, False)
    vm_ref = np.asarray(case.vm, float).reshape(-1)
    va_ref = np.asarray(case.va, float).reshape(-1)
    n = min(len(vm_f), len(vm_ref))
    vm_mae = float(np.mean(np.abs(vm_f[:n] - vm_ref[:n])))
    va_mae = float(np.mean(np.abs(((va_f[:n] - va_ref[:n] + np.pi) % (2 * np.pi)) - np.pi)))

    # Shared load scales for fair sequential compare
    scales = 0.95 + 0.1 * np.random.default_rng(0).random(n_latency + n_warmup + n_seq)

    # ---- sequential latency: FastLogv (one scenario at a time) ----
    for i in range(n_warmup):
        P, Q = solver.scale_loads(float(scales[i]))
        solver.solve(P, Q)
    lat_fast = []
    stages = []
    for i in range(n_latency):
        P, Q = solver.scale_loads(float(scales[n_warmup + i]))
        t0 = time.perf_counter()
        _, _, st = solver.solve(P, Q, profile=True)
        lat_fast.append(time.perf_counter() - t0)
        stages.append(st)

    # ---- sequential latency: stock rank_k ----
    lat_stock = []
    for i in range(min(30, n_latency)):
        scale = float(scales[n_warmup + i])
        for local, j in enumerate(solver.cp_idx):
            case.loads.at[int(j), "kW"] = float(solver.P0[local] * scale)
            case.loads.at[int(j), "kvar"] = float(solver.Q0[local] * scale)
        t0 = time.perf_counter()
        linpf.rank_k_correction_solve(case, False)
        lat_stock.append(time.perf_counter() - t0)
    for local, j in enumerate(solver.cp_idx):
        case.loads.at[int(j), "kW"] = float(solver.P0[local])
        case.loads.at[int(j), "kvar"] = float(solver.Q0[local])

    # ---- sequential latency: OpenDSS ----
    lat_od = []
    for i in range(n_warmup):
        with _quiet():
            dss.Text.Command(f"Set LoadMult={float(scales[i])}")
            dss.Solution.Solve()
    for i in range(n_latency):
        with _quiet():
            t0 = time.perf_counter()
            dss.Text.Command(f"Set LoadMult={float(scales[n_warmup + i])}")
            dss.Solution.Solve()
            lat_od.append(time.perf_counter() - t0)
    with _quiet():
        dss.Text.Command("Set LoadMult=1")
        dss.Solution.Solve()

    # ---- sequential multi-scenario wall time (same N, one-by-one) ----
    seq_scales = scales[n_warmup + n_latency : n_warmup + n_latency + n_seq]
    if len(seq_scales) < n_seq:
        extra = 0.95 + 0.1 * np.random.default_rng(2).random(n_seq - len(seq_scales))
        seq_scales = np.concatenate([seq_scales, extra])

    t0 = time.perf_counter()
    for sc in seq_scales:
        P, Q = solver.scale_loads(float(sc))
        solver.solve(P, Q)
    fast_seq_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    with _quiet():
        for sc in seq_scales:
            dss.Text.Command(f"Set LoadMult={float(sc)}")
            dss.Solution.Solve()
    od_seq_total = time.perf_counter() - t0
    with _quiet():
        dss.Text.Command("Set LoadMult=1")
        dss.Solution.Solve()

    sequential = {
        "n_scenarios": float(n_seq),
        "fast_total_sec": float(fast_seq_total),
        "opendss_total_sec": float(od_seq_total),
        "fast_mean_ms": float(1e3 * fast_seq_total / max(n_seq, 1)),
        "opendss_mean_ms": float(1e3 * od_seq_total / max(n_seq, 1)),
        "speedup_total_od_over_fast": float(od_seq_total / fast_seq_total)
        if fast_seq_total > 0
        else float("nan"),
    }

    stage_med = {
        "rhs_ms": float(np.median([s.rhs for s in stages]) * 1e3),
        "base_solve_ms": float(np.median([s.base_solve for s in stages]) * 1e3),
        "woodbury_ms": float(np.median([s.woodbury for s in stages]) * 1e3),
        "recover_ms": float(np.median([s.recover for s in stages]) * 1e3),
    }

    out = {
        "feeder": key,
        "mode": "sequential",
        "n_sys": solver.n_sys,
        "n_cp": solver.n_cp,
        "k": solver.k,
        "has_H": solver.has_H,
        "use_woodbury": solver.use_woodbury,
        "t_init_sec": float(t_init),
        "vm_mae_vs_stock": vm_mae,
        "va_mae_vs_stock_rad": va_mae,
        "latency_fast_sec": _time_stats(lat_fast),
        "latency_stock_sec": _time_stats(lat_stock),
        "latency_opendss_sec": _time_stats(lat_od),
        "stage_med_ms": stage_med,
        "sequential": sequential,
        "speedup_latency_vs_od": float(
            _time_stats(lat_od)["median"] / _time_stats(lat_fast)["median"]
        ),
        "speedup_latency_vs_stock": float(
            _time_stats(lat_stock)["median"] / _time_stats(lat_fast)["median"]
        ),
    }

    print("=" * 72)
    print(f"FAST Log(v) vs OpenDSS — {key}  [SEQUENTIAL ONLY]")
    print("=" * 72)
    print(
        f"  n_sys={solver.n_sys}  n_cp={solver.n_cp}  k={solver.k}  "
        f"dense_H={solver.has_H}  woodbury={solver.use_woodbury}"
    )
    print(f"  init (parse+matrices+fast build): {t_init:.3f} s")
    print(f"  check vs stock rank_k: |V| MAE={vm_mae:.3e} pu")
    print("  Per-scenario latency (median; one PF at a time)")
    print(
        f"    FastLogv   : {out['latency_fast_sec']['median']*1e3:.3f} ms  "
        f"(stages ms: rhs={stage_med['rhs_ms']:.3f} solve={stage_med['base_solve_ms']:.3f} "
        f"wood={stage_med['woodbury_ms']:.3f} rec={stage_med['recover_ms']:.3f})"
    )
    print(f"    stock Logv : {out['latency_stock_sec']['median']*1e3:.3f} ms")
    print(f"    OpenDSS    : {out['latency_opendss_sec']['median']*1e3:.3f} ms")
    print(
        f"    speedup Fast/OD={out['speedup_latency_vs_od']:.2f}x  "
        f"Fast/stock={out['speedup_latency_vs_stock']:.2f}x"
    )
    print(f"  Sequential wall time for N={n_seq} scenarios (no batching)")
    print(
        f"    FastLogv total : {sequential['fast_total_sec']:.4f} s  "
        f"(mean {sequential['fast_mean_ms']:.3f} ms/scenario)"
    )
    print(
        f"    OpenDSS total  : {sequential['opendss_total_sec']:.4f} s  "
        f"(mean {sequential['opendss_mean_ms']:.3f} ms/scenario)"
    )
    print(
        f"    speedup OD/Fast total = {sequential['speedup_total_od_over_fast']:.2f}x"
    )
    print("=" * 72)
    return out


def benchmark_all_feeders(repo=None, **kwargs) -> dict[str, Any]:
    out = {}
    for f in ("ieee34", "906", "8500"):
        try:
            out[f] = benchmark_fast_vs_opendss(repo, feeder=f, **kwargs)
        except Exception as exc:
            print(f"FAIL {f}: {type(exc).__name__}: {exc}")
            out[f] = {"error": str(exc)}
    return out


if __name__ == "__main__":
    from pathlib import Path

    benchmark_all_feeders(
        Path(__file__).resolve().parent,
        n_warmup=5,
        n_latency=100,
        n_sequential=100,
    )
