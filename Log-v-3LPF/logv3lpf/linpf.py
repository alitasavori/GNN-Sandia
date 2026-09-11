import warnings

import numpy as np
import pandas as pd
import opendssdirect as dss
from cmath import exp,pi,sqrt,log
from itertools import product
import scipy
from copy import deepcopy
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve as solve
from scipy.sparse.linalg import factorized as sparse_factorized
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import coo_matrix as sparse_matrix
from logv3lpf.utils import get_transformer_yprim
from scipy.linalg import pinv,inv
from math import log

try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except Exception:
    pass

# Optional diagonal ridge on reduced A before factorization (default 0 = paper-faithful).
# Set case.ainv_ridge > 0 only for numerical experiments; do not use in published baselines.
_AINV_RIDGE_DEFAULT = 0.0


def _apply_factor(solve_fn, B):
    """Apply a ``factorized`` solver to a 1-D or 2-D right-hand side."""
    B = np.asarray(B)
    if B.ndim == 1:
        return np.asarray(solve_fn(B), dtype=float)
    if B.ndim == 2 and B.shape[1] == 1:
        return np.asarray(solve_fn(B.ravel()), dtype=float).reshape(-1, 1)
    return np.asarray(solve_fn(B), dtype=float)


def _ytilde_from_Ytilde(Ytilde: np.ndarray) -> np.ndarray:
    """Paper: ``ỹ := Ỹ 1`` (row sums). Not NumPy ``sum(axis=0)`` (column sums)."""
    return np.asarray(Ytilde, dtype=complex).sum(axis=1)

def get_line_matrices(case):

    Nbrl,Nn = case.Nbrl,case.Nn

    # Define incidence matrix
    E = csr_matrix((Nn,2*Nbrl))

    Yu,Yθ,y = [],[],[]
    Yprims = []
    counter = 0

    for i in range(0,len(case.lines)):

        fbus,tbus = case.lines.fbus[i],case.lines.tbus[i]
        zbase = case.base[fbus]["ZBase"]
        fphases = case.lines.fphase[i]
        tphases = case.lines.tphase[i]
        idx_fphases = [case.bus_phases[fbus].index(phase) for phase in fphases]
        idx_tphases = [case.bus_phases[tbus].index(phase) for phase in tphases]
        if len(fphases) == len(tphases):
            nl = len(fphases)
        Δ3_n = case.base[fbus]["Δ3"]
        Δ3_m = case.base[tbus]["Δ3"]

        fidx = [case.node_from_name["{:}.{:}".format(fbus,j)] for j in fphases]
        tidx = [case.node_from_name["{:}.{:}".format(tbus,j)] for j in tphases]

        E[fidx,[j+counter for j in range(0,nl)]] = 1
        counter+=nl
        E[tidx,[j+counter for j in range(0,nl)]] = 1
        counter+=nl

        Δ3nphase = np.diag(Δ3_n[idx_fphases])
        Δ3mphase = np.diag(Δ3_m[idx_tphases])
        Δ3phase_block = block_diag(Δ3nphase, Δ3mphase)


        Ynm_n = case.lines.ynm[i]*zbase
        Ynm_m = -case.lines.ynm[i]*zbase
        Ymn_n = -case.lines.ynm[i]*zbase
        Ymn_m = case.lines.ynm[i]*zbase
        Ynm_s = case.lines.ysh[i]*zbase
        Ymn_s = case.lines.ysh[i]*zbase

        Yprim = np.block([[(Ynm_n+Ynm_s/2),Ynm_m],[Ymn_n,(Ymn_m+Ymn_s/2)]])
        Yprims.append(Yprim)
        # Paper: Ỹ := Δ (Y_prim)* (Δ)^H ,  ỹ := Ỹ 1
        Ytilde = Δ3phase_block@Yprim.conj()@Δ3phase_block.conj()
        ytilde = _ytilde_from_Ytilde(Ytilde)

        # Y.append(np.block([Ytilde + np.diag(ytilde), -1j*Ytilde+1j*np.diag(ytilde)]))
        Yu.append(Ytilde + np.diag(ytilde))
        Yθ.append(-1j*Ytilde+1j*np.diag(ytilde))
        y.append(ytilde)

    # Y = csr_matrix(block_diag(*Y))
    if len(y) != 0:
        y = np.hstack(y).reshape(-1,1)
    else:
        y = np.array([])

    case.lines["Yprim"] = Yprims
    case.Yul = Yu
    case.Yθl = Yθ
    case.yl = y
    case.El = E


def get_transformer_matrices(case):

    Nbrt,Nbrr,Nn = case.Nbrt,case.Nbrr,case.Nn

    # NOTE: Do NOT push case.transformers.taps into OpenDSS here.
    # OpenDSS YPrim already reflects the live DSS taps (after Static settle).
    # Pushing every xfmr tap on rebuild can desynchronize the DSS circuit after
    # daily loadshape rebinds. Autonomous RegControl pushes taps explicitly
    # via push_transformer_taps_to_opendss() before calling this function.

    # Define incidence matrix
    Et = csr_matrix((Nn,Nbrt))
    Er = csr_matrix((Nn,Nbrr))

    Yprims = []
    Yprim_idx = []
    Yut,Yθt,yt = [],[],[]
    counter = 0

    # Transformers are modeled as power delivery elements (lines). However, Yprim is not symmetric
    for transformer in case.transformer_names:
        i = case.transformers[case.transformers.name == transformer].index.values[0]
        Yprim_idx.append(i)
        buses = case.transformers.buses[i]
        phases = deepcopy(case.transformers.phases[i])
        [phase.remove(0) for phase in phases if 0 in phase]

        idx_phases = [[case.bus_phases[buses[j]].index(phase) for phase in phases[j]] for j in range(0,len(phases))]
        Δ3s = [np.diag(case.base[buses[j]]["Δ3"][idx_phases[j]]) for j in range(0,len(idx_phases))]

        idx = np.hstack([[case.node_from_name["{:}.{:}".format(buses[j],phase)] for phase in phases[j]] for j in range(0,len(buses))])

        Et[idx,[j+counter for j in range(0,len(idx))]] = 1
        counter += len(idx)

        Δ3phase_block = block_diag(*Δ3s)

        # Yprim,Tau = get_alt_transformer_yprim(case,i)
        # Paper Y_prim path; magnetizing branch is commented out upstream (no-load loss omitted).
        Yprim = get_transformer_yprim(case,i,pu=True)
        Yprims.append(Yprim)

        Ytilde = Δ3phase_block@Yprim.conj()@Δ3phase_block.conj()
        ytilde = _ytilde_from_Ytilde(Ytilde)

        Yut.append(Ytilde + np.diag(ytilde))
        Yθt.append(-1j*Ytilde+1j*np.diag(ytilde))
        yt.append(ytilde)

    # Yt = csr_matrix(block_diag(*Yt))
    if len(yt) != 0:
        yt = np.hstack(yt).reshape(-1,1)
    else:
        yt = np.array([])


    Yur,Yθr,yr = [],[],[]
    counter = 0
        
    for regulator in case.regulator_names:
        i = case.transformers[case.transformers.name == regulator].index.values[0]
        Yprim_idx.append(i)
        buses = case.transformers.buses[i]
        phases = deepcopy(case.transformers.phases[i])
        [phase.remove(0) for phase in phases if 0 in phase]

        idx_phases = [[case.bus_phases[buses[j]].index(phase) for phase in phases[j]] for j in range(0,len(phases))]
        Δ3s = [np.diag(case.base[buses[j]]["Δ3"][idx_phases[j]]) for j in range(0,len(idx_phases))]


        idx = np.hstack([[case.node_from_name["{:}.{:}".format(buses[j],phase)] for phase in phases[j]] for j in range(0,len(buses))])

        Er[idx,[j+counter for j in range(0,len(idx))]] = 1
        counter += len(idx)

        Δ3phase_block = block_diag(*Δ3s)

        # Yprim,Tau = get_alt_transformer_yprim(case,i)
        # Tau = np.diag(1/np.array(case.transformers.taps[i]))
        # Yprim = Tau@get_transformer_yprim(case,i,pu=True)@Tau
        Yprim = get_transformer_yprim(case,i,pu=True)
        Yprims.append(Yprim)
        # Paper: Ỹ := Δ (Y)* (Δ)^H  (upstream regulator branch omitted .conj() on right)
        Ytilde = Δ3phase_block@Yprim.conj()@Δ3phase_block.conj()
        ytilde = _ytilde_from_Ytilde(Ytilde)

        # Regulators with tap≠1 have nonzero row sums. Using Ỹ+diag(ỹ) (line formula)
        # distorts the open-circuit ratio (e.g. tap 1.0625 → Voc≈1.13) so baking OD
        # taps over-boosts |V|. Correct split for I≈Y(1+u): put Ỹ in A and ỹ on RHS.
        Yur.append(Ytilde)
        Yθr.append(-1j * Ytilde)
        yr.append(ytilde)

    # Yr = csr_matrix(block_diag(*Yr))
    if len(yr) != 0:
        yr = np.hstack(yr).reshape(-1,1)
        case.transformers["Yprim"] = [Yprims[Yprim_idx.index(j)] for j in range(0,len(Yprim_idx))] 
    else:
        yr = np.array([])

    case.Yut = Yut
    case.Yθt = Yθt
    case.yt = yt
    case.Et = Et

    case.Yur = Yur
    case.Yθr = Yθr
    case.yr = yr
    case.Er = Er


def calculate_base_matrices(case):
    refvm = case.refvm
    refva = case.refva
    refbus = case.refbus
    Nn = case.Nn

    if len(case.yl) == 0:
        Ylbus = csr_matrix((2*Nn,2*Nn))
    else:
        Ylbus = sparse_block_diag((case.El,case.El))@scipy.sparse.bmat([[sparse_block_diag(case.Yul).real,sparse_block_diag(case.Yθl).real],[sparse_block_diag(case.Yul).imag,sparse_block_diag(case.Yθl).imag]])@sparse_block_diag((case.El,case.El)).T
    if len(case.yt) == 0:  
        Ytbus = csr_matrix((2*Nn,2*Nn))
    else:
        Ytbus = sparse_block_diag((case.Et,case.Et))@scipy.sparse.bmat([[sparse_block_diag(case.Yut).real,sparse_block_diag(case.Yθt).real],[sparse_block_diag(case.Yut).imag,sparse_block_diag(case.Yθt).imag]])@sparse_block_diag((case.Et,case.Et)).T
    if len(case.yr) == 0:
        Yrbus = csr_matrix((2*Nn,2*Nn))
    else:
        Yrbus = sparse_block_diag((case.Er,case.Er))@scipy.sparse.bmat([[sparse_block_diag(case.Yur).real,sparse_block_diag(case.Yθr).real],[sparse_block_diag(case.Yur).imag,sparse_block_diag(case.Yθr).imag]])@sparse_block_diag((case.Er,case.Er)).T
    A = Ylbus + Ytbus + Yrbus

    y = np.zeros((Nn,1), dtype=complex)
    if len(case.yl) != 0:
        y += case.El@case.yl
    if len(case.yr) != 0:
        y += case.Er@case.yr
    if len(case.yt) != 0:
        y += case.Et@case.yt
    sshunt = np.vstack([y.real,y.imag])

    #Gets masks for the admittance and incidence matrices (gets rid of the reference bus)
    ref_bus_phases_real = [case.node_from_name["{:}.{:}".format(refbus,i)] for i in case.bus_phases[refbus]]
    ref_bus_phases_imag = [el+case.Nn for el in ref_bus_phases_real]
    ref_bus_phases = ref_bus_phases_real+ref_bus_phases_imag
    mask = np.setdiff1d(list(range(0, 2*case.Nn)),ref_bus_phases)

    # Sparse factorization of reduced A (applies Ã^{-1} without forming dense inverse).
    # Paper-faithful: no ridge unless case.ainv_ridge > 0 (experimental only).
    Am = A[mask, :][:, mask].tocsc().astype(float)
    nA = Am.shape[0]
    ridge = float(getattr(case, "ainv_ridge", _AINV_RIDGE_DEFAULT) or 0.0)
    if ridge > 0.0:
        Am = Am + ridge * scipy.sparse.eye(nA, format="csc", dtype=float)
    try:
        solve_A = sparse_factorized(Am)
        solve_AT = sparse_factorized(Am.T.tocsc())
    except Exception as exc:
        raise RuntimeError(
            "Failed to factor reduced Log(v) matrix Ã. This usually means a singular "
            "transformer/grounding representation — do not paper over with ridge without "
            f"disclosing it. Original error: {exc}"
        ) from exc

    Adrop = A[:,ref_bus_phases]
    n_ref = len(ref_bus_phases_real)
    # Reference contribution: same |V| and the scalar Vsource angle on every
    # eliminated phase. Balanced 120 deg offsets live in Delta3 / Shift used by
    # Yprim and angle reconstruction — not in this repeated refva vector.
    sdrop = Adrop @ np.array(
        [log(refvm)] * n_ref + [np.deg2rad(refva)] * n_ref
    ).reshape(-1, 1)

    E = scipy.sparse.bmat([[case.EloadYl,None],[None,case.EloadYl]])
    U = csr_matrix(E)[mask,:]
    V = csr_matrix(E.T)[:,mask]

    k = U.shape[1]
    I = csr_matrix(np.eye(k,dtype=float))

    U_arr = U.toarray() if scipy.sparse.issparse(U) else np.asarray(U)
    V_arr = V.toarray() if scipy.sparse.issparse(V) else np.asarray(V)
    AinvU = _apply_factor(solve_A, U_arr)
    # V @ A^{-1} = (A^{-T} @ V.T).T
    VAinv = _apply_factor(solve_AT, V_arr.T).T
    VAinvU = VAinv @ U_arr

    case.mask = mask
    case.Ylbus = Ylbus
    case.Ytbus = Ytbus
    case.Yrbus = Yrbus
    case.A = A
    case.I = I
    case.U = U
    case.V = V
    case.E = E
    case.sshunt = sshunt[mask]
    case.Adrop = Adrop
    case.sdrop = sdrop[mask]
    case._solve_A = solve_A
    case.Ainv = None  # dense inverse no longer stored
    case.AinvU = AinvU
    case.VAinv = VAinv
    case.VAinvU = VAinvU


def get_loads(case):
    Ω = case.Ω
    Π = case.Π

    s = []
    Lu = []
    Lθ = []

    for i in range(len(case.loads)):
        phases = np.array(case.loads.phases[i])-1
        bus = case.loads.bus[i]
        nphases = case.loads.nphases[i]
        model = case.loads.model[i]
        kV = case.loads.kV[i]
        kW = case.loads.kW[i]
        kvar = case.loads.kvar[i]
        isdelta = case.loads.isdelta[i]


        idx_phases = [case.bus_phases[bus].index(phase) for phase in case.loads.phases[i]]
        Δ3phase = np.diag(case.base[bus]["Δ3"][idx_phases])

        if model == 1:
            base = case.base[bus]["VABase"]
            p = np.repeat((kW*1000+1j*kvar*1000)/(nphases*base),nphases).reshape(-1,1)
            if isdelta:
                if nphases == 1:
                    Ωphase = Ω[phases,phases[0]].reshape(-1,1)
                    s.append(Ωphase@p)
                elif nphases == 3:
                    s.append(Ω@p)
            else:
                s.append(p)

        elif model == 2:
            base = case.base[bus]["ZBase"]
            s_complex = float(kW) * 1000.0 + 1j * float(kvar) * 1000.0
            if abs(s_complex) < 1e-18:
                nph = int(nphases)
                s.append(np.zeros((nph, 1), dtype=complex))
                zblock = np.zeros((nph, nph), dtype=complex)
                Lu.append(zblock)
                Lθ.append(zblock)
                continue
            if (nphases == 1) | (nphases == 3):
                z = np.repeat((kV*1000)**2 / np.conj(s_complex),nphases)
            elif nphases == 2:
                z = np.repeat((2/3)*(kV*1000)**2 / np.conj(s_complex),nphases)
            z /= base

            if isdelta:
                # Δ3phase = np.diag(case.base[bus]["Δ3"][phases,phases])
                # Delta loads cannot be 2 phase because otherwise the buses they are connected to are ambiguous
                if nphases == 1:
                    Πphase = Π[phases,phases[0]].reshape(-1,1)
                elif nphases == 3:
                    Πphase = Π
                
                X = Δ3phase.conj()@Πphase@np.diag((1/z).conj())@Πphase.T
                s.append(np.sum(Δ3phase@X.T,axis=1).reshape(-1,1))

                Lu.append((Δ3phase@X.T+np.diag(np.sum(Δ3phase@X.T,axis=1))))
                Lθ.append(1j*(np.diag(np.sum(Δ3phase@X.T,axis=1)) - Δ3phase@X.T))           
            else:
                # Δ3phase = np.diag(case.base[bus]["Δ3"][phases,phases])
                s.append((1/z).conj().reshape(-1,1))
                
                X = Δ3phase.conj()@np.diag((1/z).conj())
                load = Δ3phase@X.T+np.diag(np.sum(Δ3phase@X.T,axis=1))
                Lu.append(load)
                Lθ.append(np.zeros(load.shape))




        elif model == 5:
            base = case.base[bus]["IBase"]
            shift = case.base[bus]["Shift"]
            power_angle = np.angle(kW+1j*kvar,deg=True)
            phaseidx = [case.bus_phases[bus].index(phase) for phase in np.array(case.loads.phases[i])]
            # Δ3phase = np.diag(case.base[bus]["Δ3"][phases,phases])
            if (nphases == 1) | isdelta:
                I = np.repeat(abs(kW+1j*kvar)/(nphases*kV),nphases)
            else:
                I = np.repeat(abs(kW+1j*kvar)/(sqrt(3)*nphases*kV),nphases)
            I /= base

            if isdelta:
                va = shift + pi/6
                # Delta loads cannot be 2 phase because otherwise the buses they are connected to are ambiguous
                if nphases == 1:
                    I = (I*np.exp( 1j * (va[phaseidx[0]]-power_angle) )).reshape(-1,1)
                    Πphase = Π[phases,phases[0]].reshape(-1,1)
                elif nphases == 3:
                    I = (I*np.exp( 1j * (va[phaseidx]-power_angle) )).reshape(-1,1)
                    Πphase = Π

                load = Πphase@I.conj()
                sload = Δ3phase@load
                
                s.append(sload.reshape(-1,1))
                Lu.append(np.diag(sload.T[0]))
                Lθ.append(np.diag(1j*sload.T[0]))


            else:
                va = shift
                I = (I*np.exp( 1j * (2*pi/360)*(va[phaseidx]-power_angle))).reshape(-1,1)
                sload = Δ3phase@I.conj()
                s.append(sload)
                Lu.append(np.diag(sload.T[0]))
                Lθ.append(np.diag(1j*sload.T[0]))
        else:
            print("Load {:} is model {:} which is not supported. Try model=1,model=2 or model=5".format(case.loads.name[i],case.loads.model[i]))



    for i in range(len(case.capacitors)):
        phases = np.array(case.capacitors.phases[i])-1
        bus = case.capacitors.bus[i]
        nphases = case.capacitors.nphases[i]
        kV = case.capacitors.kV[i]
        kvar = float(case.capacitors.kvar[i])
        isdelta = case.capacitors.isdelta[i]
        # Switched-off banks are stored as kvar=0. Keep s / Lu / Lθ lengths aligned
        # with EloadS / EloadYl (which include every capacitor phase).
        if abs(kvar) < 1e-12:
            nph = int(nphases)
            s.append(np.zeros((nph, 1), dtype=complex))
            zblock = np.zeros((nph, nph), dtype=complex)
            Lu.append(zblock)
            Lθ.append(zblock)
            continue
        base = case.base[bus]["ZBase"]

        idx_phases = [case.bus_phases[bus].index(phase) for phase in case.capacitors.phases[i]]
        Δ3phase = np.diag(case.base[bus]["Δ3"][idx_phases])

        if (nphases == 1) | (nphases == 3):
            z = np.repeat((kV*1000)**2 / np.conj((-1j*kvar*1000)),nphases)
        elif nphases == 2:
            z = np.repeat((2/3)*(kV*1000)**2 / np.conj((-1j*kvar*1000)),nphases)
        z /= base

        if isdelta:
            # Δ3phase = np.diag(case.base[bus]["Δ3"][phases,phases])
            # Delta loads cannot be 2 phase because otherwise the buses they are connected to are ambiguous
            if nphases == 1:
                Πphase = Π[phases,phases[0]].reshape(-1,1)
            elif nphases == 3:
                Πphase = Π
            
            X = Δ3phase.conj()@Πphase@np.diag((1/z).conj())@Πphase.T
            s.append(np.sum(Δ3phase@X.T,axis=1).reshape(-1,1))

            Lu.append((Δ3phase@X.T+np.diag(np.sum(Δ3phase@X.T,axis=1))))
            Lθ.append(1j*(np.diag(np.sum(Δ3phase@X.T,axis=1)) - Δ3phase@X.T))           

        else:
            # Δ3phase = np.diag(case.base[bus]["Δ3"][phases,phases])
            s.append((1/z).conj().reshape(-1,1))
            
            X = Δ3phase.conj()@np.diag((1/z).conj())
            load = Δ3phase@X.T+np.diag(np.sum(Δ3phase@X.T,axis=1))
            Lu.append(load)
            Lθ.append(np.zeros(load.shape))

    case.s = -np.vstack(s)
    if len(Lu)!=0:
        Lu = sparse_block_diag(Lu)
    if len(Lθ)!=0:
        Lθ = sparse_block_diag(Lθ)

    case.Lu = Lu
    case.Lθ = Lθ

def rank_k_correction_solve(case,process_solution):

    get_loads(case)

    mask = case.mask

    Lu = case.Lu
    Lθ = case.Lθ

    ## Get s vector
    sload = case.EloadS@case.s
    sbus = np.vstack([sload.real,sload.imag])[mask] - case.sdrop - case.sshunt

    sbus_col = np.asarray(sbus, dtype=float).reshape(-1, 1)
    if type(Lu) == list:
        sol = _apply_factor(case._solve_A, sbus_col)
    else:
        L = scipy.sparse.bmat([[Lu.real,Lθ.real],[Lu.imag,Lθ.imag]])

        ## Woodbury: (A + U L V)^{-1} s = A^{-1}s - A^{-1}U (I + L V A^{-1}U)^{-1} L V A^{-1}s
        ## Solve the small dense system instead of forming inv(...).
        rhs_small = L @ (case.VAinv @ sbus_col)
        M = case.I + L @ case.VAinvU
        if scipy.sparse.issparse(M):
            M = M.toarray()
        else:
            M = np.asarray(M)
        if scipy.sparse.issparse(rhs_small):
            rhs_small = rhs_small.toarray()
        rhs_small = np.asarray(rhs_small, dtype=float)
        if rhs_small.ndim == 1:
            rhs_small = rhs_small.reshape(-1, 1)
        w = np.linalg.solve(np.asarray(M, dtype=float), rhs_small)
        corr = case.AinvU @ w
        sol = _apply_factor(case._solve_A, sbus_col) - corr

    vm = np.exp(sol[:int(len(sol)/2)])
    va = sol[int(len(sol)/2):]

    case.vm = vm
    case.va = va

    if process_solution:
        process_logv3lpf_solution(case)

def check_logv3lpf_performance(case):
    """Print Table II-style FLOP estimates using regulator_names (not RegControl name)."""
    k_phases = 0
    if len(case.loads[case.loads.model!=1].phases)!=0:
        k_phases += len(np.hstack(case.loads[case.loads.model!=1].phases))
    if case.capacitors is not None and len(case.capacitors) and len(case.capacitors.phases)!=0:
        k_phases += len(np.hstack(case.capacitors.phases))
    # Paper counts regulator windings; prefer regulator_names, else RegControl.transformer.
    reg_names = list(getattr(case, "regulator_names", None) or [])
    if not reg_names and getattr(case, "regcontrols", None) is not None and len(case.regcontrols):
        if "transformer" in case.regcontrols.columns:
            reg_names = [str(x) for x in case.regcontrols.transformer.values]
        else:
            reg_names = [str(x) for x in case.regcontrols.name.values]
    for name in reg_names:
        hit = case.transformers[case.transformers.name == name]
        for phases in hit.phases:
            el = np.array(phases)
            k_phases += len(el[el != 0])
    k = k_phases * 2
    n = case.Nn * 2
    O_KLU = int((2 / 3) * (n ** 3) + (2 * n) ** 2)
    O_logv = 3 * (k) ** 3 + (2 * n * (k ** 2)) + n ** 2 + k
    print("Nodes = ", case.Nn)
    print("k = ", k)
    print("Complexity of KLU decomposition, {:.2e} FLOPS ".format(O_KLU))
    print("Complexity of LogV algorithm, {:.2e} FLOPS ".format(O_logv))
    if k > 0:
        print("Ratio {:.2f}".format(O_KLU / O_logv))


def _wrap_deg(degrees):
    """Map degrees to (-180, 180]."""
    degrees = np.asarray(degrees, dtype=float).copy()
    for i in range(len(degrees)):
        while degrees[i] > 180:
            degrees[i] -= 360.0
        while degrees[i] <= -180:
            degrees[i] += 360.0
    return degrees


def _anchor_logv_angles_to_opendss_source(case, dssva):
    """Optional safety: rotate Log(v) angles if source still disagrees with OpenDSS.

    Primary convention is now set in ``DSSParser`` from ``Vsources.AngleDeg`` plus
    Dy Lag (-30 on LV). After that fix, offset should be ~0 on ieee34/906/8500.
    We only apply a residual rotation if |mean source error| > 0.5 deg (e.g. odd DSS).
    """
    od = getattr(case, "results", None) or {}
    od_va = (od.get("openDSS") or {}).get("va") or {}
    if not od_va:
        return dssva, 0.0
    src = case.sourcebus
    src_key = src if src in od_va else next(
        (k for k in od_va if str(k).lower() == str(src).lower()), None
    )
    if src_key is None or src not in dssva:
        src_lv = next((k for k in dssva if str(k).lower() == str(src).lower()), None)
        if src_lv is None or src_key is None:
            return dssva, 0.0
        src = src_lv
    ao = np.asarray(od_va[src_key], dtype=float)
    al = np.asarray(dssva[src], dtype=float)
    m = min(len(ao), len(al))
    if m <= 0:
        return dssva, 0.0
    d = (ao[:m] - al[:m] + 180.0) % 360.0 - 180.0
    offset = float(np.mean(d))
    if abs(offset) < 0.5:
        return dssva, offset  # report residual; do not rotate
    out = {}
    for bus, vals in dssva.items():
        out[bus] = _wrap_deg(np.asarray(vals, dtype=float) + offset)
    return out, offset


def process_logv3lpf_solution(case):
    dssvm = case.bus_phases.copy()
    dssva = case.bus_phases.copy()

    src = case.sourcebus
    if src not in case.bus_phases:
        for b in case.bus_phases:
            if str(b).lower() == str(src).lower():
                src = b
                case.sourcebus = b
                break
    ref_bus_phases_real = [case.node_from_name["{:}.{:}".format(src,i)] for i in case.bus_phases[src]]
    vm = np.zeros(case.Nn)
    vm[ref_bus_phases_real] = [case.refvm]*len(ref_bus_phases_real)
    vm[case.mask[:int(len(case.mask)/2)]] = case.vm.T[0]

    for name in dssvm.keys():
        idx = [case.node_from_name["{:}.{:}".format(name,phase)] for phase in case.bus_phases[name]]
        dssvm[name] = vm[idx]

    # Reconstruct angles in mask order with per-node Shift (do NOT use np.unique
    # bus order — that alphabetizes and can scramble phase assignment).
    va = np.zeros(case.Nn)
    src_shift = np.asarray(case.base[src]["Shift"], dtype=float)
    src_phases = list(case.bus_phases[src])
    for j, ph in enumerate(src_phases):
        # Shift vector is indexed by phase-1 when built from ABC; fall back safely.
        if len(src_shift) == len(src_phases):
            va[ref_bus_phases_real[j]] = float(src_shift[j])
        else:
            va[ref_bus_phases_real[j]] = float(src_shift[min(j, len(src_shift) - 1)])

    mask_real = case.mask[: int(len(case.mask) / 2)]
    solved = np.asarray(case.va.T[0], dtype=float).reshape(-1)
    if len(solved) != len(mask_real):
        raise ValueError(
            f"Log(v) angle length {len(solved)} != mask length {len(mask_real)}"
        )
    for local_i, node_i in enumerate(mask_real):
        nodename = case.node_from_idx[int(node_i)]
        bus, _, phase_s = nodename.partition(".")
        phase = int(phase_s)
        bus_phases = list(case.bus_phases[bus])
        shift_vec = np.asarray(case.base[bus]["Shift"], dtype=float)
        if phase in bus_phases and len(shift_vec) == len(bus_phases):
            sh = float(shift_vec[bus_phases.index(phase)])
        elif 0 <= (phase - 1) < len(shift_vec):
            sh = float(shift_vec[phase - 1])
        else:
            sh = float(shift_vec[min(len(shift_vec) - 1, 0)])
        va[int(node_i)] = float(solved[local_i]) + sh

    for name in dssva.keys():
        idx = [case.node_from_name["{:}.{:}".format(name,phase)] for phase in case.bus_phases[name]]
        dssva[name] = _wrap_deg(np.rad2deg(va[idx]))

    # Absolute OpenDSS frame: add Vsource.Angle (e.g. ieee34 = 30 deg).
    # Linearization keeps phase-A = 0 in Shift/Δ3; this is reporting only.
    vs_deg = float(getattr(case, "vsource_angle_deg", 0.0) or 0.0)
    if abs(vs_deg) > 1e-12:
        for bus in dssva:
            dssva[bus] = _wrap_deg(np.asarray(dssva[bus], dtype=float) + vs_deg)

    # Residual safety vs OpenDSS source (should be ~0 after Vsource.Angle).
    dssva, angle_offset_deg = _anchor_logv_angles_to_opendss_source(case, dssva)

    if not hasattr(case,"results"):
        case.results = {}

    case.results["logv3lpf"] = {}
    case.results["logv3lpf"]["vm"] = dssvm
    case.results["logv3lpf"]["va"] = dssva
    case.results["logv3lpf"]["va_source_offset_deg"] = float(angle_offset_deg)
    case.results["logv3lpf"]["vsource_angle_deg"] = float(vs_deg)

