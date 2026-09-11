import opendssdirect as dss
import numpy as np
import pandas as pd
from cmath import sqrt,exp,pi
import scipy
from scipy.sparse import lil_matrix
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
from copy import deepcopy

def get_transformer_incidence_matrices(case):
    # Implemented matrices : Ungrounded Wye, Grounded Wye, Delta

    Awye_bus1 = lil_matrix((4,12))
    Awye_bus2 = lil_matrix((4,12))
    AwyeG_bus1 = lil_matrix((3,12))
    AwyeG_bus2 = lil_matrix((3,12))
    Adelta_bus1 = lil_matrix((3,12))
    Adelta_bus2 = lil_matrix((3,12))

    Awye_bus1[[0,1,2,3,3,3],[0,4,8,1,5,9]] = 1
    AwyeG_bus1[[0,1,2],[0,4,8]] = 1

    Awye_bus2[[0,1,2,3,3,3],[2,6,10,3,7,11]] = 1
    AwyeG_bus2[[0,1,2],[2,6,10]] = 1

    Adelta_bus1[[0,0,1,1,2,2],[0,9,1,4,5,8]] = 1
    Adelta_bus2[[0,0,1,1,2,2],[2,11,3,6,7,10]] = 1

    Aconn = {"Wye_bus1":Awye_bus1,"Wye_bus2":Awye_bus2,"WyeG_bus1":AwyeG_bus1,"WyeG_bus2":AwyeG_bus2,"Delta_bus1":Adelta_bus1,"Delta_bus2":Adelta_bus2}

    return Aconn


def _opendss_transformer_yprim_pu(case, i):
    """Phase-node transformer Yprim in pu from the live OpenDSS circuit.

    Uses OpenDSS ``CktElement.YPrim`` (siemens), drops unused/neutral conductors,
    and converts to pu with ``Y_pu_ij = Y_ij * Vbase_i * Vbase_j / Sbase``.

    This keeps the paper Log(v) linearization (Ỹ, ỹ) while using a component
    model consistent with OpenDSS — the analytical incidence assembly for
    Delta–Wye in this package produces large column-sum shunt that is not
    present in OpenDSS Yprim.
    """
    try:
        name = str(case.transformers.name[i])
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        raw = np.asarray(dss.CktElement.YPrim(), dtype=float)
        if raw.size < 2 or raw.size % 2 != 0:
            return None
        c = raw[0::2] + 1j * raw[1::2]
        n = int(round(np.sqrt(c.size)))
        if n * n != c.size:
            return None
        Y = c.reshape(n, n)

        phases = deepcopy(case.transformers.phases[i])
        for ph in phases:
            if 0 in ph:
                ph.remove(0)
        n_expect = int(sum(len(ph) for ph in phases))
        if n_expect <= 0:
            return None

        if n == n_expect:
            keep = list(range(n))
        elif n == 4 and n_expect == 2:
            # 1ph 2-winding OpenDSS order: [H, Hn, L, Ln] — drop neutrals.
            keep = [0, 2]
        elif n == 8 and n_expect == 6:
            # Typical 3ph Delta–WyeG: [Ha,Hb,Hc,Hn, La,Lb,Lc,Ln]
            keep = [0, 1, 2, 4, 5, 6]
        else:
            keep = [
                k
                for k in range(n)
                if float(np.max(np.abs(Y[k, :]))) > 1e-12
                or float(np.max(np.abs(Y[:, k]))) > 1e-12
            ]
            if len(keep) != n_expect:
                return None

        Y = Y[np.ix_(keep, keep)]
        buses = case.transformers.buses[i]
        Vb = []
        for bi, bus in enumerate(buses):
            # case.base[*]["kVBase"] is already LN volts (DSSParser stores
            # Bus.kVBase()*1000). Y_pu = Y_siemens * V_i * V_j / Sbase_VA.
            # Multiplying by 1000 again inflated |Y| by 1e6 and systematically
            # depressed Log(v) |V| vs OpenDSS (head-of-feeder bias on IEEE-8500).
            v_ln = float(case.base[bus]["kVBase"])
            for _ in phases[bi]:
                Vb.append(v_ln)
        Vb = np.asarray(Vb, dtype=float)
        if Vb.size != Y.shape[0]:
            return None
        Sbase = float(case.base[buses[0]]["VABase"])
        if Sbase <= 0:
            return None
        return (Y * np.outer(Vb, Vb)) / Sbase
    except Exception:
        return None


def get_transformer_yprim(case,i,pu):

    windings = case.transformers.windings[i]
    # Prefer OpenDSS primitive admittance when the circuit is live (paper-faithful
    # linearization + OpenDSS-consistent transformer component model).
    # IMPORTANT: OpenDSS Yprim embeds the *live OpenDSS* tap. Callers that change
    # case.transformers.taps must push those taps into OpenDSS first (see
    # push_transformer_taps_to_opendss) or retap here — otherwise RegControl is a no-op.
    #
    # Exception — regulators: OpenDSS YPrim at tap≠1 is non-reciprocal and collapses
    # the Log(v) Ytilde solve (voltages →~0.6 pu, RegControl runs to max tap).
    # Use the analytical Tau path so case.transformers.taps actually move |V|.
    if pu and int(windings) == 2:
        name_l = str(case.transformers.name[i]).lower()
        regs = {
            str(x).lower()
            for x in (getattr(case, "regulator_names", None) or [])
        }
        if name_l not in regs:
            Yod = _opendss_transformer_yprim_pu(case, i)
            if Yod is not None:
                return Yod

    kVs = deepcopy(case.transformers.kVs[i])
    kVspu = [1]*len(kVs)
    conns = case.transformers.Conn[i]
    phases =  deepcopy(case.transformers.phases[i])
    # idx = np.hstack(phases)!=0
    idx = np.array([True,False,True,False,False,True])
    buses = case.transformers.buses[i]
    Rs = case.transformers.Rs[i]
    noloadloss = float(case.transformers.noloadloss[i])
    imag = float(case.transformers.imag[i])
    Xs = case.transformers.Xs[i]


    for phase in phases:
        if 0 in phase:
            phase.remove(0)
    nl = min([len(x) for x in phases])

    conn_l = [str(c).lower() for c in conns]
    is_delta_wye = ("delta" in conn_l) and (
        ("wye" in conn_l) or ("wyeg" in conn_l) or ("star" in conn_l)
    )

    for j in range(0, len(kVs)):
        if (not conns[j] == "Delta") & (nl >= 2):
            kVs[j] /= sqrt(3)
            # LN bus bases are already in case.base. For Delta–Wye, do not
            # also inflate kVspu by √3 (that double-counts vs OpenDSS LN bases).
            # Wye–Wye keeps the historical package kVspu=√3 scaling.
            if not (pu and is_delta_wye):
                kVspu[j] = sqrt(3)

    kvar = case.transformers.kVAs[i][0]
    kVbases = [case.base[bus]["kVBase"] for bus in buses]

    if windings == 2:
        Bt = np.array([[1], [-1]])
        if pu:
            Nt = np.array([[kVspu[0], 0], [-kVspu[0], 0], [0, kVspu[1]], [0, -kVspu[1]]])
            # OpenDSS R()/Xhl() are percent. For Delta–Wye convert %→pu without
            # the extra nl factor (kron already expands per phase). Other
            # connections keep the historical package formula.
            if is_delta_wye:
                r_pu = (float(Rs[0]) + float(Rs[1])) / 100.0
                x_pu = float(Xs[0]) / 100.0
                zsc = r_pu + 1j * x_pu
            else:
                # OpenDSS R()/Xhl() are percent for Wye–Wye as well (incl. regulators).
                # Historical package omitted /100 here; that made |Y| ~100× too small so
                # taps barely moved voltages and baking OD taps looked harmful.
                r_pu = (float(Rs[0]) + float(Rs[1])) / 100.0
                x_pu = float(Xs[0]) / 100.0
                zsc = r_pu + 1j * x_pu
                # Percent impedance is on transformer kVA; Log(v) network is system-pu.
                Ssys = float(case.base[buses[0]]["VABase"])
                Sxf = float(kvar) * 1000.0
                if Sxf > 0.0 and Ssys > 0.0:
                    zsc = zsc * (Ssys / Sxf)
        else:
            Nt = np.array(
                [
                    [1 / (kVs[0] * 1000), 0],
                    [-1 / (kVs[0] * 1000), 0],
                    [0, 1 / (kVs[1] * 1000)],
                    [0, -1 / (kVs[1] * 1000)],
                ]
            )
            zsc = (Rs[0] + Rs[1] + 1j * Xs[0]) / (kvar * 1000 / nl)

        Aconn = get_transformer_incidence_matrices(case)
        Abus1 = Aconn["{:}_bus1".format(conns[0])]
        Abus2 = Aconn["{:}_bus2".format(conns[1])]
        if nl<3:
            Abus1 = Abus1[:nl,:4*nl]
            Abus2 = Abus2[:nl,:4*nl]

        A = scipy.sparse.vstack([Abus1,Abus2])
            
        Taubus1 = np.kron(np.eye(Abus1.shape[0],dtype=int), 1/case.transformers.taps[i][0]) ## Inverse of the tap
        Taubus2 = np.kron(np.eye(Abus2.shape[0],dtype=int), 1/case.transformers.taps[i][1]) ## Inverse of the tap
        Tau = block_diag(Taubus1,Taubus2)

        B = np.kron(np.eye(nl,dtype=int),Bt)
        N = np.kron(np.eye(nl,dtype=int),Nt)
        z = np.kron(np.eye(nl,dtype=int),zsc)

        Yprim = A@N@B@inv(z)@B.T@N.T@A.T
        # Analytic path: apply taps (OpenDSS path embeds live DSS taps instead).
        Yprim = Tau @ Yprim @ Tau

    elif windings==3:
        Bt = np.array([[1,1],[-1,0],[0,-1]])
        if pu:
            Nt = np.array([[kVspu[0],0,0],[-kVspu[0],0,0],[0,kVspu[1],0],[0,-kVspu[1],0],[0,0,kVspu[2]],[0,0,-kVspu[2]]])
            zsc = nl*np.array([[Rs[0] + Rs[1] + 1j*Xs[0],Rs[0]+0.5j*(Xs[0]+Xs[1]-Xs[2])],[Rs[0]+0.5j*(Xs[0]+Xs[1]-Xs[2]),Rs[0] + Rs[2]+1j*Xs[1]]])
        else:
            Nt = np.array([[1/(kVs[0]*1000),0,0],[-1/(kVs[0]*1000),0,0],[0,1/(kVs[1]*1000),0],[0,-1/(kVs[1]*1000),0],[0,0,1/(kVs[2]*1000)],[0,0,-1/(kVs[2]*1000)]])
            zsc = np.array([[Rs[0] + Rs[1] + 1j*Xs[0],Rs[0]+0.5j*(Xs[0]+Xs[1]-Xs[2])],[Rs[0]+0.5j*(Xs[0]+Xs[1]-Xs[2]),Rs[0] + Rs[2]+1j*Xs[1]]])/(kvar*1000/nl)

        Yprim = (Nt@Bt@inv(zsc)@Bt.T@Nt.T)[idx,:][:,idx]
        # yoc = noloadloss/100 - 1j*imag/100
        # Yprim[-2:,-2:] += np.array([[yoc,-yoc],[-yoc,yoc]])


    # if np.linalg.matrix_rank(Yprim)<Yprim.shape[0]:
    #     Yprim += np.random.rand(Yprim.shape[0],Yprim.shape[0])/1000000

    return Yprim


def push_transformer_taps_to_opendss(case) -> int:
    """Write case.transformers.taps into the live OpenDSS circuit.

    Required before get_transformer_yprim() so OpenDSS YPrim reflects Log(v) taps.
    Returns number of windings successfully updated.
    """
    n = 0
    if getattr(case, "transformers", None) is None:
        return 0
    for i in range(len(case.transformers)):
        name = str(case.transformers.name[i])
        taps = list(case.transformers.taps[i])
        try:
            dss.Transformers.Name(name)
        except Exception:
            continue
        for wdg, tap in enumerate(taps, start=1):
            try:
                dss.Transformers.Wdg(int(wdg))
                dss.Transformers.Tap(float(tap))
                n += 1
            except Exception:
                continue
    return n

def create_df(case):
    results = pd.DataFrame()
    for algo in ["openDSS","logv3lpf"]:
        vm = []
        va = []
        phase = []
        buses = []
        for bus in case.results[algo]["vm"]:
            phases = case.bus_phases[bus]
            missing_phases = np.setdiff1d([1,2,3],phases)
            vm.append(case.results[algo]["vm"][bus])
            va.append(case.results[algo]["va"][bus])
            phase.append(phases)
            for missing_phase in missing_phases:
                vm.append(np.nan)
                va.append(np.nan)
                phase.append(missing_phase)
            buses.append([bus]*3)
        vm = np.hstack(vm)
        va = np.hstack(va)
        phase = np.hstack(phase)
        buses = np.hstack(buses)
        df = pd.DataFrame(np.vstack([buses,vm,va,phase]).T,columns = ["bus","vm","va","phase"])
        df["algorithm"] = algo
        results = results.append(df)
    results.vm = results.vm.astype(float)
    results.va = results.va.astype(float)

    return results



    