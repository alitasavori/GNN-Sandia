from logv3lpf.math_functions import polar_to_cartesian


def RegControl(
    Vreg,
    band,
    Vn,
    Nt,
    In,
    Ct,
    R,
    X,
    *,
    Vbase_V: float | None = None,
    max_tap_change: int = 1,
):
    """OpenDSS-style forward RegControl tap delta (integer steps).

    Matches OpenDSS ``RegControl.pas`` LDC:
      Vcontrol = Vwinding/PT + (R+jX) * (I_into_winding / CT)
      Vactual  = |Vcontrol|
    (I is into the monitored terminal; adding VLDC ≡ Vterm − (R+jX)·I_load.)

    Do **not** use ``|V| − |(R+jX)I|`` — that always under-reads and pegs max tap.

    When ``Vbase_V`` is set, steps follow OpenDSS
    ``Round((Vreg-Vactual)*PT/Vbase / 0.00625)``, clipped by ``max_tap_change``.
    """
    ct = float(Ct) if abs(float(Ct)) > 1e-12 else 1.0
    v_pt = Vn / float(Nt)
    v_ldc = ((float(R) + 1j * float(X)) / ct) * In
    vr = abs(v_pt + v_ldc)
    half = float(band) / 2.0
    err = float(Vreg) - float(vr)
    if abs(err) <= half:
        return 0

    m = max(int(max_tap_change), 1)
    if Vbase_V is not None and float(Vbase_V) > 0.0:
        boost_pu = float(err) * float(Nt) / float(Vbase_V)
        n = int(round(boost_pu / 0.00625))
        if n == 0:
            n = 1 if err > 0.0 else -1
        return int(max(-m, min(m, n)))
    return 1 if err > 0.0 else -1


def _transformer_index(case, xfmr_name: str):
    """Resolve transformer row for a RegControl target name (case-insensitive)."""
    want = str(xfmr_name).strip().lower()
    names = case.transformers.name.astype(str)
    hit = case.transformers.index[names.str.lower() == want]
    if len(hit) == 0:
        return None
    return int(hit[0])


def apply_regcontrols(case):
    tap_changes = []
    for i in range(len(case.regcontrols)):
        # Prefer explicit transformer target; fall back to RegControl name.
        xfmr = (
            case.regcontrols.transformer[i]
            if "transformer" in case.regcontrols.columns
            else case.regcontrols.name[i]
        )
        idx = _transformer_index(case, xfmr)
        if idx is None:
            # Last resort: RegControl name == transformer name
            idx = _transformer_index(case, case.regcontrols.name[i])
        if idx is None:
            tap_changes.append(0)
            continue

        Vreg = case.regcontrols.vreg[i]
        band = case.regcontrols.band[i]
        Nt = case.regcontrols.nt[i]
        Ct = case.regcontrols.ct[i]
        R = case.regcontrols.R[i]
        X = case.regcontrols.X[i]

        buses = case.transformers.buses[idx]
        phases = case.transformers.phases[idx]
        n_wdg = len(buses)
        # OpenDSS RegControl.winding is 1-based; almost always winding=2 on LTCs.
        if "winding" in case.regcontrols.columns:
            wdg = int(case.regcontrols.winding[i])
        else:
            wdg = 2 if n_wdg >= 2 else 1
        wdg = max(1, min(wdg, n_wdg))
        wdg_i = wdg - 1

        monitoring_bus = buses[wdg_i]
        taps = list(case.transformers.taps[idx])
        tap_phase = phases[wdg_i][0]
        idx_monitor = case.bus_phases[monitoring_bus].index(tap_phase)

        # case.base[*]["kVBase"] is stored in volts (DSSParser: Bus.kVBase()*1000).
        Vbase_V = case.base[monitoring_bus]["kVBase"]

        vn = case.results["logv3lpf"]["vm"][monitoring_bus][idx_monitor]
        van = case.results["logv3lpf"]["va"][monitoring_bus][idx_monitor]
        Vn = polar_to_cartesian(vn, van) * Vbase_V

        # LDC uses current on the regulated winding (same terminal as voltage).
        I_all = case.get_transformer_current(str(case.transformers.name[idx]), False)
        i0 = sum(len(phases[j]) for j in range(wdg_i))
        if I_all is None or len(I_all) <= i0:
            In = 0.0
            R, X = 0.0, 0.0
            Ct_eff = 1.0
        else:
            In = I_all[i0]
            Ct_eff = float(Ct) if abs(float(Ct)) > 1e-12 else 1.0
            if abs(float(Ct)) <= 1e-12:
                R, X = 0.0, 0.0

        step = RegControl(
            Vreg,
            band,
            Vn,
            Nt,
            In,
            Ct_eff,
            R,
            X,
            Vbase_V=float(Vbase_V),
            max_tap_change=(
                int(case.regcontrols.max_tap[i])
                if "max_tap" in case.regcontrols.columns
                else 1
            ),
        )

        # Anti-hunt: cancel an immediate reverse step (common with simultaneous
        # multi-regulator updates on the linearized model).
        if "last_step" in case.regcontrols.columns:
            try:
                prev = int(case.regcontrols.last_step[i])
            except Exception:
                prev = 0
            if prev != 0 and step != 0 and prev * step < 0:
                step = 0

        max_tap = (
            float(case.transformers.max_tap[idx])
            if "max_tap" in case.transformers.columns
            else 1.1
        )
        min_tap = (
            float(case.transformers.min_tap[idx])
            if "min_tap" in case.transformers.columns
            else 0.9
        )
        new_tap = float(taps[wdg_i]) + step * 0.00625
        new_tap = min(max(new_tap, min_tap), max_tap)
        if abs(new_tap - float(taps[wdg_i])) < 1e-12:
            step = 0
        else:
            taps[wdg_i] = new_tap
            case.transformers.at[idx, "taps"] = taps
            if "tap" in case.regcontrols.columns:
                case.regcontrols.at[i, "tap"] = new_tap
            # Keep live OpenDSS YPrim in sync for the next matrix rebuild.
            try:
                import opendssdirect as dss

                dss.Transformers.Name(str(case.transformers.name[idx]))
                dss.Transformers.Wdg(int(wdg))
                dss.Transformers.Tap(float(new_tap))
            except Exception:
                pass

        if "last_step" in case.regcontrols.columns:
            try:
                case.regcontrols.at[i, "last_step"] = int(step)
            except Exception:
                pass

        tap_changes.append(step)

    return tap_changes


def apply_controls(case):
    return apply_regcontrols(case)
