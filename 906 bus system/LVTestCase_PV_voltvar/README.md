# LVTestCase + Volt-Var PV (copy)

Stock original under `OpenDSS-master/.../IEEETestCases/LVTestCase` is **unchanged**.

`LoadShapes.txt` still references `Daily_1min_100profiles/` relatively. That folder is **gitignored** here (avoid duplicating 100 profiles); compile helpers link or abs-path to the stock `IEEETestCases/LVTestCase/Daily_1min_100profiles` so Colab clones work without a local copy.

## Files
- `Master_snapshot_PV_voltvar.dss` — snapshot + `ControlMode=STATIC`
- `Master_PV_voltvar.dss` — yearly demand-interval + Volt-Var
- `PV_voltvar_906.dss` — `PVSystem.PV906` @ `906.1` + `PVSystem.PV458` @ `458.3` + InvControls

## Inverters
| Name | Bus | Load | Notes |
|------|-----|------|-------|
| **PV906** | **906.1** | LOAD55 | Remote end, phase A |
| **PV458** | **458.3** | LOAD24 | Mid-lateral; map-far from bus 1 and 906 (replaces prior PV34 @ 34.1) |

Shared Volt-Var:
- **Pmpp = 12 kW**, **kVA = 15** each (~25% oversize)
- Deadband **0.98–1.02 pu**, **|Q| = 0.44 × kVA** (VARMAX)
- Curve `vv_curve_044`: `[0.92, 0.98, 1.02, 1.08]` → `[+0.44, 0, 0, −0.44]`

## Sizing
Run from repo root (currently sweeps **PV906** only):

```bash
python size_906_pv_voltvar.py
```

Results: `outputs/906_pv_voltvar_sizing.json` / `.csv`.

**Finding:** on this feeder stock voltages are already high (~1.03–1.05). Active power raises remote V; Volt-Var **absorbs Q** and cuts overvoltage vs the same-kW MPPT unit. Absolute MAE vs no-PV can still worsen if Pmpp is large — 12 kW is the balanced pick for PV906; PV458 uses the same starting size.
