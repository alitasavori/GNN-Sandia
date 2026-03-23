import json
from pathlib import Path


def main() -> None:
    nb_path = Path("GNN2.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell = nb["cells"][37]
    src = "".join(cell.get("source", []))

    if "open_dss_solve_s = 0.0" not in src:
        raise RuntimeError("Expected speed benchmark cell at index 37 (marker not found).")

    # Insert iteration tracking near the OpenDSS timing loop.
    src = src.replace(
        "open_dss_solve_s = 0.0\n"
        "for t in range(inj.NPTS):\n",
        "open_dss_solve_s = 0.0\n"
        "open_dss_ctrl_iters = []\n"
        "open_dss_pf_iters = []\n"
        "for t in range(inj.NPTS):\n",
    )

    # After Solve(), record iterations for converged steps (best-effort).
    marker = (
        "    inj.dss.Solution.Solve()\n"
        "    open_dss_solve_s += time.perf_counter() - t_s\n"
    )
    if marker not in src:
        raise RuntimeError("Could not find OpenDSS Solve() timing marker to patch.")

    src = src.replace(
        marker,
        marker
        + "    if inj.dss.Solution.Converged():\n"
        + "        try:\n"
        + "            open_dss_ctrl_iters.append(int(inj.dss.Solution.ControlIterations()))\n"
        + "        except Exception:\n"
        + "            pass\n"
        + "        try:\n"
        + "            open_dss_pf_iters.append(int(inj.dss.Solution.Iterations()))\n"
        + "        except Exception:\n"
        + "            pass\n",
    )

    # Print averages after OpenDSS section.
    print_marker = "print(\"  solve-only time: %.4f s | mean/step: %.3f ms\" % (open_dss_solve_s, 1000*open_dss_solve_s/inj.NPTS))\n\n"
    if print_marker not in src:
        raise RuntimeError("Could not find OpenDSS print marker to patch.")

    src = src.replace(
        print_marker,
        print_marker
        + "if len(open_dss_ctrl_iters) > 0:\n"
        + "    print(\"  avg ControlIterations (converged): %.2f over %d steps\" % (float(np.mean(open_dss_ctrl_iters)), len(open_dss_ctrl_iters)))\n"
        + "else:\n"
        + "    print(\"  avg ControlIterations (converged): n/a\")\n"
        + "if len(open_dss_pf_iters) > 0:\n"
        + "    print(\"  avg PF Iterations (converged): %.2f over %d steps\" % (float(np.mean(open_dss_pf_iters)), len(open_dss_pf_iters)))\n"
        + "else:\n"
        + "    print(\"  avg PF Iterations (converged): n/a\")\n\n",
    )

    cell["source"] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Updated OpenDSS iteration reporting in cell 37.")


if __name__ == "__main__":
    main()

