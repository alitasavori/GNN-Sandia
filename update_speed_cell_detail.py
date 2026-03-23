import json
from pathlib import Path


def main() -> None:
    nb_path = Path("GNN2.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell = nb["cells"][37]
    src = "".join(cell.get("source", []))

    # 1) Ensure we import timing helpers at the top of the cell.
    if "run_gnn3_timing_comparison import timing_one_block_detailed, print_timing" not in src:
        src = src.replace(
            "from run_gnn3_overlay_7 import load_model_for_inference\n"
            "from compare_two_models_daily import build_x_for_model, _resolve_node_list, _get_full_master_node_list\n\n",
            "from run_gnn3_overlay_7 import load_model_for_inference\n"
            "from compare_two_models_daily import build_x_for_model, _resolve_node_list, _get_full_master_node_list\n"
            "from run_gnn3_timing_comparison import timing_one_block_detailed, print_timing\n\n",
        )

    # 2) Append detailed timing per model after existing summary prints.
    summary_marker = (
        "print(\"  OpenDSS solve total: %.4f s | mean/step: %.3f ms\" % (open_dss_solve_s, 1000*open_dss_solve_s/inj.NPTS))\n"
        "for ckpt_path, t_model in results:\n"
        "    print(\"  %s: total=%.4fs | mean/step=%.3fms\" % (os.path.basename(ckpt_path), t_model, 1000*t_model/inj.NPTS))\n"
    )
    if summary_marker not in src:
        raise RuntimeError("Could not find summary print block to extend.")

    detailed_block = (
        summary_marker
        + "\n"
        "# Optional: detailed per-step timing breakdown (same format as run_gnn3_timing_comparison.py)\n"
        "print(\"\\nPer-step detailed timing for each model (OpenDSS vs GNN):\")\n"
        "for idx, (ckpt_path, t_model) in enumerate(results, 1):\n"
        "    print(\"\\n\" + \"=\" * 72)\n"
        "    print(f\"DETAILED TIMING FOR MODEL {idx}: {os.path.basename(ckpt_path)}\")\n"
        "    dss_steps, gnn_steps, is_deltav, _th, _vd, _vg, _cfg, _static = timing_one_block_detailed(\n"
        "        ckpt_path,\n"
        "        device=device,\n"
        "        block_id=idx,\n"
        "        use_batched_gnn=True,\n"
        "        pv_scale=1.0,\n"
        "    )\n"
        "    print_timing(idx, str(device).upper(), dss_steps, gnn_steps, is_deltav)\n"
    )

    src = src.replace(summary_marker, detailed_block)
    cell["source"] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Extended speed benchmark cell with detailed per-step timing.")


if __name__ == "__main__":
    main()

