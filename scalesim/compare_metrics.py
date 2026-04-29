"""
compare_metrics.py

Compares hardware profiling metrics from NVIDIA Nsight Compute (NCU)
with cycle-accurate simulation results from SCALE-Sim.

Can be used as a module via run_comparison() or run standalone.
"""

import io
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Parse NCU CSV
# ---------------------------------------------------------------------------
def parse_ncu_csv(path):
    if not os.path.exists(path):
        print(f"  [!] NCU file not found: {path}")
        return None

    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break

    if header_idx is None:
        print("  [!] Could not locate CSV header in NCU output.")
        return None

    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip() for c in df.columns]

    def safe_float(x):
        if x == "n/a" or x is None: return 0.0
        try:
            return float(str(x).replace(",", ""))
        except ValueError:
            return 0.0

    df["value"] = df["Metric Value"].apply(safe_float)

    pivot = df.pivot_table(
        index=["ID", "Kernel Name"],
        columns="Metric Name",
        values="value",
        aggfunc="first",
    ).reset_index()

    conv_keywords = ["gemm", "conv", "fprop", "xmma", "cutlass", "sgemm"]
    exclude_keywords = ["bn_fw", "elementwise", "max_pool", "nchwtonhwc", "nhwctonchw", "offsets"]

    def is_conv(name):
        name_lower = name.lower()
        if "cudnn::bn" in name_lower: return False
        if any(k in name_lower for k in exclude_keywords): return False
        return any(k in name_lower for k in conv_keywords) or ("cudnn" in name_lower and "ampere_scudnn" in name_lower)

    pivot["is_conv"] = pivot["Kernel Name"].apply(is_conv)

    # Grouping logic: Aggregate metrics from following utility kernels into the preceding conv kernel
    grouped_data = []
    current_layer = None

    for _, row in pivot.iterrows():
        l1 = row.get("l1tex__t_sectors.sum", 0)
        l2 = row.get("lts__t_sectors.sum", 0)
        dr = row.get("dram__bytes_read.sum", 0)
        dw = row.get("dram__bytes_write.sum", 0)
        cyc = row.get("sm__cycles_active.avg", 0)
        ten = row.get("sm__pipe_tensor_op_hmma_cycles_active.sum", 0)

        if row["is_conv"]:
            if current_layer is not None:
                grouped_data.append(current_layer)
            current_layer = {
                "cycles": cyc,
                "tensor_cycles": ten,
                "l1": l1,
                "l2": l2,
                "dram_r": dr,
                "dram_w": dw
            }
        elif current_layer is not None:
            current_layer["l1"] += l1
            current_layer["l2"] += l2
            current_layer["dram_r"] += dr
            current_layer["dram_w"] += dw
            # We don't sum cycles/tensor_cycles as they represent the main op latency

    if current_layer is not None:
        grouped_data.append(current_layer)

    # Calculate final sums
    conv_l1 = sum(d["l1"] for d in grouped_data)
    conv_l2 = sum(d["l2"] for d in grouped_data)
    conv_dr = sum(d["dram_r"] for d in grouped_data)
    conv_dw = sum(d["dram_w"] for d in grouped_data)
    conv_cyc = sum(d["cycles"] for d in grouped_data)
    
    # Orin Nano has 8 SMs. hmma.sum is aggregate, cycles.avg is per-SM.
    # We divide by 8 to get the per-SM tensor contribution for a fair latency comparison.
    conv_ten = sum(d["tensor_cycles"] for d in grouped_data) / 8.0

    return {
        "total_kernels":     len(pivot),
        "conv_kernels":      len(grouped_data),
        "total_cycles":      pivot["sm__cycles_active.avg"].sum() if "sm__cycles_active.avg" in pivot.columns else 0,
        "tensor_cycles":     (pivot["sm__pipe_tensor_op_hmma_cycles_active.sum"].sum() / 8.0) if "sm__pipe_tensor_op_hmma_cycles_active.sum" in pivot.columns else 0,
        "l1_sectors":        pivot["l1tex__t_sectors.sum"].sum() if "l1tex__t_sectors.sum" in pivot.columns else 0,
        "l2_sectors":        pivot["lts__t_sectors.sum"].sum() if "lts__t_sectors.sum" in pivot.columns else 0,
        "dram_read_bytes":   pivot["dram__bytes_read.sum"].sum() if "dram__bytes_read.sum" in pivot.columns else 0,
        "dram_write_bytes":  pivot["dram__bytes_write.sum"].sum() if "dram__bytes_write.sum" in pivot.columns else 0,
        "conv_cycles":       conv_cyc,
        "conv_tensor_cycles":conv_ten,
        "conv_l1_sectors":   conv_l1,
        "conv_l2_sectors":   conv_l2,
        "dram_read_bytes_conv": conv_dr,
        "dram_write_bytes_conv": conv_dw
    }


# ---------------------------------------------------------------------------
# 2. Parse SCALE-Sim outputs
# ---------------------------------------------------------------------------
def parse_scalesim_outputs(report_dir):
    if not os.path.isdir(report_dir):
        print(f"  [!] SCALE-Sim output directory not found: {report_dir}")
        return None

    result = {}

    compute_path = os.path.join(report_dir, "COMPUTE_REPORT.csv")
    if os.path.exists(compute_path):
        df = pd.read_csv(compute_path)
        df.columns = [c.strip() for c in df.columns]
        cycle_col       = [c for c in df.columns if "Cycles" in c and "incl" in c.lower()]
        if not cycle_col:
            cycle_col   = [c for c in df.columns if "Cycles" in c and "Stall" not in c]
        compute_cycle_col = [c for c in df.columns if c.strip() == "Total Cycles"]
        util_col        = [c for c in df.columns if "Overall Util" in c]
        compute_util_col= [c for c in df.columns if "Compute Util" in c]
        stall_col       = [c for c in df.columns if "Stall" in c]
        mapping_col     = [c for c in df.columns if "Mapping" in c]

        result["total_cycles"]      = int(df[cycle_col[0]].sum())        if cycle_col        else 0
        result["compute_cycles"]    = int(df[compute_cycle_col[0]].sum()) if compute_cycle_col else 0
        result["overall_util"]      = float(df[util_col[0]].mean())      if util_col         else 0.0
        result["compute_util"]      = float(df[compute_util_col[0]].mean()) if compute_util_col else 0.0
        result["total_stall_cycles"]= int(df[stall_col[0]].sum())        if stall_col        else 0
        result["mapping_efficiency"]= float(df[mapping_col[0]].mean())   if mapping_col      else 0.0
        result["num_layers"]        = len(df)
        result["compute_df"]        = df
    else:
        print(f"  [!] {compute_path} not found")

    access_path = os.path.join(report_dir, "DETAILED_ACCESS_REPORT.csv")
    if os.path.exists(access_path):
        df = pd.read_csv(access_path)
        df.columns = [c.strip() for c in df.columns]
        sram_read_cols  = [c for c in df.columns if "SRAM" in c and "Reads"  in c]
        sram_write_cols = [c for c in df.columns if "SRAM" in c and "Writes" in c]
        dram_read_cols  = [c for c in df.columns if "DRAM" in c and "Reads"  in c]
        dram_write_cols = [c for c in df.columns if "DRAM" in c and "Writes" in c]

        result["sram_reads"]  = int(df[sram_read_cols].sum().sum())  if sram_read_cols  else 0
        result["sram_writes"] = int(df[sram_write_cols].sum().sum()) if sram_write_cols else 0
        result["dram_reads"]  = int(df[dram_read_cols].sum().sum())  if dram_read_cols  else 0
        result["dram_writes"] = int(df[dram_write_cols].sum().sum()) if dram_write_cols else 0
        result["access_df"]   = df

    bw_path = os.path.join(report_dir, "BANDWIDTH_REPORT.csv")
    if os.path.exists(bw_path):
        df = pd.read_csv(bw_path)
        df.columns = [c.strip() for c in df.columns]
        result["bw_df"] = df

    return result


# ---------------------------------------------------------------------------
# 3. Formatting helpers
# ---------------------------------------------------------------------------
def fmt(v, unit=""):
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if v == int(v):
            return f"{int(v):,}{unit}"
        return f"{v:,.2f}{unit}"
    return f"{v:,}{unit}"


def pct_diff(hw, sim):
    try:
        hw_f, sim_f = float(hw), float(sim)
        if hw_f == 0:
            return "N/A"
        return f"{((sim_f - hw_f) / hw_f) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# 4. Build comparison rows (shared between stdout and file)
# ---------------------------------------------------------------------------
def _build_rows(hw, sim):
    rows = []
    if hw and sim:
        rows.append(("Total Compute Cycles",
                     hw.get("conv_cycles"),
                     sim.get("total_cycles")))

        # HW SRAM is in sectors (32 bytes). SIM SRAM is in words (assume 1 byte).
        hw_sram  = (hw.get("conv_l1_sectors", 0) + hw.get("conv_l2_sectors", 0)) * 32
        sim_sram = sim.get("sram_reads", 0) + sim.get("sram_writes", 0)
        rows.append(("On-chip Memory Access (Bytes)", hw_sram, sim_sram))

        # dram_hw  = hw.get("dram_read_bytes", float("nan")) + hw.get("dram_write_bytes", float("nan"))
        # hw_dram  = "N/A (unified mem)" if dram_hw != dram_hw else dram_hw
        # sim_dram = sim.get("dram_reads", 0) + sim.get("dram_writes", 0)
        # rows.append(("Off-chip (DRAM) Access (Bytes)", hw_dram, sim_dram))

        rows.append(("Tensor Compute Cycles",
                     hw.get("conv_tensor_cycles"),
                     sim.get("compute_cycles")))
    return rows


# ---------------------------------------------------------------------------
# 5. Public entry point
# ---------------------------------------------------------------------------
def run_comparison(ncu_path, scalesim_dir, report_path):
    """
    Compare NCU hardware metrics against SCALE-Sim outputs and write a report.

    Parameters
    ----------
    ncu_path     : str  - path to the NCU CSV file
    scalesim_dir : str  - directory containing SCALE-Sim report CSVs
    report_path  : str  - where to write comparison_report.txt
    """
    print("\n" + "=" * 72)
    print("  Hardware vs SCALE-Sim Comparison Report")
    print("=" * 72)

    print("\n[1/3]  Parsing hardware metrics ...")
    hw = parse_ncu_csv(ncu_path)

    print("[2/3]  Parsing SCALE-Sim results ...")
    sim = parse_scalesim_outputs(scalesim_dir)

    if hw is None and sim is None:
        print("\n  ERROR: No data to compare.")
        return

    rows = _build_rows(hw, sim)

    # ---- Print to stdout ----
    if hw:
        print("\n" + "-" * 72)
        print("  HARDWARE PROFILING SUMMARY  (Nsight Compute)")
        print("-" * 72)
        print(f"  Total GPU kernel invocations : {hw['total_kernels']}")
        print(f"  Conv/GEMM kernels            : {hw['conv_kernels']}")
        print(f"  Total SM cycles              : {fmt(hw['total_cycles'])}")
        print(f"  Conv-only SM cycles          : {fmt(hw['conv_cycles'])}")
        print(f"  Tensor Core cycles (all)     : {fmt(hw['tensor_cycles'])}")
        print(f"  Tensor Core cycles (conv)    : {fmt(hw['conv_tensor_cycles'])}")
        print(f"  L1 sectors (conv)            : {fmt(hw['conv_l1_sectors'])}")
        print(f"  L2 sectors (conv)            : {fmt(hw['conv_l2_sectors'])}")

    if sim:
        print("\n" + "-" * 72)
        print("  SCALE-Sim SIMULATION SUMMARY")
        print("-" * 72)
        print(f"  Layers simulated             : {sim.get('num_layers', 'N/A')}")
        print(f"  Total cycles                 : {fmt(sim.get('total_cycles'))}")
        print(f"  Total stall cycles           : {fmt(sim.get('total_stall_cycles'))}")
        print(f"  Avg overall utilization      : {fmt(sim.get('overall_util'), '%')}")
        print(f"  Avg compute utilization      : {fmt(sim.get('compute_util'), '%')}")
        print(f"  Avg mapping efficiency       : {fmt(sim.get('mapping_efficiency'), '%')}")
        print(f"  Total SRAM reads (words)     : {fmt(sim.get('sram_reads'))}")
        print(f"  Total SRAM writes (words)    : {fmt(sim.get('sram_writes'))}")
        print(f"  Total DRAM reads (words)     : {fmt(sim.get('dram_reads'))}")
        print(f"  Total DRAM writes (words)    : {fmt(sim.get('dram_writes'))}")

        if "compute_df" in sim:
            print("\n  Per-layer breakdown:")
            cdf = sim["compute_df"]
            for _, row in cdf.iterrows():
                lid  = int(row.iloc[0])
                cyc  = int(row.iloc[1])
                util = float(row.iloc[4])
                print(f"    Layer {lid:2d}:  {cyc:>10,} cycles,  {util:5.1f}% util")

    print("\n" + "=" * 72)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    header = f"{'Metric':<36} {'Hardware (NCU)':>18} {'SCALE-Sim':>18} {'Change':>10}"
    print(header)
    print("-" * 82)
    for label, hw_val, sim_val in rows:
        delta = pct_diff(hw_val, sim_val) if not isinstance(hw_val, str) and not isinstance(sim_val, str) else "-"
        print(f"  {label:<34} {fmt(hw_val):>18} {fmt(sim_val):>18} {delta:>10}")

    # ---- Write report file ----
    print(f"\n[3/3]  Writing comparison report ...")
    buf = io.StringIO()
    orig = sys.stdout
    sys.stdout = buf

    print("=" * 72)
    run_name = os.path.basename(scalesim_dir.rstrip('/\\'))
    print(f"  {run_name} - Hardware vs SCALE-Sim Comparison Report")
    print("=" * 72)
    print(f"\n  Hardware profiling: {ncu_path}")
    print(f"  Simulation results: {scalesim_dir}/")

    if hw:
        print(f"\n  --- Hardware (conv kernels) ---")
        print(f"  SM cycles        : {fmt(hw.get('conv_cycles'))}")
        print(f"  Tensor cycles    : {fmt(hw.get('conv_tensor_cycles'))}")
        print(f"  L1 sectors       : {fmt(hw.get('conv_l1_sectors'))}")
        print(f"  L2 sectors       : {fmt(hw.get('conv_l2_sectors'))}")

    if sim:
        print(f"\n  --- SCALE-Sim ---")
        print(f"  Total cycles     : {fmt(sim.get('total_cycles'))}")
        print(f"  Avg utilization  : {fmt(sim.get('overall_util'), '%')}")
        print(f"  SRAM reads       : {fmt(sim.get('sram_reads'))}")
        print(f"  SRAM writes      : {fmt(sim.get('sram_writes'))}")
        print(f"  DRAM reads       : {fmt(sim.get('dram_reads'))}")
        print(f"  DRAM writes      : {fmt(sim.get('dram_writes'))}")

    if hw and sim:
        print(f"\n  --- Comparison ---\n")
        header = f"  {'Metric':<34} {'Hardware (NCU)':>18} {'SCALE-Sim':>18} {'Change':>10}"
        print(header)
        print(f"  {'-' * 83}")
        for label, hw_val, sim_val in rows:
            delta = pct_diff(hw_val, sim_val) if not isinstance(hw_val, str) and not isinstance(sim_val, str) else "-"
            print(f"  {label:<34} {fmt(hw_val):>18} {fmt(sim_val):>18} {delta:>10}")

    sys.stdout = orig
    report_text = buf.getvalue()

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ncu",    required=True, help="Path to NCU CSV file")
    p.add_argument("--simdir", required=True, help="Path to SCALE-Sim output directory")
    p.add_argument("--out",    default="comparison_report.txt", help="Output report path")
    a = p.parse_args()
    run_comparison(a.ncu, a.simdir, a.out)
