"""
Recompute TOPS / TOPS-per-W / TOPS-per-mm2 for the fused-PE array from
mx_pe_eval_raw_metrics.csv (SCALE-Sim-derived, synthesis-independent) and the
per-PE synthesis numbers below.

Edit PE_VARIANTS with new synthesis results and rerun -- no SCALE-Sim re-run needed,
since the workload cycle counts and MAC totals do not depend on the PE's clock,
power, or area.
"""
import csv
import math

RAW_METRICS_CSV = "mx_pe_eval_raw_metrics.csv"
OUTPUT_TXT = "mx_pe_eval_results.txt"

# N (fusion width) -> per-PE synthesis results
PE_VARIANTS = {
    4: dict(clk_ns=0.56, area_um2=11302.577963, power_w=7.027e-03),
    8: dict(clk_ns=0.64, area_um2=15147.971957, power_w=9.9077e-03),
    16:dict(clk_ns=0.62, area_um2=19931.183962, power_w= 15.0800e-03),
}


def projected_cycles(total_cycles, stall_cycles, n):
    compute_only = total_cycles - stall_cycles
    return stall_cycles + math.ceil(compute_only / n)


def main():
    with open(RAW_METRICS_CSV, newline="") as f:
        raw_rows = list(csv.DictReader(f))

    results = []
    for r in raw_rows:
        num_pe = int(r["array_rows"]) * int(r["array_cols"])
        total_macs = int(r["total_macs"])
        total_cycles = int(r["baseline_cycles_N1"])
        stall_cycles = int(r["baseline_stall_cycles"])

        row = {"workload": r["workload"], "layers": r["layers"], "total_macs": total_macs}
        for n, pe in PE_VARIANTS.items():
            cyc = projected_cycles(total_cycles, stall_cycles, n)
            clk_s = pe["clk_ns"] * 1e-9
            latency_s = cyc * clk_s
            freq_hz = 1.0 / clk_s
            achieved_tops = (2 * total_macs / latency_s) / 1e12
            peak_tops = (2 * n * num_pe * freq_hz) / 1e12
            power_array_w = pe["power_w"] * num_pe
            area_array_mm2 = pe["area_um2"] * num_pe / 1e6

            row[f"N{n}_latency_ms"] = latency_s * 1e3
            row[f"N{n}_achieved_TOPS"] = achieved_tops
            row[f"N{n}_peak_TOPS"] = peak_tops
            row[f"N{n}_util_%"] = 100.0 * achieved_tops / peak_tops
            row[f"N{n}_TOPS_per_W"] = achieved_tops / power_array_w
            row[f"N{n}_TOPS_per_mm2"] = achieved_tops / area_array_mm2
        results.append(row)

    headers = list(results[0].keys())
    col_w = {h: max(len(h), *(len(f'{r[h]:.4f}' if isinstance(r[h], float) else str(r[h])) for r in results)) for h in headers}

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = []
    lines.append("  ".join(h.ljust(col_w[h]) for h in headers))
    for r in results:
        lines.append("  ".join(fmt(r[h]).ljust(col_w[h]) for h in headers))

    text = "\n".join(lines)
    print(text)
    with open(OUTPUT_TXT, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
