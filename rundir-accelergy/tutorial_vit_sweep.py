#!/usr/bin/env python3
"""
SCALE-Sim + Accelergy tutorial sweep.

Runs the full SCALE-Sim -> Accelergy energy-estimation flow for a small ViT
workload across:
    * 3 systolic-array sizes : 32x32, 64x64, 128x128
    * 3 dataflows            : OS, WS, IS
(= 9 design points), then collects latency (cycles) + energy (pJ) for each
point and produces grouped bar plots plus a results CSV.

Run from inside the `rundir-accelergy` directory:

    $ cd rundir-accelergy
    $ python3 tutorial_vit_sweep.py

Prereqs (see README_accelergy.md):
    * SCALE-Sim (this repo)
    * accelergy installed and on PATH, with at least one estimation plug-in
      (e.g. accelergy-aladdin-plug-in)
    * python deps: pyyaml, matplotlib, numpy
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

import yaml

# rundir-accelergy is the directory this script lives in; run_all.sh assumes
# it is the working directory.
RUNDIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='../topologies/ispass25_models/vit_s.csv',
                   help='Topology CSV (relative to rundir-accelergy). Default: small ViT (vit_s).')
    p.add_argument('--input-type', default='gemm', choices=['gemm', 'conv'],
                   help='Topology type. vit_s is a GEMM (M,N,K) topology.')
    p.add_argument('--array-sizes', type=int, nargs='+', default=[32, 64, 128],
                   help='Square systolic-array dimensions to sweep.')
    p.add_argument('--dataflows', nargs='+', default=['os', 'ws', 'is'],
                   help='Dataflows to sweep (os/ws/is).')
    p.add_argument('--base-cfg', default='scale.cfg',
                   help='Base config template (in rundir-accelergy) to derive sweep configs from.')
    p.add_argument('--out-dir', default='tutorial_results',
                   help='Directory (under rundir-accelergy) for all outputs.')
    p.add_argument('--skip-run', action='store_true',
                   help='Skip simulation; only re-parse + re-plot existing results.')
    return p.parse_args()


def make_config(base_cfg_path, array_size, dataflow, run_name, dest_path):
    """Write a sweep config derived from the base template, overriding the
    array dimensions, dataflow and run_name."""
    with open(base_cfg_path) as f:
        lines = f.readlines()

    out = []
    for line in lines:
        if re.match(r'\s*run_name\s*=', line):
            out.append(f'run_name = {run_name}\n')
        elif re.match(r'\s*ArrayHeight\s*:', line):
            out.append(f'ArrayHeight:    {array_size}\n')
        elif re.match(r'\s*ArrayWidth\s*:', line):
            out.append(f'ArrayWidth:    {array_size}\n')
        elif re.match(r'\s*Dataflow\s*:', line):
            out.append(f'Dataflow : {dataflow}\n')
        else:
            out.append(line)
    with open(dest_path, 'w') as f:
        f.writelines(out)


def run_point(cfg_path, model, scsim_log, out_dir, input_type, log_path):
    """Run the full run_all.sh flow for one (size, dataflow) point."""
    cmd = ['bash', './run_all.sh',
           '-c', cfg_path,
           '-t', model,
           '-p', scsim_log,
           '-o', out_dir,
           '-i', input_type]
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, cwd=RUNDIR, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def total_energy_pj(energy_yaml):
    """Sum component energies (pJ) from an Accelergy energy_estimation.yaml."""
    with open(energy_yaml) as f:
        data = yaml.safe_load(f)
    comps = data['energy_estimation']['components']
    return float(sum(c['energy'] for c in comps))


def energy_breakdown_pj(energy_yaml):
    """Aggregate component energies (pJ) by component type (leaf name)."""
    with open(energy_yaml) as f:
        data = yaml.safe_load(f)
    agg = {}
    for c in data['energy_estimation']['components']:
        key = c['name'].split('.')[-1].split('[')[0]
        agg[key] = agg.get(key, 0.0) + float(c['energy'])
    return agg


def total_cycles(compute_report_csv):
    """Sum 'Total Cycles (incl. prefetch)' (col 1) over all layers = latency."""
    total = 0
    with open(compute_report_csv) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            row = [c.strip() for c in row if c.strip() != '']
            if len(row) >= 2:
                total += int(row[1])
    return total


def main():
    args = parse_args()

    base_cfg = os.path.join(RUNDIR, args.base_cfg)
    out_dir = os.path.join(RUNDIR, args.out_dir)
    cfg_dir = os.path.join(out_dir, 'configs')
    log_dir = os.path.join(out_dir, 'logs')
    scsim_log = os.path.join(out_dir, 'scalesim_raw')
    for d in (out_dir, cfg_dir, log_dir, scsim_log):
        os.makedirs(d, exist_ok=True)

    model_tag = os.path.splitext(os.path.basename(args.model))[0]

    results = []   # list of dicts: size, dataflow, cycles, energy_pj, breakdown
    for size in args.array_sizes:
        for df in args.dataflows:
            run_name = f'{model_tag}_{size}x{size}_{df}'
            acc_yaml = os.path.join(out_dir, f'accelergy_output_{run_name}', 'energy_estimation.yaml')
            comp_csv = os.path.join(out_dir, f'scale_sim_output_{run_name}', 'COMPUTE_REPORT.csv')

            if not args.skip_run:
                print(f'[run ] {run_name} ...', flush=True)
                cfg_path = os.path.join(cfg_dir, f'{run_name}.cfg')
                make_config(base_cfg, size, df, run_name, cfg_path)
                log_path = os.path.join(log_dir, f'{run_name}.log')
                rc = run_point(cfg_path, args.model, scsim_log, out_dir,
                               args.input_type, log_path)
                if rc != 0:
                    print(f'[WARN] {run_name} exited with code {rc} '
                          f'(accelergy energy step may have failed) - see {log_path}')

            if not (os.path.exists(acc_yaml) and os.path.exists(comp_csv)):
                print(f'[SKIP] {run_name}: missing outputs '
                      f'(energy_estimation.yaml or COMPUTE_REPORT.csv)')
                continue

            energy = total_energy_pj(acc_yaml)
            cycles = total_cycles(comp_csv)
            results.append({
                'size': size,
                'dataflow': df,
                'cycles': cycles,
                'energy_pj': energy,
                'edp': energy * cycles,
                'breakdown': energy_breakdown_pj(acc_yaml),
            })
            print(f'[ ok ] {run_name}: cycles={cycles:,}  energy={energy/1e6:.3f} uJ')

    if not results:
        print('No results collected. Aborting.', file=sys.stderr)
        sys.exit(1)

    write_csv(results, os.path.join(out_dir, f'{model_tag}_sweep_results.csv'))
    plot_results(results, args.array_sizes, args.dataflows, model_tag,
                 os.path.join(out_dir, f'{model_tag}_sweep.png'))


def write_csv(results, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['array_size', 'dataflow', 'total_cycles',
                    'total_energy_pJ', 'total_energy_uJ', 'EDP_pJ_cycles'])
        for r in results:
            w.writerow([f"{r['size']}x{r['size']}", r['dataflow'].upper(),
                        r['cycles'], f"{r['energy_pj']:.3f}",
                        f"{r['energy_pj']/1e6:.6f}", f"{r['edp']:.3f}"])
    print(f'\nWrote results CSV -> {path}')


def plot_results(results, sizes, dataflows, model_tag, path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # index results by (size, dataflow)
    idx = {(r['size'], r['dataflow']): r for r in results}
    size_labels = [f'{s}x{s}' for s in sizes]
    x = np.arange(len(sizes))
    width = 0.8 / max(len(dataflows), 1)
    colors = {'os': '#4C72B0', 'ws': '#DD8452', 'is': '#55A868'}

    metrics = [
        ('cycles',     lambda r: r['cycles'],          'Latency (cycles)',          'Total Latency'),
        ('energy_uj',  lambda r: r['energy_pj'] / 1e6, 'Energy (uJ)',               'Total Energy'),
        ('edp',        lambda r: r['edp'] / 1e6,       'EDP (uJ x cycles)',         'Energy-Delay Product'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (_key, getter, ylabel, title) in zip(axes, metrics):
        for di, df in enumerate(dataflows):
            vals = [getter(idx[(s, df)]) if (s, df) in idx else 0 for s in sizes]
            offset = (di - (len(dataflows) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=df.upper(),
                   color=colors.get(df, None))
        ax.set_xticks(x)
        ax.set_xticklabels(size_labels)
        ax.set_xlabel('Systolic Array Size')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title='Dataflow')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(f'SCALE-Sim + Accelergy sweep  |  workload: {model_tag}',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    print(f'Wrote plot        -> {path}')


if __name__ == '__main__':
    main()
