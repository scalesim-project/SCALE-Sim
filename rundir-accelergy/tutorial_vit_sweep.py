#!/usr/bin/env python3
"""
SCALE-Sim + Accelergy tutorial: per-workload energy distribution (WS dataflow).

Runs the full SCALE-Sim -> Accelergy energy-estimation flow on the
Weight-Stationary (WS) dataflow for three ViT workloads, across 3 systolic-array
sizes (32x32, 64x64, 128x128), then makes a single plot of how each workload's
energy is *distributed* between compute (MAC) and on-chip SRAM (global buffers).
Bar color encodes the component (MAC vs SRAM); the hatch pattern encodes the
systolic-array size.

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
import subprocess
import sys

import yaml

# rundir-accelergy is the directory this script lives in; run_all.sh assumes
# it is the working directory.
RUNDIR = os.path.dirname(os.path.abspath(__file__))


# How leaf component names map to the energy categories we plot.
#   MAC  : the PE multiply-accumulate units
#   SRAM : on-chip global buffers (ifmap / weights / psum GLBs)
#   Other: PE-local register-file scratchpads + off-chip DRAM
def categorize(leaf_name):
    if leaf_name == 'mac':
        return 'MAC'
    if leaf_name.endswith('_glb'):
        return 'SRAM'
    return 'Other'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workloads', nargs='+',
                   default=['vit_s', 'vit_b', 'vit_l'],
                   help='Workload names under the topology dir (without .csv). '
                        '(vit_h is omitted by default as it is large/slow.)')
    p.add_argument('--topo-dir', default='../topologies/ispass25_models',
                   help='Directory of topology CSVs (relative to rundir-accelergy).')
    p.add_argument('--input-type', default='gemm', choices=['gemm', 'conv'],
                   help='Topology type. The ViT models are GEMM (M,N,K).')
    p.add_argument('--dataflow', default='ws', choices=['ws', 'os', 'is'],
                   help='Single dataflow to use (default: ws).')
    p.add_argument('--array-sizes', type=int, nargs='+', default=[32, 64, 128],
                   help='Square systolic-array dimensions to sweep.')
    p.add_argument('--base-cfg', default='scale.cfg',
                   help='Base config template to derive sweep configs from.')
    p.add_argument('--out-dir', default='tutorial_results',
                   help='Directory (under rundir-accelergy) for all outputs.')
    p.add_argument('--skip-run', action='store_true',
                   help='Skip simulation; only re-parse + re-plot existing results.')
    p.add_argument('--force', action='store_true',
                   help='Re-run a point even if its results are already cached.')
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
    """Run the full run_all.sh flow for one (workload, size) point."""
    cmd = ['bash', './run_all.sh',
           '-c', cfg_path,
           '-t', model,
           '-p', scsim_log,
           '-o', out_dir,
           '-i', input_type]
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, cwd=RUNDIR, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def energy_by_category(energy_yaml):
    """Return {'MAC': pJ, 'SRAM': pJ, 'Other': pJ} for one estimation file."""
    with open(energy_yaml) as f:
        data = yaml.safe_load(f)
    agg = {'MAC': 0.0, 'SRAM': 0.0, 'Other': 0.0}
    for c in data['energy_estimation']['components']:
        leaf = c['name'].split('.')[-1].split('[')[0]
        agg[categorize(leaf)] += float(c['energy'])
    return agg


def main():
    args = parse_args()

    base_cfg = os.path.join(RUNDIR, args.base_cfg)
    out_dir = os.path.join(RUNDIR, args.out_dir)
    cfg_dir = os.path.join(out_dir, 'configs')
    log_dir = os.path.join(out_dir, 'logs')
    scsim_log = os.path.join(out_dir, 'scalesim_raw')
    for d in (out_dir, cfg_dir, log_dir, scsim_log):
        os.makedirs(d, exist_ok=True)

    df = args.dataflow
    results = []   # list of dicts: workload, size, MAC, SRAM, Other (pJ)
    for wl in args.workloads:
        model = os.path.join(args.topo_dir, f'{wl}.csv')
        if not os.path.exists(os.path.join(RUNDIR, model)):
            print(f'[SKIP] workload {wl}: {model} not found')
            continue
        for size in args.array_sizes:
            run_name = f'{wl}_{size}x{size}_{df}'
            acc_yaml = os.path.join(out_dir, f'accelergy_output_{run_name}',
                                    'energy_estimation.yaml')

            cached = os.path.exists(acc_yaml)
            if not args.skip_run and not (cached and not args.force):
                print(f'[run ] {run_name} ...', flush=True)
                cfg_path = os.path.join(cfg_dir, f'{run_name}.cfg')
                make_config(base_cfg, size, df, run_name, cfg_path)
                log_path = os.path.join(log_dir, f'{run_name}.log')
                rc = run_point(cfg_path, model, scsim_log, out_dir,
                               args.input_type, log_path)
                if rc != 0:
                    print(f'[WARN] {run_name} exited with code {rc} '
                          f'(accelergy energy step may have failed) - see {log_path}',
                          flush=True)
            elif cached:
                print(f'[cache] {run_name}: reusing existing results', flush=True)

            if not os.path.exists(acc_yaml):
                print(f'[SKIP] {run_name}: missing energy_estimation.yaml', flush=True)
                continue

            cats = energy_by_category(acc_yaml)
            results.append({'workload': wl, 'size': size, **cats})
            print(f'[ ok ] {run_name}: MAC={cats["MAC"]/1e6:.2f} uJ  '
                  f'SRAM={cats["SRAM"]/1e6:.2f} uJ  Other={cats["Other"]/1e6:.2f} uJ',
                  flush=True)

    if not results:
        print('No results collected. Aborting.', file=sys.stderr)
        sys.exit(1)

    write_csv(results, os.path.join(out_dir, f'workload_energy_{df}.csv'))
    plot_results(results, args.workloads, args.array_sizes, df,
                 os.path.join(out_dir, f'workload_energy_{df}.png'))


def write_csv(results, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['workload', 'array_size', 'MAC_energy_uJ', 'SRAM_energy_uJ',
                    'Other_energy_uJ', 'total_energy_uJ', 'MAC_frac', 'SRAM_frac'])
        for r in results:
            tot = r['MAC'] + r['SRAM'] + r['Other']
            w.writerow([r['workload'], f"{r['size']}x{r['size']}",
                        f"{r['MAC']/1e6:.4f}", f"{r['SRAM']/1e6:.4f}",
                        f"{r['Other']/1e6:.4f}", f"{tot/1e6:.4f}",
                        f"{r['MAC']/tot:.4f}" if tot else '0',
                        f"{r['SRAM']/tot:.4f}" if tot else '0'])
    print(f'\nWrote results CSV -> {path}')


def plot_results(results, workloads, sizes, df, path):
    """Single plot: grouped+stacked bars. Color = component (MAC / SRAM);
    hatch pattern = systolic-array size."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'legend.title_fontsize': 17,
    })

    idx = {(r['workload'], r['size']): r for r in results}
    wl_present = [w for w in workloads if any(w == r['workload'] for r in results)]
    x = np.arange(len(wl_present))

    C_MAC, C_SRAM = '#C44E52', '#4C72B0'
    hatch_pool = ['', '//', 'xx', '..', '\\\\', 'oo']
    size_hatch = {s: hatch_pool[i % len(hatch_pool)] for i, s in enumerate(sizes)}

    n = len(sizes)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(12, 3.4 * len(wl_present)), 6.5))

    for si, size in enumerate(sizes):
        offset = (si - (n - 1) / 2) * width
        mac = np.array([idx[(w, size)]['MAC'] / 1e6 if (w, size) in idx else 0
                        for w in wl_present])
        sram = np.array([idx[(w, size)]['SRAM'] / 1e6 if (w, size) in idx else 0
                         for w in wl_present])
        h = size_hatch[size]
        ax.bar(x + offset, mac, width, color=C_MAC, hatch=h,
               edgecolor='black', linewidth=0.5)
        ax.bar(x + offset, sram, width, bottom=mac, color=C_SRAM, hatch=h,
               edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(wl_present)
    ax.set_xlabel('Workload')
    ax.set_ylabel('Energy (uJ)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Two-part legend: color -> component, hatch -> array size.
    comp_handles = [Patch(facecolor=C_MAC, edgecolor='black', label='MAC'),
                    Patch(facecolor=C_SRAM, edgecolor='black', label='SRAM (GLB)')]
    size_handles = [Patch(facecolor='white', edgecolor='black',
                          hatch=size_hatch[s], label=f'{s}x{s}') for s in sizes]
    leg1 = ax.legend(handles=comp_handles, title='Component', loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, title='Array size', loc='upper right')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'Wrote plot        -> {path}')


if __name__ == '__main__':
    main()
