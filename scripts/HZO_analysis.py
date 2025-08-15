import os 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from functools import reduce
from matplotlib.patches import Patch
import numpy as np

# # =========================
# # Constants / Parameters
# # =========================

# memory capacities in kb
filter_double_buffer_capacity = 256 #kb
ifmap_double_buffer_capacity = 1536 #kb
ofmap_double_buffer_capacity = 1536 #kb
checkpointing_double_buffer_capacity = 32 #kb based on the largest output layer size

# clock frequency in Hz
clock_frequency = 20000000 #20MHz
period = 1 / clock_frequency

# # =========================
# # Load Data
# # =========================

# locate the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# path to scalesim output file
scalesim_output_file = os.path.join(current_dir, '../edgeTPU_experiment_results/edgeTPU_ws/DETAILED_ACCESS_REPORT.csv')
# path to HZO data file
hzo_data_file = os.path.join(current_dir, 'HZO_data/combined_64KB&512KB.csv')

# read scalesim output file and extract timing and read/write counts
scalesim_df = pd.read_csv(scalesim_output_file)
sram_filter_reads = scalesim_df[' SRAM Filter Reads'].sum()  
sram_ofmap_writes = scalesim_df[' SRAM OFMAP Writes'].sum()
           
weight_start_cycle = scalesim_df[' SRAM Filter Start Cycle']
weight_end_cycle = scalesim_df[' SRAM Filter Stop Cycle']

output_start_cycle = scalesim_df[' SRAM OFMAP Start Cycle']
output_end_cycle = scalesim_df[' SRAM OFMAP Stop Cycle']                          

# read HZO data file and invert the df
hzo_df = pd.read_csv(hzo_data_file)
hzo_df['OPT target'] = hzo_df['Source'].str.split('_').str[2]
hzo_3_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_15_3_9.cell']
hzo_5_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_15_5_9.cell']
hzo_3_weight_df = hzo_3_df[hzo_3_df['Capacity (KB)'] == 512]
hzo_5_weight_df = hzo_5_df[hzo_5_df['Capacity (KB)'] ==  512]
hzo_3_checkpoint_df = hzo_3_df[hzo_3_df['Capacity (KB)'] == 64]
hzo_5_checkpoint_df = hzo_5_df[hzo_5_df['Capacity (KB)'] == 64]

# # =========================
# # Area Calculation
# # =========================

# plot area for both weight and checkpointing buffers under the two different HZO eNVM types with different opt targets
# for each opt target, we want to have four multi-bars:
# 1. HZO 3 eNVM weight and checkpointing buffer
# 2. HZO 5 eNVM weight and checkpointing buffer
# 3. HZO 3 eNVM weight and HZO 5 eNVM checkpointing buffer
# 4. HZO 5 eNVM weight and HZO 3 eNVM checkpointing buffer
# use stacked bar and legend to differentiate weight and checkpointing buffers
def area_prep(df, colname):
    out = df.loc[:, ["OPT target", "Area (mm^2)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Area (mm^2)"] = pd.to_numeric(out["Area (mm^2)"], errors="coerce")
    return out.rename(columns={"Area (mm^2)": colname})

df_3_w_area = area_prep(hzo_3_weight_df, "HZO 3 Weight Buffer Area (mm^2)")
df_5_w_area = area_prep(hzo_5_weight_df, "HZO 5 Weight Buffer Area (mm^2)")
df_3_c_area = area_prep(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer Area (mm^2)")
df_5_c_area = area_prep(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer Area (mm^2)")

# Outer-merge all on OPT target so missing entries are allowed
area_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_area, df_5_w_area, df_3_c_area, df_5_c_area]
).sort_values("OPT target").reset_index(drop=True)

# Optional: replace NaNs with 0 for plotting
area_df = area_df.fillna(0.0)
print(area_df)

# 1) Build long-form scenario table: 4 rows per OPT target
records = []
for _, r in area_df.fillna(0.0).iterrows():
    opt = str(r['OPT target']).strip()

    records.append({
        'OPT target': opt, 'Scenario': 'HZO3 / HZO3',
        'Weight Area (mm^2)': r['HZO 3 Weight Buffer Area (mm^2)'],
        'Checkpoint Area (mm^2)': r['HZO 3 Checkpointing Buffer Area (mm^2)'],
    })
    records.append({
        'OPT target': opt, 'Scenario': 'HZO5 / HZO5',
        'Weight Area (mm^2)': r['HZO 5 Weight Buffer Area (mm^2)'],
        'Checkpoint Area (mm^2)': r['HZO 5 Checkpointing Buffer Area (mm^2)'],
    })
    records.append({
        'OPT target': opt, 'Scenario': 'HZO3 / HZO5',
        'Weight Area (mm^2)': r['HZO 3 Weight Buffer Area (mm^2)'],
        'Checkpoint Area (mm^2)': r['HZO 5 Checkpointing Buffer Area (mm^2)'],
    })
    records.append({
        'OPT target': opt, 'Scenario': 'HZO5 / HZO3',
        'Weight Area (mm^2)': r['HZO 5 Weight Buffer Area (mm^2)'],
        'Checkpoint Area (mm^2)': r['HZO 3 Checkpointing Buffer Area (mm^2)'],
    })

scenario_df = pd.DataFrame.from_records(records)

# 2) Plot grouped, stacked bars with color-by-tech and hatch-by-buffer
scenarios = ['HZO3 / HZO3', 'HZO5 / HZO5', 'HZO3 / HZO5', 'HZO5 / HZO3']
opt_targets = sorted(scenario_df['OPT target'].astype(str).unique().tolist())

x_idx = np.arange(len(opt_targets))
group_width = 0.8
bar_width = group_width / len(scenarios)

# Color for technology
color_map = {'HZO3': '#4C72B0', 'HZO5': '#DD8452'}  # tweak if you prefer
# Hatch for buffer type
hatch_map = {'weight': '', 'ckpt': '///'}           # '' = solid; '///' = hatched

def variant_for(scenario: str, which: str) -> str:
    """Return 'HZO3' or 'HZO5' for the weight or ckpt half of a scenario."""
    w_var, c_var = scenario.split(' / ')
    return w_var if which == 'weight' else c_var

st.subheader("Weight + Checkpointing Buffer Area per OPT target")
fig, ax = plt.subplots(figsize=(11, 5))

for i, scen in enumerate(scenarios):
    sub = (scenario_df[scenario_df['Scenario'] == scen]
           .set_index('OPT target')
           .reindex(opt_targets)
           .fillna(0.0))

    weight_vals = sub['Weight Area (mm^2)'].to_numpy()
    ckpt_vals   = sub['Checkpoint Area (mm^2)'].to_numpy()
    pos = x_idx - group_width/2 + i*bar_width + bar_width/2

    # Choose tech color for each half based on scenario
    w_color = color_map['HZO3' if variant_for(scen, 'weight') == 'HZO3' else 'HZO5']
    c_color = color_map['HZO3' if variant_for(scen, 'ckpt')  == 'HZO3' else 'HZO5']

    ax.bar(
        pos, weight_vals, width=bar_width,
        color=w_color, hatch=hatch_map['weight'],
        edgecolor='black', linewidth=0.6
    )
    ax.bar(
        pos, ckpt_vals, width=bar_width, bottom=weight_vals,
        color=c_color, hatch=hatch_map['ckpt'],
        edgecolor='black', linewidth=0.6
    )

ax.set_xticks(x_idx)
ax.set_xticklabels(opt_targets, rotation=20)
ax.set_xlabel('OPT target')
ax.set_ylabel('Area (mm^2)')
ax.set_title('Weight + Checkpointing Buffer Area per OPT target (HZO technology × buffer type)')

handles = [
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['weight'], label='HZO3 Weight Buffer'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['weight'], label='HZO5 Weight Buffer'),
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO3 Checkpointing Buffer'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO5 Checkpointing Buffer'),
]

leg1 = ax.legend(handles=handles, title='Legend', loc='upper left', frameon=True)
ax.add_artist(leg1)  # keep both legends

ax.margins(x=0.02)
plt.tight_layout()

st.pyplot(fig)

# # =========================
# # Dynamic read energy per access
# # =========================
def dynamic_read_prep(df, colname):
    out = df.loc[:, ["OPT target", "Read Energy (pJ)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Read Energy (pJ)"] = pd.to_numeric(out["Read Energy (pJ)"], errors="coerce")
    return out.rename(columns={"Read Energy (pJ)": colname})

df_3_w_dyRead = dynamic_read_prep(hzo_3_weight_df, "HZO 3 Weight Buffer Read Energy (pJ)")
df_5_w_dyRead = dynamic_read_prep(hzo_5_weight_df, "HZO 5 Weight Buffer Read Energy (pJ)")

# Outer-merge all on OPT target so missing entries are allowed
dyRead_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_dyRead, df_5_w_dyRead]
).sort_values("OPT target").reset_index(drop=True)

print(dyRead_df)

# plot dynamic read energy for both HZO 3 and HZO 5 eNVM weight buffers
read_fig, ax = plt.subplots(figsize=(10, 5))
dyRead_df.plot(kind='bar', x='OPT target',
                y=['HZO 3 Weight Buffer Read Energy (pJ)', 'HZO 5 Weight Buffer Read Energy (pJ)'],
                stacked=False, ax=ax)
ax.set_ylabel('Dynamic Read Energy (pJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(dyRead_df['OPT target'])), labels=dyRead_df['OPT target'], rotation=20)
ax.set_title('Dynamic Read Energy per access for HZO eNVM Weight Buffers')
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Weight Buffer", "HZO 5 Weight Buffer"], loc="upper left")
plt.tight_layout()

# # =========================
# # Dynamic write energy per access
# # =========================
def dynamic_write_prep(df, colname):
    out = df.loc[:, ["OPT target", "Write Energy (pJ)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Write Energy (pJ)"] = pd.to_numeric(out["Write Energy (pJ)"], errors="coerce")
    return out.rename(columns={"Write Energy (pJ)": colname})

df_3_c_dyWrite = dynamic_write_prep(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer Write Energy (pJ)")
df_5_c_dyWrite = dynamic_write_prep(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer Write Energy (pJ)")

# Outer-merge all on OPT target so missing entries are allowed
dyWrite_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_c_dyWrite, df_5_c_dyWrite]
).sort_values("OPT target").reset_index(drop=True)
print(dyWrite_df)
# plot dynamic write energy for both HZO 3 and HZO 5 eNVM checkpointing buffers
write_fig, ax = plt.subplots(figsize=(10, 5))
dyWrite_df.plot(kind='bar', x='OPT target',
                y=['HZO 3 Checkpointing Buffer Write Energy (pJ)', 'HZO 5 Checkpointing Buffer Write Energy (pJ)'],
                stacked=False, ax=ax)
ax.set_ylabel('Dynamic Write Energy (pJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(dyWrite_df['OPT target'])), labels=dyWrite_df['OPT target'], rotation=20)
ax.set_title('Dynamic Write Energy per access for HZO eNVM Checkpointing Buffers')
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper right")
plt.tight_layout()

# plot the dynamic energy plots side by side
st.subheader("Dynamic Read and Write Energy per access for HZO eNVM Buffers")
col1, col2 = st.columns(2)
col1.pyplot(read_fig)
col2.pyplot(write_fig)

# # =========================
# dynamic energy per inference
# # =========================
dyRead_df['HZO 3 Weight Buffer Read Energy per Inference(mJ)'] = dyRead_df['HZO 3 Weight Buffer Read Energy (pJ)'] * sram_filter_reads / 1000000000
dyRead_df['HZO 5 Weight Buffer Read Energy per Inference(mJ)'] = dyRead_df['HZO 5 Weight Buffer Read Energy (pJ)'] * sram_filter_reads / 1000000000
dyWrite_df['HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)'] = dyWrite_df['HZO 3 Checkpointing Buffer Write Energy (pJ)'] * sram_ofmap_writes / 1000000000
dyWrite_df['HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)'] = dyWrite_df['HZO 5 Checkpointing Buffer Write Energy (pJ)'] * sram_ofmap_writes / 1000000000

# # plot the dynamic energy per inference for both HZO 3 and HZO 5 eNVM weight buffers
# do the same as the area plot where you have 4 multi-bars for each opt target
# bar 1: HZO 3 weight buffer read energy per inference and HZO 3 checkpointing buffer write energy per inference
# bar 2: HZO 5 weight buffer read energy per inference and HZO 5 checkpointing buffer write energy per inference
# bar 3: HZO 3 weight buffer read energy per inference and HZO 5 checkpointing buffer write energy per inference
# bar 4: HZO 5 weight buffer read energy per inference and HZO 3 checkpointing buffer write energy per inference

# 1) Normalize keys and keep only needed columns
dyRead_df = dyRead_df.copy()
dyWrite_df = dyWrite_df.copy()
dyRead_df['OPT target']  = dyRead_df['OPT target'].astype(str).str.strip()
dyWrite_df['OPT target'] = dyWrite_df['OPT target'].astype(str).str.strip()

read_cols  = ['OPT target',
              'HZO 3 Weight Buffer Read Energy per Inference(mJ)',
              'HZO 5 Weight Buffer Read Energy per Inference(mJ)']
write_cols = ['OPT target',
              'HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)',
              'HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)']

read_base  = dyRead_df.loc[:, read_cols].dropna(subset=['OPT target'])
write_base = dyWrite_df.loc[:, write_cols].dropna(subset=['OPT target'])

# 2) Merge read+write on OPT target
energy_base = pd.merge(read_base, write_base, on='OPT target', how='inner').fillna(0.0)

# 3) Build 4-scenario long-form table (4 rows per OPT)
records = []
for _, r in energy_base.iterrows():
    opt = str(r['OPT target']).strip()
    records.extend([
        {'OPT target': opt, 'Scenario': 'HZO3 / HZO3',
         'Weight (mJ)': r['HZO 3 Weight Buffer Read Energy per Inference(mJ)'],
         'Checkpoint (mJ)': r['HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)']},
        {'OPT target': opt, 'Scenario': 'HZO5 / HZO5',
         'Weight (mJ)': r['HZO 5 Weight Buffer Read Energy per Inference(mJ)'],
         'Checkpoint (mJ)': r['HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)']},
        {'OPT target': opt, 'Scenario': 'HZO3 / HZO5',
         'Weight (mJ)': r['HZO 3 Weight Buffer Read Energy per Inference(mJ)'],
         'Checkpoint (mJ)': r['HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)']},
        {'OPT target': opt, 'Scenario': 'HZO5 / HZO3',
         'Weight (mJ)': r['HZO 5 Weight Buffer Read Energy per Inference(mJ)'],
         'Checkpoint (mJ)': r['HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)']},
    ])
dyEnergy_df = pd.DataFrame.from_records(records)

# 4) Plot grouped+stacked bars with color-by-tech and hatch-by-buffer
scenarios   = ['HZO3 / HZO3', 'HZO5 / HZO5', 'HZO3 / HZO5', 'HZO5 / HZO3']
opt_targets = sorted(dyEnergy_df['OPT target'].astype(str).unique().tolist())

x_idx       = np.arange(len(opt_targets))
group_width = 0.8
bar_width   = group_width / len(scenarios)

# Colors: tech; HZO3 vs HZO5. Hatches: buffer type; weight vs checkpoint
color_map  = {'HZO3': '#4C72B0', 'HZO5': '#DD8452'}
hatch_map  = {'weight': '', 'ckpt': '///'}

def which_tech(scenario: str, part: str) -> str:
    w_tech, c_tech = scenario.split(' / ')
    return w_tech if part == 'weight' else c_tech

# ========== Streamlit side-by-side with your AREA plot ==========

st.subheader("Dynamic Energy per Inference (stacked) by OPT target")
fig_e, ax_e = plt.subplots(figsize=(11, 5))

for i, scen in enumerate(scenarios):
    sub = (dyEnergy_df[dyEnergy_df['Scenario'] == scen]
            .set_index('OPT target')
            .reindex(opt_targets).fillna(0.0))
    w_vals = sub['Weight (mJ)'].to_numpy()
    c_vals = sub['Checkpoint (mJ)'].to_numpy()
    pos = x_idx - group_width/2 + i*bar_width + bar_width/2

    w_color = color_map['HZO3' if which_tech(scen, 'weight') == 'HZO3' else 'HZO5']
    c_color = color_map['HZO3' if which_tech(scen, 'ckpt')  == 'HZO3' else 'HZO5']

    ax_e.bar(pos, w_vals, width=bar_width,
                color=w_color, hatch=hatch_map['weight'],
                edgecolor='black', linewidth=0.6)
    ax_e.bar(pos, c_vals, width=bar_width, bottom=w_vals,
                color=c_color, hatch=hatch_map['ckpt'],
                edgecolor='black', linewidth=0.6)

ax_e.set_xticks(x_idx)
ax_e.set_xticklabels(opt_targets, rotation=20)
ax_e.set_xlabel('OPT target')
ax_e.set_ylabel('Dynamic Energy per Inference (mJ)')
ax_e.set_title('Weight (read) + Checkpoint (write) energy — HZO tech × buffer type')

handles = [
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['weight'], label='HZO3 Weight Read'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['weight'], label='HZO5 Weight Read'),
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO3 Checkpoint Write'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO5 Checkpoint Write'),
]
leg1 = ax_e.legend(handles=handles, title='Technology', loc='upper right', frameon=True)
ax_e.add_artist(leg1)

ax_e.margins(x=0.02)
plt.tight_layout()
st.pyplot(fig_e)


# # =========================
# static energy per inference
# # =========================
# extract the leakage power from the HZO data file
def static_energy_prep(df, colname):
    out = df.loc[:, ["OPT target", "Leakage Power (mW)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Leakage Power (mW)"] = pd.to_numeric(out["Leakage Power (mW)"], errors="coerce")
    return out.rename(columns={"Leakage Power (mW)": colname})

df_3_w_static = static_energy_prep(hzo_3_weight_df, "HZO 3 Weight Buffer Static Energy (mJ)")
df_5_w_static = static_energy_prep(hzo_5_weight_df, "HZO 5 Weight Buffer Static Energy (mJ)")
df_3_c_static = static_energy_prep(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer Static Energy (mJ)")
df_5_c_static = static_energy_prep(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer Static Energy (mJ)")

leakage_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_static, df_5_w_static, df_3_c_static, df_5_c_static]
).sort_values("OPT target").reset_index(drop=True)
print(leakage_df)

# method 1
weight_buffer_cycle_counts = weight_end_cycle - weight_start_cycle
weight_buffer_cycle_timing = weight_buffer_cycle_counts * period  # in seconds
checkpoint_buffer_cycle_counts = output_end_cycle - output_start_cycle
checkpoint_buffer_cycle_timing = checkpoint_buffer_cycle_counts * period  # in seconds

# method 2
# extract read and write latencies from the HZO data file
def static_read_latency_prep(df, colname):
    out = df.loc[:, ["OPT target", "Read Latency (ns)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Read Latency (ns)"] = pd.to_numeric(out["Read Latency (ns)"], errors="coerce")
    return out.rename(columns={"Read Latency (ns)": colname})
df_3_w_static_latency = static_read_latency_prep(hzo_3_weight_df, "HZO 3 Weight Buffer read latency (ns)")
df_5_w_static_latency = static_read_latency_prep(hzo_5_weight_df, "HZO 5 Weight Buffer read latency (ns)")

def static_write_latency_prep(df, colname):
    out = df.loc[:, ["OPT target", "Write Latency (ns)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Write Latency (ns)"] = pd.to_numeric(out["Write Latency (ns)"], errors="coerce")
    return out.rename(columns={"Write Latency (ns)": colname})
df_3_c_static_latency = static_write_latency_prep(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer write latency (ns)")
df_5_c_static_latency = static_write_latency_prep(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer write latency (ns)")

#  TODO

# calculate the static energy for the weight and checkpointing buffers per inference
leakage_df['HZO 3 Weight Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 3 Weight Buffer Static Energy (mJ)'] * weight_buffer_cycle_timing
leakage_df['HZO 5 Weight Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 5 Weight Buffer Static Energy (mJ)'] * weight_buffer_cycle_timing
leakage_df['HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 3 Checkpointing Buffer Static Energy (mJ)'] * checkpoint_buffer_cycle_timing
leakage_df['HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 5 Checkpointing Buffer Static Energy (mJ)'] * checkpoint_buffer_cycle_timing

# # plot the static energy for both HZO 3 and HZO 5 eNVM weight buffers
# do the same as the area plot where you have 4 multi-bars for each opt target
# bar 1: HZO 3 weight buffer static energy per inference and HZO 3 checkpointing buffer static energy per inference
# bar 2: HZO 5 weight buffer static energy per inference and HZO 5 checkpointing buffer static energy per inference
# bar 3: HZO 3 weight buffer static energy per inference and HZO 5 checkpointing buffer static energy per inference
# bar 4: HZO 5 weight buffer static energy per inference and HZO 3 checkpointing buffer static energy per inference

# Build long-form scenario table (4 rows per OPT target)
records = []
for _, r in leakage_df.fillna(0.0).iterrows():
    opt = str(r["OPT target"]).strip()
    records.extend([
        {"OPT target": opt, "Scenario": "HZO3 / HZO3",
         "Weight (mJ)": r["HZO 3 Weight Buffer Static Energy (mJ) per Inference"],
         "Checkpoint (mJ)": r["HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference"]},
        {"OPT target": opt, "Scenario": "HZO5 / HZO5",
         "Weight (mJ)": r["HZO 5 Weight Buffer Static Energy (mJ) per Inference"],
         "Checkpoint (mJ)": r["HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference"]},
        {"OPT target": opt, "Scenario": "HZO3 / HZO5",
         "Weight (mJ)": r["HZO 3 Weight Buffer Static Energy (mJ) per Inference"],
         "Checkpoint (mJ)": r["HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference"]},
        {"OPT target": opt, "Scenario": "HZO5 / HZO3",
         "Weight (mJ)": r["HZO 5 Weight Buffer Static Energy (mJ) per Inference"],
         "Checkpoint (mJ)": r["HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference"]},
    ])

staticEnergy_df = pd.DataFrame.from_records(records)

# Plot grouped + stacked (color = tech, hatch = buffer type)
scenarios   = ["HZO3 / HZO3", "HZO5 / HZO5", "HZO3 / HZO5", "HZO5 / HZO3"]
opt_targets = sorted(staticEnergy_df["OPT target"].astype(str).unique().tolist())

x_idx       = np.arange(len(opt_targets))
group_width = 0.8
bar_width   = group_width / len(scenarios)

color_map = {"HZO3": "#4C72B0", "HZO5": "#DD8452"}   # tech color
hatch_map = {"weight": "", "ckpt": "///"}            # buffer hatch

def which_tech(scenario: str, part: str) -> str:
    w_tech, c_tech = scenario.split(" / ")
    return w_tech if part == "weight" else c_tech

st.subheader("Static Energy per Inference (stacked) by OPT target")
st.write("TODO: do the static execution time check")
fig, ax = plt.subplots(figsize=(11, 5))

for i, scen in enumerate(scenarios):
    sub = (staticEnergy_df[staticEnergy_df["Scenario"] == scen]
           .set_index("OPT target")
           .reindex(opt_targets)
           .fillna(0.0))
    w_vals = sub["Weight (mJ)"].to_numpy()
    c_vals = sub["Checkpoint (mJ)"].to_numpy()

    pos = x_idx - group_width/2 + i*bar_width + bar_width/2

    w_color = color_map["HZO3" if which_tech(scen, "weight") == "HZO3" else "HZO5"]
    c_color = color_map["HZO3" if which_tech(scen, "ckpt")  == "HZO3" else "HZO5"]

    ax.bar(pos, w_vals, width=bar_width,
           color=w_color, hatch=hatch_map["weight"],
           edgecolor="black", linewidth=0.6)
    ax.bar(pos, c_vals, width=bar_width, bottom=w_vals,
           color=c_color, hatch=hatch_map["ckpt"],
           edgecolor="black", linewidth=0.6)

ax.set_xticks(x_idx)
ax.set_xticklabels(opt_targets, rotation=20)
ax.set_xlabel("OPT target")
ax.set_ylabel("Static Energy per Inference (mJ)")
ax.set_title("Static (Leakage) Energy per Inference — HZO tech × buffer type")

handles = [
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['weight'], label='HZO3 Weight Read'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['weight'], label='HZO5 Weight Read'),
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO3 Checkpoint Write'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO5 Checkpoint Write'),
]
leg1 = ax.legend(handles=handles, title="Technology", loc="upper left", frameon=True)
ax.add_artist(leg1)

ax.margins(x=0.02)
plt.tight_layout()

st.pyplot(fig)

# =========================
# Total Energy per Inference (Dynamic + Static)
# =========================

# 1) Merge dynamic & static scenario tables and compute totals
dyn_scen = dyEnergy_df.rename(columns={
    "Weight (mJ)": "Weight_dyn_mJ",
    "Checkpoint (mJ)": "Checkpoint_dyn_mJ"
})
stat_scen = staticEnergy_df.rename(columns={
    "Weight (mJ)": "Weight_stat_mJ",
    "Checkpoint (mJ)": "Checkpoint_stat_mJ"
})

total_base = pd.merge(dyn_scen, stat_scen, on=["OPT target", "Scenario"], how="outer").fillna(0.0)

total_base["Weight_total_mJ"]     = total_base["Weight_dyn_mJ"]     + total_base["Weight_stat_mJ"]
total_base["Checkpoint_total_mJ"] = total_base["Checkpoint_dyn_mJ"] + total_base["Checkpoint_stat_mJ"]
total_base["Total_mJ"]            = total_base["Weight_total_mJ"]   + total_base["Checkpoint_total_mJ"]

# 2) Plot grouped + stacked bars (same 4 scenarios)
scenarios   = ["HZO3 / HZO3", "HZO5 / HZO5", "HZO3 / HZO5", "HZO5 / HZO3"]
opt_targets = sorted(total_base["OPT target"].astype(str).unique().tolist())

x_idx       = np.arange(len(opt_targets))
group_width = 0.8
bar_width   = group_width / len(scenarios)

# Reuse your style: color = tech, hatch = buffer type
color_map = {"HZO3": "#4C72B0", "HZO5": "#DD8452"}
hatch_map = {"weight": "", "ckpt": "///"}

def which_tech(scenario: str, part: str) -> str:
    w_tech, c_tech = scenario.split(" / ")
    return w_tech if part == "weight" else c_tech

st.subheader("Total Energy per Inference (stacked) by OPT target")
fig_t, ax_t = plt.subplots(figsize=(11, 5))

for i, scen in enumerate(scenarios):
    sub = (total_base[total_base["Scenario"] == scen]
           .set_index("OPT target")
           .reindex(opt_targets)
           .fillna(0.0))
    w_vals = sub["Weight_total_mJ"].to_numpy()
    c_vals = sub["Checkpoint_total_mJ"].to_numpy()

    pos = x_idx - group_width/2 + i*bar_width + bar_width/2

    w_color = color_map["HZO3" if which_tech(scen, "weight") == "HZO3" else "HZO5"]
    c_color = color_map["HZO3" if which_tech(scen, "ckpt")  == "HZO3" else "HZO5"]

    ax_t.bar(pos, w_vals, width=bar_width,
             color=w_color, hatch=hatch_map["weight"],
             edgecolor="black", linewidth=0.6)
    ax_t.bar(pos, c_vals, width=bar_width, bottom=w_vals,
             color=c_color, hatch=hatch_map["ckpt"],
             edgecolor="black", linewidth=0.6)

ax_t.set_xticks(x_idx)
ax_t.set_xticklabels(opt_targets, rotation=20)
ax_t.set_xlabel("OPT target")
ax_t.set_ylabel("Total Energy per Inference (mJ)")
ax_t.set_title("Total (Dynamic + Static) Energy — HZO tech × buffer type")

handles = [
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['weight'], label='HZO3 Weight Read'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['weight'], label='HZO5 Weight Read'),
    Patch(facecolor=color_map['HZO3'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO3 Checkpoint Write'),
    Patch(facecolor=color_map['HZO5'], edgecolor='black', hatch=hatch_map['ckpt'], label='HZO5 Checkpoint Write'),
]
leg1 = ax_t.legend(handles=handles, title="Technology", loc="upper center", frameon=True)

ax_t.margins(x=0.02)
plt.tight_layout()
st.pyplot(fig_t)



# calculate read energy
# weight_buffer_access_count_timing = sram_filter_reads * 
# hzo_df['weight buffer static energy(mJ)'] = hzo_df['Leakage Power (mW)'] * weight_buffer_cycle_timing
# weight_cycle_counts = weight_end_cycle - weight_start_cycle
# hzo_df['weight buffer read energy(mJ)'] = hzo_df['Dynamic Read Energy per access(pJ)'] * sram_filter_reads / 1000000 + hzo_df['weight buffer static energy(mJ)']
# # calculate write energy
# hzo_df['checkpointing buffer static energy(mJ)'] = hzo_df['Leakage Power (mW)'] * checkpoint_buffer_cycle_timing
# checkpointing_buffer_static_energy = hzo_df['Leakage Power (mW)'] * checkpoint_buffer_cycle_timing
# hzo_df['checkpointing buffer write energy(mJ)'] = hzo_df['Dynamic Write Energy per access(pJ)'] * sram_ofmap_writes / 1000000 + hzo_df['checkpointing buffer static energy(mJ)']

# # result dataframe
# # should only contain the read and write energy for the weight buffer and checkpointing buffer
# # and the area information
# result_df = pd.DataFrame({
#     'OPT_target': hzo_df['OPT_target'],
#     'Weight Buffer Area (mm^2)': hzo_df['weight buffer area(mm^2)'],
#     'Checkpointing Buffer Area (mm^2)': hzo_df['checkpointing buffer area(mm^2)'],
#     'Weight Buffer Read Energy (mJ)': hzo_df['weight buffer read energy(mJ)'],
#     'Checkpointing Buffer Write Energy (mJ)': hzo_df['checkpointing buffer write energy(mJ)']
# })

# # plot the area and stacked energy info in two seperate subplots
# # one for area
# # another for energy
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
# # Area plot stacked
# result_df.plot(kind='bar', x='OPT_target',
#                 y=['Weight Buffer Area (mm^2)', 'Checkpointing Buffer Area (mm^2)'],
#                 stacked=True, ax=ax1)
# ax1.set_ylabel('Area (mm^2)')
# ax1.set_xlabel('OPT Type')
# ax1.set_xticks(range(len(result_df['OPT_target'])), labels=result_df['OPT_target'], rotation=20)
# # Energy plot
# result_df.plot(kind='bar', x='OPT_target', 
#                 y=['Weight Buffer Read Energy (mJ)', 'Checkpointing Buffer Write Energy (mJ)'],
#                 stacked=True, ax=ax2)
# ax2.set_ylabel('Energy (mJ)')
# ax2.set_xlabel('OPT Type')
# # Rotate x-axis labels for better readability
# plt.xticks(rotation = 20)
# plt.tight_layout()
# plt.savefig(os.path.join(current_dir, 'HZO_analysis_results/HZO_analysis_results.png'))

