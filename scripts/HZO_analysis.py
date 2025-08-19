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
model_file = os.path.join(current_dir, '../topologies/MLperf_tiny/vww.csv')

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

weight_buffer_area_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_area, df_5_w_area]
).sort_values("OPT target").reset_index(drop=True)
checkpointing_buffer_area_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_c_area, df_5_c_area]
).sort_values("OPT target").reset_index(drop=True)

# plot the weight buffer area for both HZO 3 and HZO 5 eNVM weight buffers
st.subheader("Area per OPT target")
weight_buffer_area_fig, ax = plt.subplots(figsize=(10, 5))
weight_buffer_area_df.plot(kind='bar', x='OPT target',
                            y=['HZO 3 Weight Buffer Area (mm^2)', 'HZO 5 Weight Buffer Area (mm^2)'],
                            stacked=False, ax=ax)
ax.set_ylabel('Weight Buffer Area (mm^2)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(weight_buffer_area_df['OPT target'])), labels=weight_buffer_area_df['OPT target'], rotation=20)

handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Weight Buffer", "HZO 5 Weight Buffer"], loc="upper left")
plt.tight_layout()

# plot the checkpointing buffer area for both HZO 3 and HZO 5 eNVM checkpointing buffers
checkpointing_buffer_area_fig, ax = plt.subplots(figsize=(10, 5))
checkpointing_buffer_area_df.plot(kind='bar', x='OPT target',
                                    y=['HZO 3 Checkpointing Buffer Area (mm^2)', 'HZO 5 Checkpointing Buffer Area (mm^2)'],
                                    stacked=False, ax=ax)
# hatch the bars for checkpointing buffers
hatches = ['///', '\\\\\\']  # HZO3, HZO5
for container, hatch in zip(ax.containers, hatches):
    for bar in container:
        bar.set_hatch(hatch)
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
ax.set_ylabel('Checkpointing Buffer Area (mm^2)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(checkpointing_buffer_area_df['OPT target'])), labels=checkpointing_buffer_area_df['OPT target'], rotation=20)
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper left")
plt.tight_layout()

col1, col2 = st.columns(2)
col1.pyplot(weight_buffer_area_fig)
col2.pyplot(checkpointing_buffer_area_fig)

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

# plot dynamic write energy for both HZO 3 and HZO 5 eNVM checkpointing buffers
write_fig, ax = plt.subplots(figsize=(10, 5))
dyWrite_df.plot(kind='bar', x='OPT target',
                y=['HZO 3 Checkpointing Buffer Write Energy (pJ)', 'HZO 5 Checkpointing Buffer Write Energy (pJ)'],
                stacked=False, ax=ax)
# add hatches to the bars for checkpointing buffers
hatches = ['///', '\\\\\\']  # HZO3, HZO5
for container, hatch in zip(ax.containers, hatches):
    for bar in container:
        bar.set_hatch(hatch)
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
ax.set_ylabel('Dynamic Write Energy (pJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(dyWrite_df['OPT target'])), labels=dyWrite_df['OPT target'], rotation=20)
ax.set_title('Dynamic Write Energy per access for HZO eNVM Checkpointing Buffers')
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper right")
plt.tight_layout()

# plot the dynamic energy plots side by side
st.subheader("Dynamic Read and Write Energy per access (pJ) by OPT target")
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

# plot the dynamic energy for both HZO 3 and HZO 5 eNVM weight buffers
dyEnergy_weight_fig, ax = plt.subplots(figsize=(10, 5))
energy_base.plot(kind='bar', x='OPT target',
                    y=['HZO 3 Weight Buffer Read Energy per Inference(mJ)',
                        'HZO 5 Weight Buffer Read Energy per Inference(mJ)'],
                    stacked=False, ax=ax)
ax.set_ylabel('Dynamic Read Energy per Inference (mJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(energy_base['OPT target'])), labels=energy_base['OPT target'], rotation=20)
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Weight Buffer", "HZO 5 Weight Buffer"], loc="upper left")
plt.tight_layout()

# plot the dynamic energy for both HZO 3 and HZO 5 eNVM checkpointing buffers
dyEnergy_ckpt_fig, ax = plt.subplots(figsize=(10, 5))
energy_base.plot(kind='bar', x='OPT target',
                    y=['HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)',
                        'HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)'],
                    stacked=False, ax=ax)
hatches = ['///', '\\\\\\']  # HZO3, HZO5
for container, hatch in zip(ax.containers, hatches):
    for bar in container:
        bar.set_hatch(hatch)
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
ax.set_ylabel('Dynamic Write Energy per Inference (mJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(energy_base['OPT target'])), labels=energy_base['OPT target'], rotation=20)
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper right")
plt.tight_layout()

st.subheader("Dynamic Energy per Inference (mJ) by OPT target")
col1, col2 = st.columns(2)
col1.pyplot(dyEnergy_weight_fig)
col2.pyplot(dyEnergy_ckpt_fig)

# # =========================
# static energy per inference
# # =========================
# extract the leakage power from the HZO data file
def static_energy_prep(df, colname):
    out = df.loc[:, ["OPT target", "Leakage Power (mW)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Leakage Power (mW)"] = pd.to_numeric(out["Leakage Power (mW)"], errors="coerce")
    return out.rename(columns={"Leakage Power (mW)": colname})

df_3_w_static = static_energy_prep(hzo_3_weight_df, "HZO 3 Weight Buffer Static Power (mW)")
df_5_w_static = static_energy_prep(hzo_5_weight_df, "HZO 5 Weight Buffer Static Power (mW)")
df_3_c_static = static_energy_prep(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer Static Power (mW)")
df_5_c_static = static_energy_prep(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer Static Power (mW)")

leakage_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_static, df_5_w_static, df_3_c_static, df_5_c_static]
).sort_values("OPT target").reset_index(drop=True)
print(leakage_df)

# method 1
weight_buffer_cycle_counts = weight_end_cycle - weight_start_cycle
weight_buffer_cycle_counts_total = weight_buffer_cycle_counts.sum()
weight_buffer_cycle_timing = weight_buffer_cycle_counts_total * period  # in seconds

checkpoint_buffer_cycle_counts = output_end_cycle - output_start_cycle
checkpoint_buffer_cycle_counts_total = checkpoint_buffer_cycle_counts.sum()
checkpoint_buffer_cycle_timing = checkpoint_buffer_cycle_counts_total * period  # in seconds

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

# calculatethe latency per access times the number of accesses
latency_df = reduce(
    lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
    [df_3_w_static_latency, df_5_w_static_latency, df_3_c_static_latency, df_5_c_static_latency]
).sort_values("OPT target").reset_index(drop=True)
latency_df['HZO 3 Weight Buffer Total Read Latency (s)'] = latency_df['HZO 3 Weight Buffer read latency (ns)'] * sram_filter_reads / 1e9    
latency_df['HZO 5 Weight Buffer Total Read Latency (s)'] = latency_df['HZO 5 Weight Buffer read latency (ns)'] * sram_filter_reads / 1e9
latency_df['HZO 3 Checkpointing Buffer Total Write Latency (s)'] = latency_df['HZO 3 Checkpointing Buffer write latency (ns)'] * sram_ofmap_writes / 1e9
latency_df['HZO 5 Checkpointing Buffer Total Write Latency (s)'] = latency_df['HZO 5 Checkpointing Buffer write latency (ns)'] * sram_ofmap_writes / 1e9

# compare the two methods and use the max of the two
# make entries in hzo3 and hzo5 the compare with weight buffer cycle timing and take the max
for col in latency_df['HZO 3 Weight Buffer Total Read Latency (s)']:
    col = max(col, weight_buffer_cycle_timing)
for col in latency_df['HZO 5 Weight Buffer Total Read Latency (s)']:
    col = max(col, weight_buffer_cycle_timing)
for col in latency_df['HZO 3 Checkpointing Buffer Total Write Latency (s)']:
    col = max(col, checkpoint_buffer_cycle_timing)
for col in latency_df['HZO 5 Checkpointing Buffer Total Write Latency (s)']:
    col = max(col, checkpoint_buffer_cycle_timing)

# make sure the OPT target in latency_df matches with leakage_df
leakage_df['OPT target'] = leakage_df['OPT target'].astype(str).str.strip()
latency_df['OPT target'] = latency_df['OPT target'].astype(str).str.strip()
# merge the leakage_df and latency_df on OPT target
leakage_df = pd.merge(leakage_df, latency_df, on='OPT target', how='outer').fillna(0.0)
# calculate the static energy for the weight and checkpointing buffers
leakage_df['HZO 3 Weight Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 3 Weight Buffer Static Power (mW)'] * leakage_df['HZO 3 Weight Buffer Total Read Latency (s)']
leakage_df['HZO 5 Weight Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 5 Weight Buffer Static Power (mW)'] * leakage_df['HZO 5 Weight Buffer Total Read Latency (s)']
leakage_df['HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 3 Checkpointing Buffer Static Power (mW)'] * leakage_df['HZO 3 Checkpointing Buffer Total Write Latency (s)']
leakage_df['HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference'] = leakage_df['HZO 5 Checkpointing Buffer Static Power (mW)'] * leakage_df['HZO 5 Checkpointing Buffer Total Write Latency (s)']

# plot the static energy for both HZO 3 and HZO 5 eNVM weight buffers
st.subheader("Static Energy per Inference (mJ) by OPT target")
static_energy_weight_fig, ax = plt.subplots(figsize=(10, 5))
leakage_df.plot(kind='bar', x='OPT target',
                y=['HZO 3 Weight Buffer Static Energy (mJ) per Inference',
                   'HZO 5 Weight Buffer Static Energy (mJ) per Inference'],
                stacked=False, ax=ax)
ax.set_ylabel('Static Energy per Inference (mJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(leakage_df['OPT target'])), labels=leakage_df['OPT target'], rotation=20)
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Weight Buffer", "HZO 5 Weight Buffer"], loc="upper left")
plt.tight_layout()

# plot the static energy for both HZO 3 and HZO 5 eNVM checkpointing buffers
static_energy_ckpt_fig, ax = plt.subplots(figsize=(10, 5))
leakage_df.plot(kind='bar', x='OPT target',
                y=['HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference',
                   'HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference'],
                stacked=False, ax=ax)
# add hatches to the bars for checkpointing buffers 
hatches = ['///', '\\\\\\']  # HZO3, HZO5
for container, hatch in zip(ax.containers, hatches):
    for bar in container:
        bar.set_hatch(hatch)
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
ax.set_ylabel('Static Energy per Inference (mJ)')
ax.set_xlabel('OPT Target')
ax.set_xticks(range(len(leakage_df['OPT target'])), labels=leakage_df['OPT target'], rotation=20)
handles, _ = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper left")
plt.tight_layout()
col1, col2 = st.columns(2)
col1.pyplot(static_energy_weight_fig)
col2.pyplot(static_energy_ckpt_fig)

# =========================
# Total Energy per Inference (Dynamic + Static)
# =========================

# plot heatmap 
# x-axis: OPT target for weight buffer
# y-axis: OPT target for checkpointing buffer
# color: total energy per inference (mJ)

# -------------------------
# HEATMAP 1: HZO3-only (Total = Weight_total_HZO3[x] + Checkpoint_total_HZO3[y])
# -------------------------

# 1) Build per-OPT totals for HZO3 (dynamic + static) separately for weight and checkpoint
w3_tot = (
    pd.merge(
        energy_base.loc[:, ["OPT target", "HZO 3 Weight Buffer Read Energy per Inference(mJ)"]]
                   .rename(columns={"HZO 3 Weight Buffer Read Energy per Inference(mJ)": "dyn"}),
        leakage_df.loc[:, ["OPT target", "HZO 3 Weight Buffer Static Energy (mJ) per Inference"]]
                  .rename(columns={"HZO 3 Weight Buffer Static Energy (mJ) per Inference": "stat"}),
        on="OPT target", how="outer"
    )
    .fillna(0.0)
)
w3_tot["total"] = w3_tot["dyn"] + w3_tot["stat"]
w3_series = w3_tot.set_index("OPT target")["total"]

c3_tot = (
    pd.merge(
        energy_base.loc[:, ["OPT target", "HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)"]]
                   .rename(columns={"HZO 3 Checkpointing Buffer Write Energy per Inference(mJ)": "dyn"}),
        leakage_df.loc[:, ["OPT target", "HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference"]]
                  .rename(columns={"HZO 3 Checkpointing Buffer Static Energy (mJ) per Inference": "stat"}),
        on="OPT target", how="outer"
    )
    .fillna(0.0)
)
c3_tot["total"] = c3_tot["dyn"] + c3_tot["stat"]
c3_series = c3_tot.set_index("OPT target")["total"]

# 2) Create x (weight OPT target) and y (checkpoint OPT target) label sets
x_labels = sorted(w3_series.index.astype(str).unique().tolist())
y_labels = sorted(c3_series.index.astype(str).unique().tolist())

# 3) Build the heatmap matrix: each cell = weight_total(x) + checkpoint_total(y)
heat1 = np.zeros((len(y_labels), len(x_labels)), dtype=float)
for i, y in enumerate(y_labels):
    for j, x in enumerate(x_labels):
        heat1[i, j] = float(w3_series.get(x, 0.0)) + float(c3_series.get(y, 0.0))

# 4) Plot (pure Matplotlib; no seaborn, no explicit colors)
# st.subheader("Heatmap 1 — Total Energy per Inference (mJ), HZO3-only")
fig_h1, ax_h1 = plt.subplots(figsize=(8, 6))
vmin = np.nanmin(heat1)
vmax = np.nanmax(heat1)
im = ax_h1.imshow(heat1, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
# draw a bounding rectangle around each cell
rows, cols = heat1.shape
for i in range(rows):
    for j in range(cols):
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="black", linewidth=0.8)
        ax_h1.add_patch(rect)
# add text annotations
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax_h1.text(j, i, f"{heat1[i, j]:.5f}", ha="center", va="center", color="black")

# axis labels and ticks
ax_h1.set_xticks(np.arange(len(x_labels)))
ax_h1.set_yticks(np.arange(len(y_labels)))
ax_h1.set_xticklabels(x_labels, rotation=45, ha="right")
ax_h1.set_yticklabels(y_labels)
ax_h1.set_xlabel("Weight OPT target")
ax_h1.set_ylabel("Checkpoint OPT target")
ax_h1.set_title("Total Energy per Inference (mJ): HZO3 weight × HZO3 checkpoint")

# colorbar
cbar = fig_h1.colorbar(im, ax=ax_h1)
cbar.set_label("mJ per inference")

plt.tight_layout()
# st.pyplot(fig_h1)

# -------------------------
# HEATMAP 2: HZO3-only (Total = Weight_total_HZO3[x] + Checkpoint_total_HZO3[y])
# -------------------------
w5_tot = (
    pd.merge(
        energy_base.loc[:, ["OPT target", "HZO 5 Weight Buffer Read Energy per Inference(mJ)"]]
                     .rename(columns={"HZO 5 Weight Buffer Read Energy per Inference(mJ)": "dyn"}),
        leakage_df.loc[:, ["OPT target", "HZO 5 Weight Buffer Static Energy (mJ) per Inference"]]
                    .rename(columns={"HZO 5 Weight Buffer Static Energy (mJ) per Inference": "stat"}),
        on="OPT target", how="outer"
    )
    .fillna(0.0)
)
w5_tot["total"] = w5_tot["dyn"] + w5_tot["stat"]
w5_series = w5_tot.set_index("OPT target")["total"]

c5_tot = (
    pd.merge(
        energy_base.loc[:, ["OPT target", "HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)"]]
                        .rename(columns={"HZO 5 Checkpointing Buffer Write Energy per Inference(mJ)": "dyn"}),
        leakage_df.loc[:, ["OPT target", "HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference"]]
                        .rename(columns={"HZO 5 Checkpointing Buffer Static Energy (mJ) per Inference": "stat"}),
        on="OPT target", how="outer"
    )
    .fillna(0.0)
)
c5_tot["total"] = c5_tot["dyn"] + c5_tot["stat"]
c5_series = c5_tot.set_index("OPT target")["total"]

# 2) Create x (weight OPT target) and y (checkpoint OPT target) label sets
x_labels = sorted(w5_series.index.astype(str).unique().tolist())
y_labels = sorted(c5_series.index.astype(str).unique().tolist())

# 3) Build the heatmap matrix: each cell = weight_total(x) + checkpoint_total(y)
heat2 = np.zeros((len(y_labels), len(x_labels)), dtype=float)
for i, y in enumerate(y_labels):
    for j, x in enumerate(x_labels):
        heat2[i, j] = float(w5_series.get(x, 0.0)) + float(c5_series.get(y, 0.0))

# 4) Plot (pure Matplotlib; no seaborn, no explicit colors)
# st.subheader("Heatmap 2 — Total Energy per Inference (mJ), HZO5-only")
fig_h2, ax_h2 = plt.subplots(figsize=(8, 6))
vmin = np.nanmin(heat2) 
vmax = np.nanmax(heat2)
im = ax_h2.imshow(heat2, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
# draw a bounding rectangle around each cell
rows, cols = heat2.shape
for i in range(rows):
    for j in range(cols):
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="black", linewidth=0.8)
        ax_h2.add_patch(rect)
# add text annotations
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        text = ax_h2.text(j, i, f"{heat2[i, j]:.5f}", ha="center", va="center", color="black")
# axis labels and ticks
ax_h2.set_xticks(np.arange(len(x_labels)))
ax_h2.set_yticks(np.arange(len(y_labels)))
ax_h2.set_xticklabels(x_labels, rotation=45, ha="right")
ax_h2.set_yticklabels(y_labels)
ax_h2.set_xlabel("Weight OPT target")
ax_h2.set_ylabel("Checkpoint OPT target")
ax_h2.set_title("Total Energy per Inference (mJ): HZO5 weight × HZO5 checkpoint")
# colorbar
cbar = fig_h2.colorbar(im, ax=ax_h2)
cbar.set_label("mJ per inference")
plt.tight_layout()
# st.pyplot(fig_h2)

# -------------------------
# HEATMAP 3: Best-of-each (per OPT target, choose min across HZO3/HZO5)
# -------------------------
# ---- make latency lookup dicts ----
# Weight (read latency) dicts
w3_rl_dict = df_3_w_static_latency.set_index("OPT target")["HZO 3 Weight Buffer read latency (ns)"].to_dict()
w5_rl_dict = df_5_w_static_latency.set_index("OPT target")["HZO 5 Weight Buffer read latency (ns)"].to_dict()
# Checkpoint (write latency) dicts
c3_wl_dict = df_3_c_static_latency.set_index("OPT target")["HZO 3 Checkpointing Buffer write latency (ns)"].to_dict()
c5_wl_dict = df_5_c_static_latency.set_index("OPT target")["HZO 5 Checkpointing Buffer write latency (ns)"].to_dict()

def _min_ignore_nan(a, b):
    if a is None or (isinstance(a, float) and np.isnan(a)): a = np.inf
    if b is None or (isinstance(b, float) and np.isnan(b)): b = np.inf
    m = min(a, b)
    return None if not np.isfinite(m) else m

def pick_rl_from_tag(tag: str, key: str):
    """Return read latency (ns) for WEIGHT side given tag w3/w5/w3/5."""
    key = str(key)
    if tag == "w3":
        return w3_rl_dict.get(key, None)
    if tag == "w5":
        return w5_rl_dict.get(key, None)
    if tag == "w3/5":
        return _min_ignore_nan(w3_rl_dict.get(key, None), w5_rl_dict.get(key, None))
    return None  # w?

def pick_wl_from_tag(tag: str, key: str):
    """Return write latency (ns) for CHECKPOINT side given tag c3/c5/c3/5."""
    key = str(key)
    if tag == "c3":
        return c3_wl_dict.get(key, None)
    if tag == "c5":
        return c5_wl_dict.get(key, None)
    if tag == "c3/5":
        return _min_ignore_nan(c3_wl_dict.get(key, None), c5_wl_dict.get(key, None))
    return None  # c?

def pick_best_with_ties(d3, d5, key, base, tol=1e-12):
    """
    d3/d5: dicts for HZO3/HZO5 totals
    key: OPT target
    base: 'c' for checkpoint, 'w' for weight
    Returns (value, tag) where tag in {f'{base}3', f'{base}5', f'{base}3/5', f'{base}?'}
    """
    a = d3.get(key, np.inf)
    b = d5.get(key, np.inf)
    a_finite, b_finite = np.isfinite(a), np.isfinite(b)

    if a_finite and b_finite:
        # relative + absolute tolerance
        diff = abs(a - b)
        scale = max(1.0, abs(a), abs(b))
        if diff <= tol * scale:
            return float(a), f"{base}3/5"  # tie: mark both
        return (float(a), f"{base}3") if a < b else (float(b), f"{base}5")
    if a_finite:
        return float(a), f"{base}3"
    if b_finite:
        return float(b), f"{base}5"
    return 0.0, f"{base}?"

# Dicts from earlier series
w3_dict = w3_series.to_dict()
w5_dict = w5_series.to_dict()
c3_dict = c3_series.to_dict()
c5_dict = c5_series.to_dict()

# Labels (union across techs)
x_labels3 = sorted(set(map(str, w3_series.index)) | set(map(str, w5_series.index)))
y_labels3 = sorted(set(map(str, c3_series.index)) | set(map(str, c5_series.index)))

# Build matrices + tags + chosen latencies
rows, cols = len(y_labels3), len(x_labels3)
heat3 = np.zeros((rows, cols), dtype=float)
tag3  = np.empty((rows, cols), dtype=object)
rl3   = np.empty((rows, cols), dtype=object)  # read latency (ns) for WEIGHT
wl3   = np.empty((rows, cols), dtype=object)  # write latency (ns) for CHECKPOINT

for i, y in enumerate(y_labels3):
    c_val, c_tag = pick_best_with_ties(c3_dict, c5_dict, y, base="c")
    for j, x in enumerate(x_labels3):
        w_val, w_tag = pick_best_with_ties(w3_dict, w5_dict, x, base="w")
        heat3[i, j] = c_val + w_val
        tag3[i, j]  = f"({c_tag},{w_tag})"
        # Choose RL/WL to display; for 3/5 we pick the LOWER latency
        rl_val = pick_rl_from_tag(w_tag, x)
        wl_val = pick_wl_from_tag(c_tag, y)
        rl3[i, j] = None if rl_val is None else float(rl_val)
        wl3[i, j] = None if wl_val is None else float(wl_val)

# Plot
st.subheader("Heatmap 3 — Total Energy per Inference (mJ), best HZO per side (tags show picks)")
fig_h3, ax_h3 = plt.subplots(figsize=(8, 6))
vmin, vmax = np.nanmin(heat3), np.nanmax(heat3)
im = ax_h3.imshow(heat3, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)

# Cell borders + annotations (value, picks, RL/WL)
for i in range(rows):
    for j in range(cols):
        ax_h3.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.8))
        # Build annotation text
        val_txt = f"{heat3[i, j]:.5f}"
        pick_txt = tag3[i, j]
        rl_txt = "-" if rl3[i, j] is None else f"{rl3[i, j]:.3f}"
        wl_txt = "-" if wl3[i, j] is None else f"{wl3[i, j]:.3f}"
        ax_h3.text(
            j, i,
            f"{val_txt}\n{pick_txt}\nR:{rl_txt}\nW:{wl_txt}",
            ha="center", va="center", fontsize=9, color="black"
        )

# Axes + colorbar
ax_h3.set_xticks(np.arange(len(x_labels3)))
ax_h3.set_yticks(np.arange(len(y_labels3)))
ax_h3.set_xticklabels(x_labels3, rotation=45, ha="right")
ax_h3.set_yticklabels(y_labels3)
ax_h3.set_xlabel("Weight OPT target (best of HZO3/HZO5)")
ax_h3.set_ylabel("Checkpoint OPT target (best of HZO3/HZO5)")
ax_h3.set_title("Total Energy per Inference (mJ): best weight × best checkpoint")

cbar = fig_h3.colorbar(im, ax=ax_h3)
cbar.set_label("mJ per inference")

plt.tight_layout()
st.text(
    "Each cell shows:\n"
    "• Total energy per inference (mJ)\n"
    "• The chosen HZO technology for the checkpoint/weight buffer\n"
    "    - e.g. c3: checkpointing with HZO3 is better\n"
    "    - e.g. w5: weight with HZO5 weight\n"
    "    - e.g. c3/5: checkpointing with buffer with HZO3 or HZO5 yields the same result\n"
    "• RL = read latency (ns) for the chosen weight side\n"
    "• WL = write latency (ns) for the chosen checkpoint side\n"
    "    - For 3/5 tags, RL/WL show the lower latency between HZO3 and HZO5\n"
)
st.pyplot(fig_h3)
st.text("The optimal results are "
        "weight buffer with ReadEDP as the OPT target and based on either HZO3 or HZO5 +"
        "checkpointing buffer with either ReadDynamicEnergy or ReadEDP as the OPT based on HZO3 technology")

# read energy stacked bar
# checkpoointing read count for each layer = checkointing write count for each layer
# model_df = pd.read_csv("vww.csv")
# # read the IFMAP Height and Width, as well as channels to calculate the output size of each layer
# ifmap_height = model_df['IFMAP Height'].values[0]
# ifmap_width = model_df['IFMAP Width'].values[0]
# ifmap_channels = model_df['Channels'].values[0]
# # calculate the output size of each layer
# output_size = ifmap_height * ifmap_width * ifmap_channels
# # calculate the read energy per layer for both HZO 3 and HZO 5 eNVM checkpointing buffers

# calculate the per-layer read energy for each layer under HZO 3 and HZO 5 with different OPT targets and plot as a stacked bar chart
# import the vww.csv and calculate the read energy per layer based on the 

# def read_energy_per_layer(df, colname):
#     out = df.loc[:, ["OPT target", "Read Energy (pJ)"]].copy()
#     out["OPT target"] = out["OPT target"].astype(str).str.strip()
#     out["Read Energy (pJ)"] = pd.to_numeric(out["Read Energy (pJ)"], errors="coerce")
#     out["Read Energy per Layer (pJ)"] = out["Read Energy (pJ)"] * checkpoint_reads
#     print(out["Read Energy per Layer (pJ)"])
#     return out.rename(columns={"Read Energy per Layer (pJ)": colname})
# df_3_c_read_energy = read_energy_per_layer(hzo_3_checkpoint_df, "HZO 3 Checkpointing Buffer Read Energy per Layer (pJ)")
# df_5_c_read_energy = read_energy_per_layer(hzo_5_checkpoint_df, "HZO 5 Checkpointing Buffer Read Energy per Layer (pJ)")

# # Outer-merge all on OPT target so missing entries are allowed
# read_energy_df = reduce(
#     lambda left, right: pd.merge(left, right, on="OPT target", how="outer"),
#     [df_3_c_read_energy, df_5_c_read_energy]
# ).sort_values("OPT target").reset_index(drop=True)

# # plot the read energy per layer for both HZO 3 and HZO 5 eNVM checkpointing buffers
# read_energy_fig, ax = plt.subplots(figsize=(10, 5))
# read_energy_df.plot(kind='bar', x='OPT target',
#                      y=['HZO 3 Checkpointing Buffer Read Energy per Layer (pJ)',
#                         'HZO 5 Checkpointing Buffer Read Energy per Layer (pJ)'],
#                      stacked=False, ax=ax)
# # add hatches to the bars for checkpointing buffers
# hatches = ['///', '\\\\\\']  # HZO3, HZO5
# for container, hatch in zip(ax.containers, hatches):
#     for bar in container:
#         bar.set_hatch(hatch)
#         bar.set_edgecolor('black')
#         bar.set_linewidth(0.8)
# ax.set_ylabel('Read Energy per Layer (pJ)')
# ax.set_xlabel('OPT Target')
# ax.set_xticks(range(len(read_energy_df['OPT target'])), labels=read_energy_df['OPT target'], rotation=20)
# ax.set_title('Read Energy per Layer for HZO eNVM Checkpointing Buffers')
# handles, _ = ax.get_legend_handles_labels()
# if handles:
#     ax.legend(handles, ["HZO 3 Checkpointing Buffer", "HZO 5 Checkpointing Buffer"], loc="upper left")
# plt.tight_layout()
# st.subheader("Read Energy per Layer (pJ) by OPT target")
# st.pyplot(read_energy_fig)
