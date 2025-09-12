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

# # memory capacities in kb
# filter_double_buffer_capacity = 2048 #kb
# ifmap_double_buffer_capacity = 8192 #kb
# ofmap_double_buffer_capacity = 8192 #kb

systolic_array_area = 0.00121*32*32 # mm^2, 0.00121 mm^2 per PE, 32x32 array

# clock frequency in Hz
clock_frequency = 20000000 #20MHz
period = 1 / clock_frequency

# # =========================
# # Load Data
# # =========================

# locate the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# path to scalesim output file
scalesim_output_file = os.path.join(current_dir, '../edgeTPU_fitMobileNet_experiment_results/edgeTPU_ws_MobileNet/DETAILED_ACCESS_REPORT.csv')
# path to SRAM and HZO data file
HZO_data_file = os.path.join(current_dir, 'HZO_data/combined_HZO3&5_cellsize30_2&4MB.csv')
# HZO_data_file = os.path.join(current_dir, 'HZO_data/combined_HZO3&HZO5_2MB&4MB.csv')
SRAM_data_file = os.path.join(current_dir, 'HZO_data/combined_SRAM_2&4&32MB.csv')

# extract 4MB HZO
hzo_df = pd.read_csv(HZO_data_file)
hzo_df['OPT target'] = hzo_df['Source'].str.split('_').str[2]
# sort OPT target in an alphabetical order
hzo_df['OPT target'] = hzo_df['OPT target'].astype(str).str.strip()
hzo_df = hzo_df.sort_values(by=['OPT target'])
hzo_3_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_30_3_9.cell']
hzo_5_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_30_5_9.cell']
hzo_3_weight_df = hzo_3_df[hzo_3_df['Capacity (MB)'] == 4]
hzo_5_weight_df = hzo_5_df[hzo_5_df['Capacity (MB)'] ==  4]

# extract 4MB SRAM, and 32MB SRAM data
sram_df = pd.read_csv(SRAM_data_file)
sram_df['OPT target'] = sram_df['Source'].str.split('_').str[2]
# sort OPT target in an alphabetical order
sram_df['OPT target'] = sram_df['OPT target'].astype(str).str.strip()
sram_df = sram_df.sort_values(by=['OPT target'])
sram_best_df = sram_df[sram_df['MemoryCellInputFile'] == ' data/cell_cfgs/SRAM_best_case.cell']
sram_worst_df = sram_df[sram_df['MemoryCellInputFile'] == ' data/cell_cfgs/SRAM_worst_case.cell']
sram_best_weight_df = sram_best_df[sram_best_df['Capacity (MB)'] == 4]
sram_worst_weight_df = sram_worst_df[sram_worst_df['Capacity (MB)'] == 4]
sram_best_io_df = sram_best_df[sram_best_df['Capacity (MB)'] == 2]
sram_worst_io_df = sram_worst_df[sram_worst_df['Capacity (MB)'] == 2]

# read scalesim output file and extract relevant data
scalesim_df = pd.read_csv(scalesim_output_file)

weight_reads = scalesim_df[' SRAM Filter Reads'].sum()  
output_writes = scalesim_df[' SRAM OFMAP Writes'].sum()
input_reads = scalesim_df[' SRAM IFMAP Reads'].sum()

weight_start_cycle = scalesim_df[' SRAM Filter Start Cycle']
weight_end_cycle = scalesim_df[' SRAM Filter Stop Cycle']

output_start_cycle = scalesim_df[' SRAM OFMAP Start Cycle']
output_end_cycle = scalesim_df[' SRAM OFMAP Stop Cycle']     

input_start_cycle = scalesim_df[' SRAM IFMAP Start Cycle']
input_end_cycle = scalesim_df[' SRAM IFMAP Stop Cycle']

# # =========================
# # Area Calculation
# # =========================
# plot 32MB SRAM data under different OPT targets
def area_prep(df, colname):
    out = df.loc[:, ["OPT target", "Area (mm^2)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Area (mm^2)"] = pd.to_numeric(out["Area (mm^2)"], errors="coerce")
    return out.rename(columns={"Area (mm^2)": colname})

df_sram_best_io_area = area_prep(sram_best_io_df, "SRAM IO buffer area (mm^2)")
df_sram_worst_io_area = area_prep(sram_worst_io_df, "SRAM IO buffer area (mm^2)")

df_sram_best_weight_area = area_prep(sram_best_weight_df, "SRAM Weight Buffer Area (mm^2)")
df_sram_worst_weight_area = area_prep(sram_worst_weight_df, "SRAM Weight Buffer Area (mm^2)")
df_hzo_3_weight_area = area_prep(hzo_3_weight_df, "HZO3 Weight Buffer Area (mm^2)")
df_hzo_5_weight_area = area_prep(hzo_5_weight_df, "HZO5 Weight Buffer Area (mm^2)")

# mix and match the IO buffer under different OPT targets with different weight buffer technologies 
# under the same OPT target
# make the first column the OPT target of the IO buffer
# make the second column the OPT target of the weight buffer
# make the third column the wright buffer technology (SRAM, HZO3, HZO5)
# make the fourth column the total area of the IO buffer and weight buffer
# -----------------------------
# Keep only IO best
# -----------------------------
io_best = df_sram_best_io_area.rename(
    columns={
        "OPT target": "IO buffer OPT target",
        "SRAM IO buffer area (mm^2)": "IO Area (mm^2)"
    }
)
io_best["IO Area (mm^2)"] = pd.to_numeric(io_best["IO Area (mm^2)"], errors="coerce")
io_best["IO buffer OPT target"] = io_best["IO buffer OPT target"].astype(str).str.strip()

# -----------------------------
# Weight (4MB) technologies: SRAM-Best, HZO3, HZO5
# -----------------------------
w_sram_best = df_sram_best_weight_area.rename(
    columns={"OPT target": "Weight buffer OPT target",
             "SRAM Weight Buffer Area (mm^2)": "Weight Area (mm^2)"}
).assign(**{"Weight buffer Tech": "SRAM-Best"})

w_hzo3 = df_hzo_3_weight_area.rename(
    columns={"OPT target": "Weight buffer OPT target",
             "HZO3 Weight Buffer Area (mm^2)": "Weight Area (mm^2)"}
).assign(**{"Weight buffer Tech": "HZO3"})

w_hzo5 = df_hzo_5_weight_area.rename(
    columns={"OPT target": "Weight buffer OPT target",
             "HZO5 Weight Buffer Area (mm^2)": "Weight Area (mm^2)"}
).assign(**{"Weight buffer Tech": "HZO5"})

weight_all = pd.concat([w_sram_best, w_hzo3, w_hzo5], ignore_index=True)
weight_all["Weight Area (mm^2)"] = pd.to_numeric(weight_all["Weight Area (mm^2)"], errors="coerce")
weight_all["Weight buffer OPT target"] = weight_all["Weight buffer OPT target"].astype(str).str.strip()

# -----------------------------
# Cartesian product (mix & match)
# -----------------------------
mix_df = io_best.merge(weight_all, how="cross")
# mix_df["Total Area (mm^2)"] = mix_df["IO Area (mm^2)"] + mix_df["Weight Area (mm^2)"] + systolic_array_area
mix_df["Systolic Array Area (mm^2)"] = systolic_array_area

# Final columns only
mix_df = mix_df[[
    "IO buffer OPT target",
    "Weight buffer OPT target",
    "Weight buffer Tech",
    "IO Area (mm^2)",
    "Weight Area (mm^2)",
    "Systolic Array Area (mm^2)",
]].sort_values(["IO buffer OPT target", "Weight buffer OPT target", "Weight buffer Tech"]).reset_index(drop=True)

print(mix_df)
# save df to csv
mix_df.to_csv("scripts/HZO_analysis_results/area_mix_and_match.csv", index=False)

# plot multiple bars for the best and worst case SRAM area over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_area['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, df_sram_best_io_area['SRAM IO buffer area (mm^2)'], width, label='Best Case SRAM Area', color='b')
bars2 = ax.bar(x + width/2, df_sram_worst_io_area['SRAM IO buffer area (mm^2)'], width, label='Worst Case SRAM Area', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Area (mm^2)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_io_area['OPT target'], rotation=45)
ax.legend()

# plot multiple bars for the best and worst case SRAM area and HZO area over different OPT targets
fig_weight, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # the width of the bars
x = np.arange(len(df_sram_best_weight_area['OPT target']))  # the label locations
bars1 = ax.bar(x - width, df_sram_best_weight_area['SRAM Weight Buffer Area (mm^2)'], width, label='Best Case SRAM Area', color='b')
bars2 = ax.bar(x, df_hzo_5_weight_area['HZO5 Weight Buffer Area (mm^2)'], width, label='HZO5 Area', color='g')
bars3 = ax.bar(x + width, df_hzo_3_weight_area['HZO3 Weight Buffer Area (mm^2)'], width, label='HZO3 Area', color='y')
bars4 = ax.bar(x + 2*width, df_sram_worst_weight_area['SRAM Weight Buffer Area (mm^2)'], width, label='Worst Case SRAM Area', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Area (mm^2)')
ax.set_title('Weight buffer (4MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_weight_area['OPT target'], rotation=45)
ax.legend()

st.subheader("Area")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)
col2.pyplot(fig_weight)

area_df = reduce(lambda left, right: pd.merge(left, right, on='OPT target'), [df_sram_best_io_area, df_sram_worst_io_area, df_sram_best_weight_area, df_sram_worst_weight_area, df_hzo_3_weight_area, df_hzo_5_weight_area]).sort_values(['OPT target']).reset_index(drop=True)

print(area_df)

# # =========================
# # Dynamic read energy per access
# # =========================

def dynamic_read_prep(df, colname):
    out = df.loc[:, ["OPT target", "Read Energy (pJ)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Read Energy (pJ)"] = pd.to_numeric(out["Read Energy (pJ)"], errors="coerce")
    return out.rename(columns={"Read Energy (pJ)": colname})

df_sram_best_io_dynamic_Read = dynamic_read_prep(sram_best_io_df, "SRAM 32MB Dynamic Read Energy (pJ)")
df_sram_worst_io_dynamic_Read = dynamic_read_prep(sram_worst_io_df, "SRAM 32MB Dynamic Read Energy (pJ)")

df_sram_best_weight_dynamic_Read = dynamic_read_prep(sram_best_weight_df, "SRAM 4MB Dynamic Read Energy (pJ)")
df_sram_worst_weight_dynamic_Read = dynamic_read_prep(sram_worst_weight_df, "SRAM 4MB Dynamic Read Energy (pJ)")
df_hzo_3_weight_dynamic_Read = dynamic_read_prep(hzo_3_weight_df, "HZO3 4MB Dynamic Read Energy (pJ)")
df_hzo_5_weight_dynamic_Read = dynamic_read_prep(hzo_5_weight_df, "HZO5 4MB Dynamic Read Energy (pJ)")

dyRead_df = reduce(lambda left, right: pd.merge(left, right, on='OPT target'), [df_sram_best_io_dynamic_Read, df_sram_worst_io_dynamic_Read, df_sram_best_weight_dynamic_Read, df_sram_worst_weight_dynamic_Read, df_hzo_3_weight_dynamic_Read, df_hzo_5_weight_dynamic_Read]).sort_values(['OPT target']).reset_index(drop=True)

# plot multiple bars for the best and worst case SRAM dynamic read energy over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_dynamic_Read['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, df_sram_best_io_dynamic_Read['SRAM 32MB Dynamic Read Energy (pJ)'], width, label='Best Case SRAM Dynamic Read Energy', color='b')
bars2 = ax.bar(x + width/2, df_sram_worst_io_dynamic_Read['SRAM 32MB Dynamic Read Energy (pJ)'], width, label='Worst Case SRAM Dynamic Read Energy', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dynamic Read Energy (pJ)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_io_dynamic_Read['OPT target'], rotation=45)
ax.legend()

# plot multiple bars for the best and worst case SRAM dynamic read energy and HZO dynamic read energy over different OPT targets
fig_weight, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # the width of the bars
x = np.arange(len(df_sram_best_weight_dynamic_Read['OPT target']))  # the label locations
bars1 = ax.bar(x - width, df_sram_best_weight_dynamic_Read['SRAM 4MB Dynamic Read Energy (pJ)'], width, label='Best Case SRAM Dynamic Read Energy', color='b')
bars2 = ax.bar(x, df_hzo_5_weight_dynamic_Read['HZO5 4MB Dynamic Read Energy (pJ)'], width, label='HZO5 Dynamic Read Energy', color='g')
bars3 = ax.bar(x + width, df_hzo_3_weight_dynamic_Read['HZO3 4MB Dynamic Read Energy (pJ)'], width, label='HZO3 Dynamic Read Energy', color='y')
bars4 = ax.bar(x + 2*width, df_sram_worst_weight_dynamic_Read['SRAM 4MB Dynamic Read Energy (pJ)'], width, label='Worst Case SRAM Dynamic Read Energy', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dynamic Read Energy (pJ)')
ax.set_title('Weight buffer (4MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_weight_dynamic_Read['OPT target'], rotation=45)
ax.legend()

st.subheader("Dynamic Read Energy per Access")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)
col2.pyplot(fig_weight)

# # =========================
# # Dynamic write energy per access
# # =========================
def dynamic_write_prep(df, colname):
    out = df.loc[:, ["OPT target", "Write Energy (pJ)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Write Energy (pJ)"] = pd.to_numeric(out["Write Energy (pJ)"], errors="coerce")
    return out.rename(columns={"Write Energy (pJ)": colname})

df_sram_best_io_dynamic_Write = dynamic_write_prep(sram_best_io_df, "SRAM 32MB Dynamic Write Energy (pJ)")
df_sram_worst_io_dynamic_Write = dynamic_write_prep(sram_worst_io_df, "SRAM 32MB Dynamic Write Energy (pJ)")

dyWrite_df = reduce(lambda left, right: pd.merge(left, right, on='OPT target'), [df_sram_best_io_dynamic_Write, df_sram_worst_io_dynamic_Write]).sort_values(['OPT target']).reset_index(drop=True)

# plot multiple bars for the best and worst case SRAM dynamic write energy over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_dynamic_Write['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, df_sram_best_io_dynamic_Write['SRAM 32MB Dynamic Write Energy (pJ)'], width, label='Best Case SRAM Dynamic Write Energy', color='b')
bars2 = ax.bar(x + width/2, df_sram_worst_io_dynamic_Write['SRAM 32MB Dynamic Write Energy (pJ)'], width, label='Worst Case SRAM Dynamic Write Energy', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dynamic Write Energy (pJ)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_io_dynamic_Write['OPT target'], rotation=45)
ax.legend()

st.subheader("Dynamic Write Energy per Access")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)

# # =========================
# dynamic energy per inference
# # =========================
# io buffer dynamic energy per inference
def dynamic_energy_per_inference(df, read_colname, write_colname, read_count, write_count):
    out = df.loc[:, ["OPT target", read_colname, write_colname]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out[read_colname] = pd.to_numeric(out[read_colname], errors="coerce")
    out[write_colname] = pd.to_numeric(out[write_colname], errors="coerce")
    out["Dynamic Energy per Inference (mJ)"] = (out[read_colname] * read_count + out[write_colname] * write_count) * 1e-6
    return out.loc[:, ["OPT target", "Dynamic Energy per Inference (mJ)"]]
# weight buffer dynamic energy per inference
def dynamic_energy_per_inference_weight(df, read_colname, read_count):
    out = df.loc[:, ["OPT target", read_colname]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out[read_colname] = pd.to_numeric(out[read_colname], errors="coerce")
    out["Dynamic Energy per Inference (mJ)"] = (out[read_colname] * read_count) * 1e-6
    return out.loc[:, ["OPT target", "Dynamic Energy per Inference (mJ)"]]
df_sram_best_io_dynamic_energy = dynamic_energy_per_inference(df_sram_best_io_dynamic_Read.merge(df_sram_best_io_dynamic_Write, on='OPT target'), "SRAM 32MB Dynamic Read Energy (pJ)", "SRAM 32MB Dynamic Write Energy (pJ)", input_reads, output_writes)
df_sram_worst_io_dynamic_energy = dynamic_energy_per_inference(df_sram_worst_io_dynamic_Read.merge(df_sram_worst_io_dynamic_Write, on='OPT target'), "SRAM 32MB Dynamic Read Energy (pJ)", "SRAM 32MB Dynamic Write Energy (pJ)", input_reads, output_writes)

df_hzo_3_weight_dynamic_energy = dynamic_energy_per_inference_weight(df_hzo_3_weight_dynamic_Read, "HZO3 4MB Dynamic Read Energy (pJ)", weight_reads)
df_hzo_5_weight_dynamic_energy = dynamic_energy_per_inference_weight(df_hzo_5_weight_dynamic_Read, "HZO5 4MB Dynamic Read Energy (pJ)", weight_reads)
df_sram_best_weight_dynamic_energy = dynamic_energy_per_inference_weight(df_sram_best_weight_dynamic_Read, "SRAM 4MB Dynamic Read Energy (pJ)", weight_reads)
df_sram_worst_weight_dynamic_energy = dynamic_energy_per_inference_weight(df_sram_worst_weight_dynamic_Read, "SRAM 4MB Dynamic Read Energy (pJ)", weight_reads)

# plot multiple bars for the best and worst case SRAM dynamic energy per inference over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_dynamic_energy['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, df_sram_best_io_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='Best Case SRAM Dynamic Energy per Inference', color='b')
bars2 = ax.bar(x + width/2, df_sram_worst_io_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='Worst Case SRAM Dynamic Energy per Inference', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dynamic Energy per Inference (mJ)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_io_dynamic_energy['OPT target'], rotation=45)
ax.legend()

# plot multiple bars for the best and worst case SRAM dynamic energy per inference and HZO dynamic energy per inference over different OPT targets
fig_weight, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # the width of the bars
x = np.arange(len(df_sram_best_weight_dynamic_energy['OPT target']))  # the label locations
bars1 = ax.bar(x - width, df_sram_best_weight_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='Best Case SRAM Dynamic Energy per Inference', color='b')
bars2 = ax.bar(x, df_hzo_5_weight_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='HZO5 Dynamic Energy per Inference', color='g')
bars3 = ax.bar(x + width, df_hzo_3_weight_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='HZO3 Dynamic Energy per Inference', color='y')
bars4 = ax.bar(x + 2*width, df_sram_worst_weight_dynamic_energy['Dynamic Energy per Inference (mJ)'], width, label='Worst Case SRAM Dynamic Energy per Inference', color='r')
# annotate bars with values
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation = 45)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation = 45)
for bar in bars3:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation = 45)
for bar in bars4:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation = 45)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dynamic Energy per Inference (mJ)')
ax.set_title('Weight buffer (4MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_weight_dynamic_energy['OPT target'], rotation=45)
ax.legend()

st.subheader("Dynamic Energy per Inference")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)
col2.pyplot(fig_weight)

# # =========================
# static energy per inference
# # =========================
# extract the leakage power from the HZO data file
def static_energy_prep(df, colname):
    out = df.loc[:, ["OPT target", "Leakage Power (mW)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Leakage Power (mW)"] = pd.to_numeric(out["Leakage Power (mW)"], errors="coerce")
    return out.rename(columns={"Leakage Power (mW)": colname})

df_sram_best_io_static_power = static_energy_prep(sram_best_io_df, "SRAM 32MB Best Static Power (mW)")
df_sram_worst_io_static_power = static_energy_prep(sram_worst_io_df, "SRAM 32MB Worst Static Power (mW)")
df_sram_best_weight_static_power = static_energy_prep(sram_best_weight_df, "SRAM 4MB Best Static Power (mW)")
df_sram_worst_weight_static_power = static_energy_prep(sram_worst_weight_df, "SRAM 4MB Worst Static Power (mW)")
df_hzo_3_weight_static_power = static_energy_prep(hzo_3_weight_df, "HZO3 4MB Static Power (mW)")
df_hzo_5_weight_static_power = static_energy_prep(hzo_5_weight_df, "HZO5 4MB Static Power (mW)")

leak_df = reduce(lambda left, right: pd.merge(left, right, on='OPT target'), [df_sram_best_io_static_power, df_sram_worst_io_static_power, df_sram_best_weight_static_power, df_sram_worst_weight_static_power, df_hzo_3_weight_static_power, df_hzo_5_weight_static_power]).sort_values(['OPT target']).reset_index(drop=True)

# method 1
input_buffer_cycle_counts = input_end_cycle - input_start_cycle
input_buffer_cycle_counts_total = input_buffer_cycle_counts.sum()
input_buffer_time = input_buffer_cycle_counts_total * period # in seconds

output_buffer_cycle_counts = output_end_cycle - output_start_cycle
output_buffer_cycle_counts_total = output_buffer_cycle_counts.sum()
output_buffer_time = output_buffer_cycle_counts_total * period # in seconds

weight_buffer_cycle_counts = weight_end_cycle - weight_start_cycle
weight_buffer_cycle_counts_total = weight_buffer_cycle_counts.sum()
weight_buffer_time = weight_buffer_cycle_counts_total * period # in seconds

# method 2
# extract read and write latencies from the HZO data file
def static_read_latency_prep(df, colname):
    out = df.loc[:, ["OPT target", "Read Latency (ns)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Read Latency (ns)"] = pd.to_numeric(out["Read Latency (ns)"], errors="coerce")
    return out.rename(columns={"Read Latency (ns)": colname})
df_sram_best_io_read_latency = static_read_latency_prep(sram_best_io_df, "SRAM 32MB Best Read Latency (ns)")
df_sram_worst_io_read_latency = static_read_latency_prep(sram_worst_io_df, "SRAM 32MB Worst Read Latency (ns)")
df_sram_best_weight_read_latency = static_read_latency_prep(sram_best_weight_df, "SRAM 4MB Best Read Latency (ns)")
df_sram_worst_weight_read_latency = static_read_latency_prep(sram_worst_weight_df, "SRAM 4MB Worst Read Latency (ns)")
df_hzo_3_weight_read_latency = static_read_latency_prep(hzo_3_weight_df, "HZO3 4MB Read Latency (ns)")
df_hzo_5_weight_read_latency = static_read_latency_prep(hzo_5_weight_df, "HZO5 4MB Read Latency (ns)")

def static_write_latency_prep(df, colname):
    out = df.loc[:, ["OPT target", "Write Latency (ns)"]].copy()
    out["OPT target"] = out["OPT target"].astype(str).str.strip()
    out["Write Latency (ns)"] = pd.to_numeric(out["Write Latency (ns)"], errors="coerce")
    return out.rename(columns={"Write Latency (ns)": colname})
df_sram_best_io_write_latency = static_write_latency_prep(sram_best_io_df, "SRAM 32MB Best Write Latency (ns)")
df_sram_worst_io_write_latency = static_write_latency_prep(sram_worst_io_df, "SRAM 32MB Worst Write Latency (ns)")

# calculate the latency per access times the number of accesses
latency_df = reduce(lambda left, right: pd.merge(left, right, on='OPT target'), [df_sram_best_io_read_latency, df_sram_best_io_write_latency, df_sram_worst_io_read_latency, df_sram_worst_io_write_latency, df_sram_best_weight_read_latency, df_sram_worst_weight_read_latency, df_hzo_3_weight_read_latency, df_hzo_5_weight_read_latency]).sort_values(['OPT target']).reset_index(drop=True)
latency_df["SRAM 32MB Best Read Latency (s)"] = (latency_df["SRAM 32MB Best Read Latency (ns)"] * input_reads) * 1e-9
latency_df["SRAM 32MB Best Write Latency (s)"] = (latency_df["SRAM 32MB Best Write Latency (ns)"] * output_writes) * 1e-9
latency_df["SRAM 32MB Worst Read Latency (s)"] = (latency_df["SRAM 32MB Worst Read Latency (ns)"] * input_reads) * 1e-9
latency_df["SRAM 32MB Worst Write Latency (s)"] = (latency_df["SRAM 32MB Worst Write Latency (ns)"] * output_writes) * 1e-9
latency_df["SRAM 4MB Best Read Latency (s)"] = (latency_df["SRAM 4MB Best Read Latency (ns)"] * weight_reads) * 1e-9
latency_df["SRAM 4MB Worst Read Latency (s)"] = (latency_df["SRAM 4MB Worst Read Latency (ns)"] * weight_reads) * 1e-9
latency_df["HZO3 4MB Read Latency (s)"] = (latency_df["HZO3 4MB Read Latency (ns)"] * weight_reads) * 1e-9
latency_df["HZO5 4MB Read Latency (s)"] = (latency_df["HZO5 4MB Read Latency (ns)"] * weight_reads) * 1e-9
# latency_df["SRAM 32MB Total Latency (s)"] = (latency_df["SRAM 32MB Read Latency (ns)"] * input_reads + latency_df["SRAM 32MB Write Latency (ns)"] * output_writes) * 1e-9
# latency_df["SRAM 4MB Total Latency (s)"] = (latency_df["SRAM 4MB Read Latency (ns)"] * weight_reads) * 1e-9
# latency_df["HZO3 4MB Total Latency (s)"] = (latency_df["HZO3 4MB Read Latency (ns)"] * weight_reads + latency_df["HZO3 4MB Write Latency (ns)"] * weight_reads) * 1e-9
# latency_df["HZO5 4MB Total Latency (s)"] = (latency_df["HZO5 4MB Read Latency (ns)"] * weight_reads + latency_df["HZO5 4MB Write Latency (ns)"] * weight_reads) * 1e-9

# compare the two methods and take the larger one
for col in latency_df["SRAM 32MB Best Read Latency (s)"]:
    col = max(col, input_buffer_time)
for col in latency_df["SRAM 32MB Best Write Latency (s)"]:
    col = max(col, output_buffer_time)
for col in latency_df["SRAM 32MB Worst Read Latency (s)"]:
    col = max(col, input_buffer_time)
for col in latency_df["SRAM 32MB Worst Write Latency (s)"]:
    col = max(col, output_buffer_time)
for col in latency_df["SRAM 4MB Best Read Latency (s)"]:
    col = max(col, weight_buffer_time)
for col in latency_df["SRAM 4MB Worst Read Latency (s)"]:
    col = max(col, weight_buffer_time)
for col in latency_df["HZO3 4MB Read Latency (s)"]:
    col = max(col, weight_buffer_time)
for col in latency_df["HZO5 4MB Read Latency (s)"]:
    col = max(col, weight_buffer_time) 

# establish a new column for the total latency per access
latency_df["SRAM 32MB Best Latency (s)"] = 0
latency_df["SRAM 32MB Worst Latency (s)"] = 0
latency_df["SRAM 4MB Best Latency (s)"] = 0
latency_df["SRAM 4MB Worst Latency (s)"] = 0
latency_df["HZO3 4MB Latency (s)"] = 0
latency_df["HZO5 4MB Latency (s)"] = 0
# double buffering: take the max of the read and write latency for IO buffer
# the latency is determined by the slower operation
latency_df["SRAM 32MB Best Latency (s)"] = latency_df[["SRAM 32MB Best Read Latency (s)", "SRAM 32MB Best Write Latency (s)"]].max(axis=1)
latency_df["SRAM 32MB Worst Latency (s)"] = latency_df[["SRAM 32MB Worst Read Latency (s)", "SRAM 32MB Worst Write Latency (s)"]].max(axis=1)
latency_df["SRAM 4MB Best Latency (s)"] = latency_df["SRAM 4MB Best Read Latency (s)"]
latency_df["SRAM 4MB Worst Latency (s)"] = latency_df["SRAM 4MB Worst Read Latency (s)"]
latency_df["HZO3 4MB Latency (s)"] = latency_df["HZO3 4MB Read Latency (s)"]
latency_df["HZO5 4MB Latency (s)"] = latency_df["HZO5 4MB Read Latency (s)"]

# take the max of the read and write latency for IO buffer
latency_df["SRAM 32MB Best Static Energy per Inference (mJ)"] = latency_df["SRAM 32MB Best Latency (s)"] * leak_df["SRAM 32MB Best Static Power (mW)"] 
latency_df["SRAM 32MB Worst Static Energy per Inference (mJ)"] = latency_df["SRAM 32MB Worst Latency (s)"] * leak_df["SRAM 32MB Worst Static Power (mW)"]
latency_df["SRAM 4MB Best Static Energy per Inference (mJ)"] = latency_df["SRAM 4MB Best Latency (s)"] * leak_df["SRAM 4MB Best Static Power (mW)"]
latency_df["SRAM 4MB Worst Static Energy per Inference (mJ)"] = latency_df["SRAM 4MB Worst Latency (s)"] * leak_df["SRAM 4MB Worst Static Power (mW)"]
latency_df["HZO3 4MB Static Energy per Inference (mJ)"] = latency_df["HZO3 4MB Latency (s)"] * leak_df["HZO3 4MB Static Power (mW)"]
latency_df["HZO5 4MB Static Energy per Inference (mJ)"] = latency_df["HZO5 4MB Latency (s)"] * leak_df["HZO5 4MB Static Power (mW)"]

latency_df.to_csv("scripts/HZO_analysis_results/latency_and_static_energy_per_inference.csv", index=False)                           
print(leak_df)

# plot multiple bars for the best and worst case SRAM static energy per inference over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_static_power['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, latency_df['SRAM 32MB Best Static Energy per Inference (mJ)'], width, label='Best Case SRAM Static Energy per Inference', color='b')
bars2 = ax.bar(x + width/2, latency_df['SRAM 32MB Worst Static Energy per Inference (mJ)'], width, label='Worst Case SRAM Static Energy per Inference', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Static Energy per Inference (mJ)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_io_static_power['OPT target'], rotation=45)
ax.legend()

# plot multiple bars for the best and worst case SRAM static energy per inference and HZO static energy per inference over different OPT targets
fig_weight, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # the width of the bars
x = np.arange(len(df_sram_best_weight_static_power['OPT target']))  # the label locations
bars1 = ax.bar(x - width, latency_df['SRAM 4MB Best Static Energy per Inference (mJ)'], width, label='Best Case SRAM Static Energy per Inference', color='b')
bars2 = ax.bar(x, latency_df['HZO5 4MB Static Energy per Inference (mJ)'], width, label='HZO5 Static Energy per Inference', color='g')
bars3 = ax.bar(x + width, latency_df['HZO3 4MB Static Energy per Inference (mJ)'], width, label='HZO3 Static Energy per Inference', color='y')
bars4 = ax.bar(x + 2*width, latency_df['SRAM 4MB Worst Static Energy per Inference (mJ)'], width, label='Worst Case SRAM Static Energy per Inference', color='r')
# annotate bars with values
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)
for bar in bars3:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)
for bar in bars4:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Static Energy per Inference (mJ)')
ax.set_title('Weight buffer (4MB)')
ax.set_xticks(x)
ax.set_xticklabels(df_sram_best_weight_static_power['OPT target'], rotation=45)
ax.legend()

st.subheader("Static Energy per Inference")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)
col2.pyplot(fig_weight)


# # # =========================
# # total energy per inference
# # # =========================

# calculate the total energy per inference by adding the dynamic and static energy per inference 
# for each OPT target (SRAM)
# --- 1) Give unique names to each dynamic-energy column ---
df_sram_best_io_dynamic_energy  = df_sram_best_io_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "IO Best Dynamic (mJ)"})
df_sram_worst_io_dynamic_energy = df_sram_worst_io_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "IO Worst Dynamic (mJ)"})

df_sram_best_weight_dynamic_energy  = df_sram_best_weight_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "Weight SRAM-Best Dynamic (mJ)"})
df_sram_worst_weight_dynamic_energy = df_sram_worst_weight_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "Weight SRAM-Worst Dynamic (mJ)"})
df_hzo_3_weight_dynamic_energy      = df_hzo_3_weight_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "Weight HZO3 Dynamic (mJ)"})
df_hzo_5_weight_dynamic_energy      = df_hzo_5_weight_dynamic_energy.rename(
    columns={"Dynamic Energy per Inference (mJ)": "Weight HZO5 Dynamic (mJ)"})

# --- 2) Merge everything on OPT target without suffix fights ---
static_cols = [
    "SRAM 32MB Best Static Energy per Inference (mJ)",
    "SRAM 32MB Worst Static Energy per Inference (mJ)",
    "SRAM 4MB Best Static Energy per Inference (mJ)",
    "SRAM 4MB Worst Static Energy per Inference (mJ)",
    "HZO3 4MB Static Energy per Inference (mJ)",
    "HZO5 4MB Static Energy per Inference (mJ)",
]

total_energy_df = (
    df_sram_best_io_dynamic_energy
      .merge(df_sram_worst_io_dynamic_energy,  on="OPT target")
      .merge(df_sram_best_weight_dynamic_energy,  on="OPT target")
      .merge(df_sram_worst_weight_dynamic_energy, on="OPT target")
      .merge(df_hzo_3_weight_dynamic_energy,      on="OPT target")
      .merge(df_hzo_5_weight_dynamic_energy,      on="OPT target")
      .merge(latency_df[["OPT target"] + static_cols], on="OPT target")
      .sort_values("OPT target")
      .reset_index(drop=True)
)

# plot multiple bars for the best and worst case SRAM total energy per inference over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(total_energy_df['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, total_energy_df['IO Best Dynamic (mJ)'] + total_energy_df['SRAM 32MB Best Static Energy per Inference (mJ)'], width, label='Best Case SRAM Total Energy per Inference', color='b')
bars2 = ax.bar(x + width/2, total_energy_df['IO Worst Dynamic (mJ)'] + total_energy_df['SRAM 32MB Worst Static Energy per Inference (mJ)'], width, label='Worst Case SRAM Total Energy per Inference', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Total Energy per Inference (mJ)')
ax.set_title('IO buffer (2MB)')
ax.set_xticks(x)
ax.set_xticklabels(total_energy_df['OPT target'], rotation=45)
ax.legend()
# plot multiple bars for the best and worst case SRAM total energy per inference and HZO total energy per inference over different OPT targets
fig_weight, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # the width of the bars
x = np.arange(len(total_energy_df['OPT target']))  # the label locations
bars1 = ax.bar(x - width, total_energy_df['Weight SRAM-Best Dynamic (mJ)'] + total_energy_df['SRAM 4MB Best Static Energy per Inference (mJ)'], width, label='Best Case SRAM Total Energy per Inference', color='b')
bars2 = ax.bar(x, total_energy_df['Weight HZO5 Dynamic (mJ)'] + total_energy_df['HZO5 4MB Static Energy per Inference (mJ)'], width, label='HZO5 Total Energy per Inference', color='g')
bars3 = ax.bar(x + width, total_energy_df['Weight HZO3 Dynamic (mJ)'] + total_energy_df['HZO3 4MB Static Energy per Inference (mJ)'], width, label='HZO3 Total Energy per Inference', color='y')
bars4 = ax.bar(x + 2*width, total_energy_df['Weight SRAM-Worst Dynamic (mJ)'] + total_energy_df['SRAM 4MB Worst Static Energy per Inference (mJ)'], width, label='Worst Case SRAM Total Energy per Inference', color='r')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Total Energy per Inference (mJ)')
ax.set_title('Weight buffer (4MB)')
ax.set_xticks(x)
ax.set_xticklabels(total_energy_df['OPT target'], rotation=45)
ax.legend()
st.subheader("Total Energy per Inference")
col1, col2 = st.columns(2)
col1.pyplot(fig_io)
col2.pyplot(fig_weight)

def _num(s): return pd.to_numeric(s, errors="coerce")

# IO buffer (2MB): keep ONLY SRAM-Best total = Best Dynamic + 32MB Best Static
plotted_df_io = (
    pd.DataFrame({
        "OPT target": total_energy_df["OPT target"].astype(str).str.strip(),
        "IO SRAM-Best Total (mJ)": (
            _num(total_energy_df["IO Best Dynamic (mJ)"]) +
            _num(total_energy_df["SRAM 32MB Best Static Energy per Inference (mJ)"])
        ),
    })
    .sort_values("OPT target")
    .reset_index(drop=True)
)

# Weight buffer (4MB): keep SRAM-Best + HZO5 + HZO3 totals; DROP SRAM-Worst
plotted_df_weight = (
    pd.DataFrame({
        "OPT target": total_energy_df["OPT target"].astype(str).str.strip(),
        "Weight SRAM-Best Total (mJ)": (
            _num(total_energy_df["Weight SRAM-Best Dynamic (mJ)"]) +
            _num(total_energy_df["SRAM 4MB Best Static Energy per Inference (mJ)"])
        ),
        "Weight HZO5 Total (mJ)": (
            _num(total_energy_df["Weight HZO5 Dynamic (mJ)"]) +
            _num(total_energy_df["HZO5 4MB Static Energy per Inference (mJ)"])
        ),
        "Weight HZO3 Total (mJ)": (
            _num(total_energy_df["Weight HZO3 Dynamic (mJ)"]) +
            _num(total_energy_df["HZO3 4MB Static Energy per Inference (mJ)"])
        ),
    })
    .sort_values("OPT target")
    .reset_index(drop=True)
)

# mix and match the IO buffer under different OPT targets with different weight buffer technologies 
# under the same OPT target
# make the first column the OPT target of the IO buffer
# make the second column the OPT target of the weight buffer
# make the third column the wright buffer technology (SRAM, HZO3, HZO5)
# make the fourth column the total energy per inference of the IO buffer and weight buffer

# --- Ensure _num helper exists ---
if "_num" not in locals():
    def _num(s): return pd.to_numeric(s, errors="coerce")

# 1) IO totals (SRAM-Best only)
plotted_df_io = (
    pd.DataFrame({
        "OPT target": total_energy_df["OPT target"].astype(str).str.strip(),
        "IO Total (mJ)": (
            _num(total_energy_df["IO Best Dynamic (mJ)"]) +
            _num(total_energy_df["SRAM 32MB Best Static Energy per Inference (mJ)"])
        ),
    })
    .sort_values("OPT target")
    .reset_index(drop=True)
    .rename(columns={"OPT target": "IO OPT target"})
)

# 2) Weight totals (SRAM-Best, HZO5, HZO3), then melt to long
plotted_df_weight = (
    pd.DataFrame({
        "OPT target": total_energy_df["OPT target"].astype(str).str.strip(),
        "SRAM-Best": (
            _num(total_energy_df["Weight SRAM-Best Dynamic (mJ)"]) +
            _num(total_energy_df["SRAM 4MB Best Static Energy per Inference (mJ)"])
        ),
        "HZO5": (
            _num(total_energy_df["Weight HZO5 Dynamic (mJ)"]) +
            _num(total_energy_df["HZO5 4MB Static Energy per Inference (mJ)"])
        ),
        "HZO3": (
            _num(total_energy_df["Weight HZO3 Dynamic (mJ)"]) +
            _num(total_energy_df["HZO3 4MB Static Energy per Inference (mJ)"])
        ),
    })
    .sort_values("OPT target")
    .reset_index(drop=True)
    .rename(columns={"OPT target": "Weight OPT target"})
)

weight_long = (
    plotted_df_weight
      .melt(id_vars="Weight OPT target", var_name="Weight Tech", value_name="Weight Total (mJ)")
)

# 3) Cartesian product (cross join) IO × Weight
# Pandas ≥ 1.2:
mix_energy_df = plotted_df_io.merge(weight_long, how="cross")

# If your pandas is older than 1.2, use this instead:
# mix_energy_df = (plotted_df_io.assign(_k=1)
#                  .merge(weight_long.assign(_k=1), on="_k")
#                  .drop(columns="_k"))

# 4) Sum IO + Weight totals
mix_energy_df["Total Energy per Inference (mJ)"] = (
    _num(mix_energy_df["IO Total (mJ)"]) + _num(mix_energy_df["Weight Total (mJ)"])
)

# 5) Final 4 columns in requested order
mix_energy_df = (
    mix_energy_df[["IO OPT target", "Weight OPT target", "Weight Tech", "Total Energy per Inference (mJ)"]]
    .sort_values(["IO OPT target", "Weight OPT target", "Weight Tech"])
    .reset_index(drop=True)
)

# (Optional) Streamlit view
# st.subheader("Cartesian Mix & Match: IO (SRAM-Best) × Weight (SRAM-Best/HZO5/HZO3)")
# st.dataframe(mix_energy_df, use_container_width=True)


mix_energy_df.to_csv("scripts/HZO_analysis_results/totalEnergyPerInference_mix_and_match.csv", index=False)

# # =========================
# plot overall carbon emission
# # =========================

grid_CI = 380  # in gCO2/kWh
inference_frequency = 0.1  # in Hz, i.e., 5 inferences per second
deployment_duration_years = 3  # in years
seconds_per_year = 365 * 24 * 3600  # in seconds
total_inferences = inference_frequency * deployment_duration_years * seconds_per_year  # total inferences over the deployment duration
# total energy in kWh
mix_energy_df["Total Energy (kWh)"] = mix_energy_df["Total Energy per Inference (mJ)"] * total_inferences * 2.77778e-10
# total carbon emission in gCO2
mix_energy_df["Total Operational Carbon Emission (gCO2)"] = mix_energy_df["Total Energy (kWh)"] * grid_CI

print(mix_energy_df)

# plot mix energy df
# x axis are combinations of IO OPT target, weight OPT target, and weight tech
# y axis is total operational carbon emission
fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(mix_energy_df))
bars = ax.bar(x, mix_energy_df["Total Operational Carbon Emission (gCO2)"], color='b')
ax.set_ylabel('Total Operational Carbon Emission (gCO2)')
ax.set_title('Total Operational Carbon Emission for Different OPT Target Combinations and Weight Technologies')
ax.set_xticks(x)
ax.set_xticklabels(mix_energy_df.apply(lambda row: f"IO:{row['IO OPT target']}-W:{row['Weight OPT target']}-{row['Weight Tech']}", axis=1), rotation=90, fontsize=8)
plt.tight_layout()
st.subheader("Total Operational Carbon Emission")   
st.pyplot(fig)

# import the embodied carbon data
embodied_carbon_df = pd.read_csv("scripts/HZO_analysis_results/area_mix_and_match_totalcarbon.csv")
# plot the embodied carbon data
fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(embodied_carbon_df))
bars = ax.bar(x, embodied_carbon_df["total_carbon_g"], color='g')
ax.set_ylabel('Embodied Carbon (gCO2)')
ax.set_title('Embodied Carbon for Different OPT Target Combinations and Weight Technologies')
ax.set_xticks(x)
ax.set_xticklabels(embodied_carbon_df.apply(lambda row: f"IO:{row['IO buffer OPT target']}-W:{row['weight buffer opt target']}-{row['weight buffer tech']}", axis=1), rotation=90, fontsize=8)
plt.tight_layout()
st.subheader("Embodied Carbon")
st.pyplot(fig)

# plot total carbon emission stacked
total_carbon_df = mix_energy_df.merge(embodied_carbon_df, left_on=["IO OPT target", "Weight OPT target", "Weight Tech"], right_on=["IO buffer OPT target", "weight buffer opt target", "weight buffer tech"])
total_carbon_df["Overall Carbon Emission (gCO2)"] = total_carbon_df["Total Operational Carbon Emission (gCO2)"] + total_carbon_df["total_carbon_g"]
fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(total_carbon_df))
bars1 = ax.bar(x, total_carbon_df["Total Operational Carbon Emission (gCO2)"], label='Operational Carbon Emission', color='b')
bars2 = ax.bar(x, total_carbon_df["total_carbon_g"], bottom=total_carbon_df["Total Operational Carbon Emission (gCO2)"], label='Embodied Carbon', color='g')
ax.set_ylabel('Overall Carbon Emission (gCO2)')
ax.set_xticks(x)
ax.set_xticklabels(total_carbon_df.apply(lambda row: f"IO:{row['IO OPT target']}-W:{row['Weight OPT target']}-{row['Weight Tech']}", axis=1), rotation=90, fontsize=8)
plt.tight_layout()
st.subheader("Overall Carbon Emission")
st.pyplot(fig)

# report the minimum overall carbon emission and the corresponding OPT target combination and weight technology
min_overall_carbon = total_carbon_df["Overall Carbon Emission (gCO2)"].min()
min_overall_carbon_row = total_carbon_df[total_carbon_df["Overall Carbon Emission (gCO2)"] == min_overall_carbon]
st.subheader("Minimum Overall Carbon Emission")
st.write(f"The minimum overall carbon emission is {min_overall_carbon:.2f} gCO2, achieved with the following configuration:")
st.dataframe(min_overall_carbon_row[["IO OPT target", "Weight OPT target", "Weight Tech", "Overall Carbon Emission (gCO2)"]], use_container_width=True)

# plot the total energy per inference breakdown 
# in terms of dynamic read, dynamic write, and static energy
# for mix and match between IO buffer (SRAM-Best only) with OPT targets
# of area, readDynamicEnergy, ReadEDP
# and weight buffer (SRAM-Best, HZO5, HZO3) with OPT targets
# of area, readDynamicEnergy, ReadEDP
# and put the start on the two minimum overall carbon emission configurations

# min_configs = min_overall_carbon_row[["IO OPT target", "Weight OPT target", "Weight Tech"]].values.tolist()
# min_configs = [tuple(x) for x in min_configs]
# fig, ax = plt.subplots(figsize=(18, 10))
# width = 0.2  # the width of the bars
# x = np.arange(len(mix_energy_df))  # the label locations
# bars1 = ax.bar(x - width, mix_energy_df["IO Total (mJ)"], width, label='IO Total Energy per Inference (mJ)', color='b')
# bars2 = ax.bar(x, mix_energy_df["Weight Total (mJ)"], width, label='Weight Total Energy per Inference (mJ)', color='g')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Total Energy per Inference (mJ)')
# ax.set_title('Total Energy per Inference Breakdown for Different OPT Target Combinations and Weight Technologies')
# ax.set_xticks(x)
# ax.set_xticklabels(mix_energy_df.apply(lambda row: f"IO:{row['IO OPT target']}-W:{row['Weight OPT target']}-{row['Weight Tech']}", axis=1), rotation=90, fontsize=8)
# # highlight the two minimum overall carbon emission configurations
# for i, row in mix_energy_df.iterrows():
#     if (row["IO OPT target"], row["Weight OPT target"], row["Weight Tech"]) in min_configs:
#         ax.get_children()[i*2].set_edgecolor('r')
#         ax.get_children()[i*2].set_linewidth(2)
#         ax.get_children()[i*2+1].set_edgecolor('r')
#         ax.get_children()[i*2+1].set_linewidth(2)
# ax.legend()
# plt.tight_layout()
# st.subheader("Total Energy per Inference Breakdown")
# st.pyplot(fig)

# =========================
# Total energy breakdown (stacked) + highlight top-2 lowest-carbon combos
# =========================

# --- IO per-OPT breakdown (SRAM-Best only) ---
io_break = (
    df_sram_best_io_dynamic_Read[["OPT target", "SRAM 32MB Dynamic Read Energy (pJ)"]]
    .merge(df_sram_best_io_dynamic_Write[["OPT target", "SRAM 32MB Dynamic Write Energy (pJ)"]], on="OPT target")
    .merge(latency_df[["OPT target", "SRAM 32MB Best Static Energy per Inference (mJ)"]], on="OPT target")
    .rename(columns={"OPT target": "IO OPT target"})
)
io_break["IO Dyn Read (mJ)"]   = io_break["SRAM 32MB Dynamic Read Energy (pJ)"]  * input_reads   * 1e-6
io_break["IO Dyn Write (mJ)"]  = io_break["SRAM 32MB Dynamic Write Energy (pJ)"] * output_writes * 1e-6
io_break["IO Static (mJ)"]     = io_break["SRAM 32MB Best Static Energy per Inference (mJ)"]
io_break = io_break[["IO OPT target", "IO Dyn Read (mJ)", "IO Dyn Write (mJ)", "IO Static (mJ)"]].copy()
io_break["IO OPT target"] = io_break["IO OPT target"].astype(str).str.strip()

# --- Weight per-OPT breakdown for each tech ---
def weight_breakdown(read_df, static_col_name):
    out = (
        read_df[["OPT target", read_df.columns[-1]]]  # last col is the renamed "Dynamic Read Energy (pJ)"
        .merge(latency_df[["OPT target", static_col_name]], on="OPT target")
        .rename(columns={"OPT target": "Weight OPT target",
                         read_df.columns[-1]: "Read Energy (pJ)",
                         static_col_name: "Weight Static (mJ)"})
    )
    out["Weight Dyn Read (mJ)"] = out["Read Energy (pJ)"] * weight_reads * 1e-6
    out = out[["Weight OPT target", "Weight Dyn Read (mJ)", "Weight Static (mJ)"]].copy()
    out["Weight OPT target"] = out["Weight OPT target"].astype(str).str.strip()
    return out

wb_sram  = weight_breakdown(df_sram_best_weight_dynamic_Read, "SRAM 4MB Best Static Energy per Inference (mJ)")
wb_hzo5  = weight_breakdown(df_hzo_5_weight_dynamic_Read,    "HZO5 4MB Static Energy per Inference (mJ)")
wb_hzo3  = weight_breakdown(df_hzo_3_weight_dynamic_Read,    "HZO3 4MB Static Energy per Inference (mJ)")

wb_sram["Weight Tech"] = "SRAM-Best"
wb_hzo5["Weight Tech"] = "HZO5"
wb_hzo3["Weight Tech"] = "HZO3"

weight_break_all = pd.concat([wb_sram, wb_hzo5, wb_hzo3], ignore_index=True)

# --- Cartesian product IO × Weight, then build component totals ---
breakdown = io_break.merge(weight_break_all, how="cross")

# Order & labeling
breakdown = breakdown[[
    "IO OPT target", "Weight OPT target", "Weight Tech",
    "IO Dyn Read (mJ)", "IO Dyn Write (mJ)", "IO Static (mJ)",
    "Weight Dyn Read (mJ)", "Weight Static (mJ)"
]].sort_values(["IO OPT target", "Weight OPT target", "Weight Tech"], ignore_index=True)

# Add total energy per inference (mJ) to cross-check with your previous mix_energy_df
breakdown["Total Energy per Inference (mJ)"] = (
    breakdown[["IO Dyn Read (mJ)", "IO Dyn Write (mJ)", "IO Static (mJ)",
               "Weight Dyn Read (mJ)", "Weight Static (mJ)"]].sum(axis=1)
)

# Bring in Overall Carbon Emission to find the top-2 minimum
# Ensure keys match the ones used in total_carbon_df
bk_for_join = breakdown.merge(
    total_carbon_df[[
        "IO OPT target", "Weight OPT target", "Weight Tech", "Overall Carbon Emission (gCO2)"
    ]],
    on=["IO OPT target", "Weight OPT target", "Weight Tech"],
    how="left"
)

# Indices of the two minimum overall carbon configurations
top2_idx = bk_for_join["Overall Carbon Emission (gCO2)"].nsmallest(2).index.tolist()

# --- Plot: stacked energy components per combo ---
labels = bk_for_join.apply(
    lambda r: f"IO:{r['IO OPT target']}-W:{r['Weight OPT target']}-{r['Weight Tech']}", axis=1
)

fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(bk_for_join))

# Stacks
b1 = ax.bar(x, bk_for_join["IO Dyn Read (mJ)"], label="IO Dyn Read (mJ)")
b2 = ax.bar(x, bk_for_join["IO Dyn Write (mJ)"], bottom=bk_for_join["IO Dyn Read (mJ)"], label="IO Dyn Write (mJ)")
b3 = ax.bar(x,
            bk_for_join["IO Static (mJ)"],
            bottom=bk_for_join["IO Dyn Read (mJ)"] + bk_for_join["IO Dyn Write (mJ)"],
            label="IO Static (mJ)")
b4 = ax.bar(x,
            bk_for_join["Weight Dyn Read (mJ)"],
            bottom=bk_for_join["IO Dyn Read (mJ)"] + bk_for_join["IO Dyn Write (mJ)"] + bk_for_join["IO Static (mJ)"],
            label="Weight Dyn Read (mJ)")
b5 = ax.bar(x,
            bk_for_join["Weight Static (mJ)"],
            bottom=bk_for_join["IO Dyn Read (mJ)"] + bk_for_join["IO Dyn Write (mJ)"] + bk_for_join["IO Static (mJ)"] + bk_for_join["Weight Dyn Read (mJ)"],
            label="Weight Static (mJ)")

ax.set_ylabel("Total Energy per Inference (mJ)")
ax.set_title("Energy Breakdown per Inference by IO/Weight OPT Target and Tech")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.legend(ncols=3, fontsize=9)
plt.tight_layout()

# Add ⭐ above the two minimum-carbon bars
y_tops = (
    bk_for_join["IO Dyn Read (mJ)"]
    + bk_for_join["IO Dyn Write (mJ)"]
    + bk_for_join["IO Static (mJ)"]
    + bk_for_join["Weight Dyn Read (mJ)"]
    + bk_for_join["Weight Static (mJ)"]
)
for idx in top2_idx:
    ax.text(x[idx], y_tops.iloc[idx] * 1.02, "★", ha="center", va="bottom", fontsize=16)

st.subheader("Total Energy per Inference – Breakdown (with ⭐ on 2 lowest overall-carbon configs)")
st.pyplot(fig)

# export the breakdown dataframe
# breakdown.to_csv("scripts/HZO_analysis_results/totalEnergyPerInference_breakdown_mix_and_match.csv", index=False)

bk_for_join.to_csv("scripts/HZO_analysis_results/totalEnergyPerInference_breakdown.csv", index=False)







