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
HZO_data_file = os.path.join(current_dir, 'HZO_data/combined_HZO3&HZO5_2MB&4MB.csv')
SRAM_data_file = os.path.join(current_dir, 'HZO_data/combined_SRAM_2&4&32MB.csv')

# extract 4MB HZO
hzo_df = pd.read_csv(HZO_data_file)
hzo_df['OPT target'] = hzo_df['Source'].str.split('_').str[2]
# sort OPT target in an alphabetical order
hzo_df['OPT target'] = hzo_df['OPT target'].astype(str).str.strip()
hzo_df = hzo_df.sort_values(by=['OPT target'])
hzo_3_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_15_3_9.cell']
hzo_5_df = hzo_df[hzo_df['MemoryCellInputFile'] == ' data/cell_cfgs/FeFET_HZO_15_5_9.cell']
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

df_sram_best_io_area = area_prep(sram_best_io_df, "SRAM 32MB Area (mm^2)")
df_sram_worst_io_area = area_prep(sram_worst_io_df, "SRAM 32MB Area (mm^2)")

df_sram_best_weight_area = area_prep(sram_best_weight_df, "SRAM 4MB Area (mm^2)")
df_sram_worst_weight_area = area_prep(sram_worst_weight_df, "SRAM 4MB Area (mm^2)")
df_hzo_3_weight_area = area_prep(hzo_3_weight_df, "HZO3 4MB Area (mm^2)")
df_hzo_5_weight_area = area_prep(hzo_5_weight_df, "HZO5 4MB Area (mm^2)")

print(df_sram_best_io_area)

# plot multiple bars for the best and worst case SRAM area over different OPT targets
fig_io, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # the width of the bars
x = np.arange(len(df_sram_best_io_area['OPT target']))  # the label locations
bars1 = ax.bar(x - width/2, df_sram_best_io_area['SRAM 32MB Area (mm^2)'], width, label='Best Case SRAM Area', color='b')
bars2 = ax.bar(x + width/2, df_sram_worst_io_area['SRAM 32MB Area (mm^2)'], width, label='Worst Case SRAM Area', color='r')
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
bars1 = ax.bar(x - width, df_sram_best_weight_area['SRAM 4MB Area (mm^2)'], width, label='Best Case SRAM Area', color='b')
bars2 = ax.bar(x, df_hzo_5_weight_area['HZO5 4MB Area (mm^2)'], width, label='HZO5 Area', color='g')
bars3 = ax.bar(x + width, df_hzo_3_weight_area['HZO3 4MB Area (mm^2)'], width, label='HZO3 Area', color='y')
bars4 = ax.bar(x + 2*width, df_sram_worst_weight_area['SRAM 4MB Area (mm^2)'], width, label='Worst Case SRAM Area', color='r')
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
print(dyRead_df)

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








