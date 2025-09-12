import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =========================
# Constants / Parameters
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
embodied_carbon_file = os.path.join(current_dir, 'HZO_analysis_results/area_mix_and_match_breakdown_with_carbon.csv')
operational_energy_file = os.path.join(current_dir, 'HZO_analysis_results/totalEnergyPerInference_breakdown.csv')
embodied_carbon_df = pd.read_csv(embodied_carbon_file)
operational_energy_df = pd.read_csv(operational_energy_file)
# form a df with IO buffer OPT target = WriteEDP
# weight buffer OPT target = area, readDynamicEnergy, ReadEDP
# Weight buffer Tech = SRAM-Best, HZO5
filtered_df = embodied_carbon_df[
    (embodied_carbon_df['IO buffer OPT target'] == 'WriteEDP') &
    (embodied_carbon_df['Weight buffer OPT target'].isin(['Area', 'ReadDynamicEnergy', 'ReadEDP'])) &
    (embodied_carbon_df['Weight buffer Tech'].isin(['SRAM-Best', 'HZO5']))
]
filtered_operational_energy_df = operational_energy_df.loc[
    (operational_energy_df['IO OPT target'] == 'WriteEDP') &
    (operational_energy_df['Weight OPT target'].isin(['Area', 'ReadDynamicEnergy', 'ReadEDP'])) &
    (operational_energy_df['Weight Tech'].isin(['SRAM-Best', 'HZO5']))
].copy()  

# change SRAM-Best to SRAM
filtered_df['Weight buffer Tech'] = filtered_df['Weight buffer Tech'].replace({'SRAM-Best': 'SRAM'})
# print(filtered_df)
filtered_operational_energy_df['Weight Tech'] = filtered_operational_energy_df['Weight Tech'].replace({'SRAM-Best': 'SRAM'})
print(filtered_operational_energy_df)

# carbon intensity in g CO2e per kWh
carbon_intensity = 380  # g CO2e/kWh

# replace your four assignments with .loc on the copy (either works once you've .copy()'d)
k = carbon_intensity * 2.7777777777778e-10

filtered_operational_energy_df.loc[:, 'io buffer Dyn Read']   = filtered_operational_energy_df['IO Dyn Read (mJ)']   * k
filtered_operational_energy_df.loc[:, 'io buffer Dyn Write']   = filtered_operational_energy_df['IO Dyn Write (mJ)']   * k
filtered_operational_energy_df.loc[:, 'io buffer static']      = filtered_operational_energy_df['IO Static (mJ)']      * k
filtered_operational_energy_df.loc[:, 'weight buffer Dyn Read']= filtered_operational_energy_df['Weight Dyn Read (mJ)']* k
filtered_operational_energy_df.loc[:, 'weight buffer static']  = filtered_operational_energy_df['Weight Static (mJ)']  * k

# =========================
total_mac_op = 520400529 # MAC operations
systolic_array_per_PE_energy_200MHz = 0.2296 # pJ
systolic_array_per_PE_power_200MHz = systolic_array_per_PE_energy_200MHz / (1/200e6) 
systolic_array_per_PE_power_20MHz = systolic_array_per_PE_power_200MHz / 10 
systolic_array_total_energy_pJ = systolic_array_per_PE_power_20MHz * (1/20e6) * total_mac_op # pJ
# systolic_array_total_energy_pJ = systolic_array_per_PE_power_20MHz * (1/20e6) * 32*32*1433453
systolic_array_total_energy_kwh = systolic_array_total_energy_pJ * 2.7777777777778e-19 # kWh
systolic_array_carbon_per_inference = systolic_array_total_energy_kwh * carbon_intensity # g CO2e
print(f"Systolic array carbon per inference: {systolic_array_total_energy_kwh:.15f} g CO2e")
filtered_operational_energy_df.loc[:, 'systolic array'] = systolic_array_carbon_per_inference

# ---- Assumes you already built filtered_df exactly as in your snippet ----
# Columns present:
# 'Weight buffer OPT target', 'Weight buffer Tech',
# 'io buffer carbon (g)', 'weight buffer carbon (g)', 'systolic array carbon (g)', 'total carbon (g)'

# Keep only the fields we need and enforce order
embodied_component_cols = [
    'io buffer carbon (g)',
    'systolic array carbon (g)',
    'weight buffer carbon (g)',
]

operation_component_cols = [
    'io buffer Dyn Read',
    'io buffer Dyn Write',
    'io buffer static',
    'weight buffer Dyn Read',
    'weight buffer static',
    'systolic array'
]

# Order the x-ax[0]is (OPT targets)
opt_order = ['Area', 'ReadDynamicEnergy', 'ReadEDP']
tech_order = ['HZO5', 'SRAM']

plot_df = (
    filtered_df
    .loc[filtered_df['Weight buffer OPT target'].isin(opt_order)
         & filtered_df['Weight buffer Tech'].isin(tech_order),
         ['Weight buffer OPT target', 'Weight buffer Tech'] + embodied_component_cols + ['total carbon (g)']]
    .copy()
)

operational_plot_df = (
    filtered_operational_energy_df
    .loc[filtered_operational_energy_df['Weight OPT target'].isin(opt_order)
         & filtered_operational_energy_df['Weight Tech'].isin(tech_order),
         ['Weight OPT target', 'Weight Tech'] + operation_component_cols]
    .copy()
)

# Categorical ordering
plot_df['Weight buffer OPT target'] = pd.Categorical(plot_df['Weight buffer OPT target'], categories=opt_order, ordered=True)
plot_df['Weight buffer Tech'] = pd.Categorical(plot_df['Weight buffer Tech'], categories=tech_order, ordered=True)
operational_plot_df['Weight OPT target'] = pd.Categorical(operational_plot_df['Weight OPT target'], categories=opt_order, ordered=True)
operational_plot_df['Weight Tech'] = pd.Categorical(operational_plot_df['Weight Tech'], categories=tech_order, ordered=True)

# Sort for plotting: by OPT target, then Tech
plot_df = plot_df.sort_values(['Weight buffer OPT target', 'Weight buffer Tech'])
operational_plot_df = operational_plot_df.sort_values(['Weight OPT target', 'Weight Tech'])

# Build positions: for each OPT target, we place two bars (HZO5, SRAM-Best)
x_ticks = opt_order
x = np.arange(len(x_ticks))
bar_width = 0.35
offsets = {
    'HZO5': -bar_width/2,
    'SRAM': +bar_width/2,
}

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# To create a consistent color mapping for stacks, just let matplotlib pick default colors.
# We'll add hatch patterns to distinguish Tech.
hatch_for_tech = {
    'HZO5': '///',
    'SRAM': '',
}
component_colors = {
    'io buffer carbon (g)': '#1f77b4',  # blue
    'weight buffer carbon (g)': '#ff7f0e', # orange
    'systolic array carbon (g)': '#2ca02c', # green
}

# Keep handles to one set of stacks for component legend
component_handles = []
component_labels = ['IO buffer', 'Systolic array', 'Weight buffer',]

# Plot bars
for tech in tech_order:
    sub = plot_df[plot_df['Weight buffer Tech'] == tech]
    # Ensure order by OPT target
    sub = sub.set_index('Weight buffer OPT target').loc[opt_order]

    base = np.zeros(len(opt_order))
    handles_this_bar = []
    for i, comp in enumerate(embodied_component_cols):
        h = ax[0].bar(
            x + offsets[tech],
            sub[comp].values,
            width=bar_width,
            bottom=base,
            label=component_labels[i] if tech == tech_order[0] else None,  # add component legend once
            hatch=hatch_for_tech[tech],
            edgecolor='black', linewidth=0.6,
            color=component_colors[comp],
        )
        if tech == tech_order[0]:
            component_handles.append(h[0])
        base += sub[comp].values
        handles_this_bar.append(h)

    # # Annotate total on top of each bar
    # totals = sub['total carbon (g)'].values if 'total carbon (g)' in sub.columns else base
    # for xi, total in zip(x + offsets[tech], totals):
    #     ax[0].text(xi, total + max(totals)*0.01, f"{total:.1f}", ha='center', va='bottom', fontsize=9)


# ax[0]es, ticks, labels
ax[0].set_xticks(x)
ax[0].set_xticklabels(x_ticks, rotation=0)
ax[0].set_ylabel('Embodied carbon (g CO₂e)')
ax[0].set_ylim(0, 150)

# Legends:
# 1) Component legend (colors)
# leg1 = ax[0].legend(component_handles, component_labels, title='Components', loc='upper left', bbox_to_anchor=(1.02, 1.0))

# # 2) Tech legend (hatches) — create proxy artists
tech_patches = [Patch(facecolor='white', edgecolor='black', hatch=hatch_for_tech[t], label=t) for t in tech_order]
# leg2 = ax[0].legend(handles=tech_patches, title='Weight buffer\nTechnology', loc='upper left', bbox_to_anchor=(1.02, 0.6))

# ax[0].add_artist(leg1)  # ensure both legends show

# ---- Operational energy plot ----
# Keep handles to one set of stacks for component legend
component_handles_op = []
component_labels_op = ['IO buffer Dyn Read', 'IO buffer Dyn Write', 'IO buffer Static', 'Weight buffer Dyn Read', 'Weight buffer Static', 'Systolic array']
operation_component_colors = {
    'io buffer Dyn Read':   '#17becf',  # cyan
    'io buffer Dyn Write':  '#bcbd22',  # olive
    'io buffer static':     '#8c564b',  # brown
    'weight buffer Dyn Read':'#e377c2', # pink
    'weight buffer static': '#7f7f7f',  # gray
    'systolic array':   '#d62728', 
}
# Plot bars
for tech in tech_order:
    sub = operational_plot_df[operational_plot_df['Weight Tech'] == tech]
    # Ensure order by OPT target
    sub = sub.set_index('Weight OPT target').loc[opt_order]

    base = np.zeros(len(opt_order))
    handles_this_bar = []
    for i, comp in enumerate(operation_component_cols):
        h = ax[1].bar(
            x + offsets[tech],
            sub[comp].values * carbon_intensity / 1000,  # convert mJ to kWh and multiply by carbon intensity
            width=bar_width,
            bottom=base,
            label=component_labels_op[i] if tech == tech_order[0] else None,  # add component legend once
            hatch=hatch_for_tech[tech],
            edgecolor='black', linewidth=0.6,
            color=operation_component_colors[comp],
        )
        if tech == tech_order[0]:
            component_handles_op.append(h[0])
        base += sub[comp].values * carbon_intensity / 1000
        handles_this_bar.append(h)

    # # Annotate total on top of each bar
    # totals = base
    # for xi, total in zip(x + offsets[tech], totals):
    #     ax[1].text(xi, total + max(totals)*0.01, f"{total:.2f}", ha='center', va='bottom', fontsize=9)
# ax[1]es, ticks, labels
ax[1].set_xticks(x)
ax[1].set_xticklabels(x_ticks, rotation=0)
ax[1].set_ylabel('Operational carbon per inference (g CO₂e)')
# Legends:
# leg1_op = ax[1].legend(component_handles_op, component_labels_op, title='Components', loc='upper right', bbox_to_anchor=(0.5, 1.0))
# leg2_op = ax[1].legend(handles=tech_patches, title='Weight buffer\nTechnology', loc='upper left', bbox_to_anchor=(1.02, 0.6))
# ax[1].add_artist(leg1_op)  # ensure both legends show

# keep component legend inside subplot
leg1 = ax[0].legend(component_handles, component_labels, loc='upper right')
ax[0].add_artist(leg1)

leg1_op = ax[1].legend(component_handles_op, component_labels_op, loc='upper right')

# move tech legend to the top of the figure, centered
tech_patches = [
    Patch(facecolor='white', edgecolor='black', hatch='///', label='HZO5'),
    Patch(facecolor='white', edgecolor='black', label='SRAM')
]
fig.legend(handles=tech_patches,
           loc='upper center', bbox_to_anchor=(0.3, 1.02), ncol=2)


plt.tight_layout()
fig.text(0.55, 0.005, 'Weight buffer OPT target', ha='center', va='bottom', fontsize=10) 

# Save the plot
output_file = os.path.join(current_dir, 'HZO_analysis_results/embodied_carbon_breakdown_plot.pdf')
plt.savefig(output_file)






