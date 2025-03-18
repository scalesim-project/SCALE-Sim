import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import numpy as np
# Data
benchmarks = [r'64', '128', '256', '512', '1024']
# designs = ['#BW=64', '#BW=128', '#BW=256', '#BW=512', '#BW=1024']
designs = ['#Bank=1', '#Bank=2', '#Bank=4', '#Bank=8', '#Bank=16']

slowdown_ratio=[[1.0140, 1.0104, 1.0087, 1.0078, 1.0073],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

patterns = [ "//" , "\\\\" , "oo" , "*" , "o-" , "xx", "o", "O", "", "." ]

shape_list = ["o", "*", "X", "p", ">", "d", "s", "H", "<"]
legend_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
color_list = ["#D3D3D3", "#A9A9A9", "#808080", "#696969", "#404040"]
slowdown_ratio = np.array(slowdown_ratio).transpose()-1
# slowdown_ratio = np.log(slowdown_ratio)
# Settings for the plot
x = np.arange(len(benchmarks))  # The label locations
width = 0.1  # The width of the bars

# Create the figure and axes
fig = plt.figure(figsize=[15, 5])
gs = gridspec.GridSpec(1, 3) 
ax0 = plt.subplot(gs[0])


# Plot each design's latency overhead on the bar chart
plt_handler = []
for i in range(len(designs)):
    plt_handler.append(ax0.bar(x - (len(slowdown_ratio)//2)*width + i * width, slowdown_ratio[i], width, label=designs[i], color=color_list[i], edgecolor="black")[0])

# Add labels, title, and customize ticks
# ax0.set_xlabel('# Bank')
ax0.set_ylabel('Slowdown vs BW model')
ax0.set_xlabel('Input Stationary')
# ax0.set_yticks([0, 4, 8, 12, 14])  # Adjust ticks to center them under groups
ax0.set_xticks([0, 1, 2, 3, 4]
)  # Adjust ticks to center them under groups
ax0.set_xticklabels(benchmarks)
# ax0.set_xticklabels(benchmarks)
ax0.set_yticks([-0.005, 0, 0.005, 0.01, 0.015])  # Adjust ticks to center them under groups
for ytick in ax0.get_yticks():
    ax0.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

###########
ax1 = plt.subplot(gs[1])

slowdown_ratio=[[0.7597, 0.7550, 0.7526, 0.7514, 0.7508],
[0.8224, 0.8224, 0.8224, 0.8224, 0.8224],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
slowdown_ratio = np.array(slowdown_ratio).transpose()-1

# Plot each design's latency overhead on the bar chart
plt_handler = []
for i in range(len(designs)):
    plt_handler.append(ax1.bar(x - (len(slowdown_ratio)//2)*width + i * width, slowdown_ratio[i], width, label=designs[i], color=color_list[i], edgecolor="black")[0])
plt.legend(plt_handler, designs, loc='best', ncol=2, fontsize=14, labelspacing=0, columnspacing=0.1)

# Add labels, title, and customize ticks
ax1.set_xticks([0, 1, 2, 3, 4]
)  # Adjust ticks to center them under groups
ax1.set_xlabel('Weights Stationary')
ax1.set_xticklabels(benchmarks)
ax1.set_yticks([-0.3, -0.2, -0.1, 0, 0.1])  # Adjust ticks to center them under groups
for ytick in ax1.get_yticks():
    ax1.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5)
# ax1.set_xticklabels(benchmarks)
# ax1.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])  # Adjust ticks to center them under groups
# ax.set_yticks([0, 4, 8, 12, 14])  # Adjust ticks to center them under groups
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # leg
# plt.setp(ax1.get_yticklabels(), visible=False)

##########
ax2 = plt.subplot(gs[2])
slowdown_ratio=[[1.0355, 1.0303, 1.0283, 1.0256, 1.0250],
[1.0124, 1.0124, 1.0124, 1.0124, 1.0124],
[1.0127, 1.0127, 1.0127, 1.0127, 1.0127],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
slowdown_ratio = np.array(slowdown_ratio).transpose()-1

# Plot each design's latency overhead on the bar chart
for i in range(len(designs)):
    ax2.bar(x - (len(slowdown_ratio)//2)*width + i * width, slowdown_ratio[i], width, label=designs[i], color=color_list[i], edgecolor="black")

# Add labels, title, and customize ticks
ax2.set_xticks([0, 1, 2, 3, 4])  # Adjust ticks to center them under groups
ax2.set_xlabel('Output Stationary')
ax2.set_yticks([-0.005, 0,  0.01, 0.02, 0.03, 0.04])  # Adjust ticks to center them under groups
ax2.set_xticklabels(benchmarks)
for ytick in ax2.get_yticks():
    ax2.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # leg
# plt.setp(ax2.get_yticklabels(), visible=False)

# Display the chart
# plt.subplots_adjust(hspace=1)
plt.subplots_adjust(wspace=0.23)
# plt.tight_layout()
plt.savefig('layout_plots/multi_bank_vs_bandwidth_model_vit.pdf', bbox_inches="tight", transparent=True) 
plt.show()
