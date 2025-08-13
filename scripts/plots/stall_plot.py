from matplotlib import pyplot as plt
import numpy as np 
import os
import re

benchmark = ['Alexnet','Resnet18','ViT_b','ViT_bg','ViT_s', 'ViT_l', 'ViT_h' ]
config = ['32','128','512']
color = ['#edf8fb','#88419d']

reportFile = dict()
result = dict()

def file_extraction(filename, index, size):
    stall_cycles = 0
    total_cycles = 0
    with open(filename,'r') as f:
        for line in f.readlines():
            line.rstrip()
            if 'Total cycles' in line:
                total_cycles += (int(line.split(' ')[-1]))
            if 'Stall cycles' in line:
                stall_cycles += (int(line.split(' ')[-1]))
    f.close()
    result[(index, size)] = (total_cycles,stall_cycles)
    

def plot_bw():
    barWidth = 0.25
    label = 0
    plt.ylabel("Execution Time (in Cycles)",fontweight='bold',fontsize=12)
    xMid = np.arange(len(benchmark))
    x1 = [x - barWidth for x in xMid ]
    x2 = [x + barWidth for x in xMid ]
    plt.xticks(xMid, benchmark, rotation=30, ha='right',fontweight='bold',fontsize=10)
    for i in xMid:
        (total_cycles,stall_cycles) = result[(benchmark[i].lower(),config[0])]
        ratio = float(stall_cycles*100) / total_cycles
        print("[{0},{1}] - {2} {3} {4}".format(benchmark[i],config[0],stall_cycles,total_cycles, ratio))
        if label == 0:
            plt.bar(x1[i], total_cycles, width=barWidth, color=color[0], edgecolor='black', label='Total Cycles')
            plt.bar(x1[i], stall_cycles, width=barWidth, color=color[1], edgecolor='black', label='Stall Cycles')
            label=1
        else:
            plt.bar(x1[i], total_cycles, width=barWidth, color=color[0], edgecolor='black')
            plt.bar(x1[i], stall_cycles, width=barWidth, color=color[1], edgecolor='black')
        (total_cycles,stall_cycles) = result[(benchmark[i].lower(),config[1])]
        ratio = float(stall_cycles*100) / total_cycles
        print("[{0},{1}] - {2} {3} {4}".format(benchmark[i],config[1],stall_cycles,total_cycles, ratio))
        plt.bar(xMid[i], total_cycles, width=barWidth, color=color[0], edgecolor='black')
        plt.bar(xMid[i], stall_cycles, width=barWidth, color=color[1], edgecolor='black')
        
        (total_cycles,stall_cycles) = result[(benchmark[i].lower(),config[2])]
        ratio = float(stall_cycles*100) / total_cycles
        print("[{0},{1}] - {2} {3} {4}".format(benchmark[i],config[2],stall_cycles,total_cycles, ratio))
        plt.bar(x2[i], total_cycles, width=barWidth, color=color[0], edgecolor='black')
        plt.bar(x2[i], stall_cycles, width=barWidth, color=color[1], edgecolor='black')   
    plt.legend()
    plt.show()
    

if __name__ =="__main__":

    run_folder = os.getcwd() + '/Exp2/stall_cycles'
    for i in range(len(benchmark)):
        if not os.listdir(run_folder):
            assert "The run folder does not exist"
        for file in os.listdir(run_folder):
            if file.endswith('stall_out'):
                name = file.replace('_stall_out','')
                size = name.split('_')[-1]
                name = name.replace('_'+size, '')
                reportFile[(name,size)] = file
        for index in benchmark:
            for size in config:
                filepath = run_folder + '/' + reportFile[(index.lower(),size)]
                file_extraction(filepath, index.lower(), size)
    plot_bw()



