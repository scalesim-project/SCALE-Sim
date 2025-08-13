from matplotlib import pyplot as plt
import numpy as np 
import os
import re
import argparse

cycle = 1 / 2.4     # Time period in ns (2.4GHz)
layerName = ['Conv1', 'Conv2_1a', 'Conv2_1b', 'Conv2_2a', 'Conv2_2b',
             'Conv3_1a', 'Conv3_1b', 'Conv3_s', 'Conv3_2a', 'Conv3_2b',
             'Conv4_1a', 'Conv4_1b', 'Conv4_s', 'Conv4_2a', 'Conv4_2b',
             'Conv5_1a', 'Conv5_1b', 'Conv5_s', 'Conv5_2a', 'Conv5_2b',
             'FC']
config = ['r1c1','r1c2','r1c4','r1c8']
color = ['#edf8fb','#b3cde3','#8c96c6','#88419d']

reportFile = dict()
result = dict()

def file_extraction(filename):
    dram_cycles = 0.0
    incoming_requests = 0.0
    with open(filename,'r') as f:
        for line in f.readlines():
            if 'dram_cycles' in line:
                dram_cycles = (float(re.split('\s+',line)[2]))
            if 'incoming_requests' in line:
                incoming_requests = (float(re.split('\s+',line)[2]))
    f.close()
    total_time = dram_cycles*cycle
    bw_value = (incoming_requests/total_time )* 1000
    return bw_value
    

def plot_bw():
    barWidth = 0.2
    plt.ylabel("Memory throughput (in MB/s)",fontweight='bold',fontsize=15)
    xMid = np.arange(len(result[config[0]]))
    x1 = [x - 1.5*barWidth for x in xMid ]
    x2 = [x - 0.5*barWidth for x in xMid ]
    x3 = [x + 0.5*barWidth for x in xMid ]
    x4 = [x + 1.5*barWidth for x in xMid ]
    plt.xticks(xMid,layerName, rotation=45, ha='right',fontweight='bold',fontsize=14)
    plt.bar(x1, result[config[0]], width=barWidth, color=color[0], edgecolor='black',label='1 Channel')
    plt.bar(x2, result[config[1]], width=barWidth, color=color[1], edgecolor='black',label='2 Channels')
    plt.bar(x3, result[config[2]], width=barWidth, color=color[2], edgecolor='black',label='4 Channels')
    plt.bar(x4, result[config[3]], width=barWidth, color=color[3], edgecolor='black',label='8 Channels')
    plt.xlim(x1[0]-2*barWidth, x4[-1]+2*barWidth)
    plt.legend()
    plt.show()
    

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-benchmark', metavar='benchmark', type=str,
                        default='Resnet18',
                        help='The benchmark to be run'
                        )
    arg = parser.parse_args()
    benchmark = arg.benchmark
    
    for i in range(len(config)):
        num_file=0
        run_folder = os.getcwd() + '/Exp1/' + benchmark+'_'+ config[i]
        if not os.listdir(run_folder):
            assert "The specific configuration does not exist"
        for file in os.listdir(run_folder):
            if file.startswith('DDR4_'):
                #index = int(re.findall('\d+',file)[1][2:])
                index = int(file.split('.')[0].split('_')[1])
                reportFile[(config[i],index)] = file
                num_file+=1
        bw = []
        for index in range(num_file):
            filepath = run_folder + '/' + reportFile[((config[i],index))]
            bw.append(file_extraction(filepath))
        result[config[i]] = bw
    plot_bw()



