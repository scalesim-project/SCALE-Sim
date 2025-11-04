import numpy as np
import os, os.path
import sys
import argparse
import subprocess
import queue
import multiprocessing as mp

rootPath=os.getcwd()
resultsPath=os.getcwd()+"/results/"

mp.set_start_method('spawn',True)

# ---- Replace data structures to panda dataframe for quicker access --- #
class dataExtraction:
    def __init__(   self,
                    ifmapFile=[],
                    ofmapFile=[],
                    filterFile=[],
                    multiTenant=False,
                    traceMap='',
                    ramulatorOut=''
                    ):
        self.multiTenant = multiTenant
        self.traceMap = traceMap
        self.ramulatorOut = ramulatorOut
        self.ifmap_file = ifmapFile
        self.ofmap_file = ofmapFile
        self.filter_file = filterFile

        self.ifmapStartCycle = 0
        self.filterStartCycle = 0
        self.ofmapStartCycle = 0
        self.bw = 1
        self.cycle =[]
    
    def extractAddress(self,ifmapFile,ofmapFile,filterFile,layerNo,shaper):
        ifmapAddress=[]
        ofmapAddress=[]
        filterAddress=[]
        ofmapIntegrityAddress=[]
        ifmapCycle=[]
        ofmapCycle=[]
        ofmapIntegrityCycle=[]
        filterCycle=[]
        flag = 0
        txnCount=0
        fake_address = 40000000
    # ----- Extract DRAM transactions for input feature map ----- #
        with open(ifmapFile,'r') as f:
            for line in f.readlines():
                line = line.strip()
                x = line.split(',')
                
                if flag ==0:
                    self.ifmapStartCycle = int(float(x[0]))
                    self.bw = len(x) - 1
                    flag=1
                ifmapCycle.append(int(float(x[0])))
                
                for i in range(1,len(x)):
                    ifmapAddress.append(hex(int(float(x[i]))))
                txnCount+=1
                
                #cycle = int(x[0])
                #self.txnCount.append(len(x) -1)
        f.close()
        print("Layer%s: Number of IFMAP lines is %d" % (layerNo, txnCount))
        print("Layer%s: Reading IFMAP file complete" % layerNo)
        
        txnCount=0
        flag = 0
    # ----- Extract DRAM transactions for filter map ----- #
        with open(filterFile,'r') as f:
            for line in f.readlines():
                line = line.strip()
                x = line.split(',')
                if flag == 0:
                    self.filterStartCycle = int(float(x[0]))
                    flag=1
                filterCycle.append(int(float(x[0])))
                for i in range(1,len(x)):
                    filterAddress.append(hex(int(float(x[i]))))  
                txnCount+=1
        f.close()
        print("Layer%s: Number of FILTER lines is %d" % (layerNo, txnCount))
        print("Layer%s: Reading FILTER file complete" % layerNo)
        txnCount=0
        flag = 0
        integrityRMW = 0
    # ----- Extract DRAM transactions for output feature map ----- #
        with open(ofmapFile,'r') as f:
            for line in f.readlines():
                line = line.strip()
                x = line.split(',')
                if float(x[1]) > 0.0:
                    ofmapCycle.append(int(float(x[0])))
                    for i in range(1,len(x)):
                        ofmapAddress.append(hex(int(float(x[i]))))
                    txnCount+=1
                else:
                    ofmapIntegrityCycle.append(int(float(x[0])))
                    for i in range(1,len(x)):
                        ofmapIntegrityAddress.append(hex(-int(float(x[i]))))
                    integrityRMW+=1
        f.close()
        self.ofmapStartCycle = min(ofmapCycle)
        print("Layer%s: Number of OFMAP lines is %d" % (layerNo, txnCount))
        print("Layer%s: Number of OFMAP Integrity Read lines is %d" % (layerNo, integrityRMW))
        print("Layer%s: Reading OFMAP file complete" % layerNo)
        np.asarray(ifmapAddress)
        np.asarray(filterAddress)
        np.asarray(ofmapAddress)
        np.asarray(ofmapIntegrityAddress)
        np.asarray(ifmapCycle)
        np.asarray(filterCycle)
        np.asarray(ofmapCycle)
        np.asarray(ofmapIntegrityCycle)
        print("Layer%s: Conversion to array complete" % layerNo)

        ifmapIndex=0
        filterIndex=0
        ofmapIndex=0
        ofmapIntegrityIndex=0
        flag =0
        total_count = 0
        
        f = open(self.traceMap,'w')
        cycle = min(self.ifmapStartCycle,self.filterStartCycle,self.ofmapStartCycle)
        maxCycle= max(ifmapCycle[-1],filterCycle[-1],ofmapCycle[-1])
        while (cycle !=maxCycle+1):
            #if cycle != ifmapCycle[0] or cycle!=ofmapCycle[0] or cycle!=filterCycle[0]:
            if cycle in range(ifmapCycle[0],ifmapCycle[-1]+1):        # Valid cycle range for the input feature map trace
                if cycle == ifmapCycle[ifmapIndex]:
                    flag=1
                    for i in range(self.bw):
                        value = ifmapAddress[ifmapIndex*self.bw+i]
                        if not value== "-0x1": 
                            f.write(str(value)+" R\n")
                        elif shaper == 1:
                            f.write(str(hex(fake_address)) + " R\n")

                    ifmapIndex+=1

            if cycle in range(filterCycle[0],filterCycle[-1]+1):        # Valid cycle range for the filter map trace          
                if cycle == filterCycle[filterIndex]: 
                    flag=1
                    for i in range(self.bw):
                        value = filterAddress[filterIndex*self.bw+i]
                        if not value== "-0x1": 
                            f.write(str(value)+" R\n")
                        elif shaper == 1:
                            f.write(str(hex(fake_address)) + " R\n")
                    filterIndex+=1
           
            if cycle in range(ofmapCycle[0],ofmapCycle[-1]+1):        # Valid cycle range for the output feature map trace   
                if cycle == ofmapCycle[ofmapIndex]:
                    flag=1
                    for i in range(self.bw):
                        value = ofmapAddress[ofmapIndex*self.bw+i]
                        if not value== "-0x1": 
                            f.write(str(value)+" W\n")                        
                        elif shaper == 1:
                            f.write(str(hex(fake_address)) + " W\n")
                    ofmapIndex+=1
            if len(ofmapIntegrityCycle) !=0:
                if cycle in range(ofmapIntegrityCycle[0],ofmapIntegrityCycle[-1]+1):        # Valid cycle range for the input feature map trace   
                    if cycle == ofmapIntegrityCycle[ofmapIntegrityIndex]:
                        flag=1
                        for i in range(self.bw):
                            value = ofmapIntegrityAddress[ofmapIntegrityIndex*self.bw+i]
                            if not value== "-0x1": 
                                f.write(str(value)+" R\n")                        
                            elif shaper == 1:
                                f.write(str(hex(fake_address)) + " R\n")
                        ofmapIntegrityIndex+=1
            if flag == 1:
                total_count += self.bw
            else:
                print(cycle)
                print(ifmapCycle[ifmapIndex])
                print(filterCycle[filterIndex])
                print(ofmapCycle[ofmapIndex])
            cycle+=1
        f.close()
        print("The number of IFMAP, FILTER, OFMAP, integrity index is {} {} {} {}".format(ifmapIndex,filterIndex,ofmapIndex,ofmapIntegrityIndex))

    def runRamulator(self,prefix):
        output=subprocess.check_output([rootPath+"/submodules/ramulator/ramulator",
                        rootPath+"/submodules/ramulator/configs/DDR4-config.cfg",
                        "--mode=dram",
                        "--stats",
                        "results/DDR4_"+prefix+".stats",
                        self.traceMap], universal_newlines=True)
        f = open(self.ramulatorOut, "w")
        f.write(output)
        f.close()

def worker(layer_path, topo, shaper):
    """ Running ramulator for multiple layers request in parallel """
    layer_no = layer_path.split('/')[-1].replace('layer','')
    if not os.path.isdir(layer_path):
        sys.exit("Please run scalesim with oracle memory first to get the demand requests")
    ifmap_file  = layer_path+"/IFMAP_DRAM_TRACE.csv"  #args.ifmap_file
    filter_file = layer_path+"/FILTER_DRAM_TRACE.csv" #args.filter_file
    ofmap_file  = layer_path+"/OFMAP_DRAM_TRACE.csv"  #args.ofmap_file
    if not os.path.isdir(resultsPath):
        os.mkdir(resultsPath)
    mem_trace_in  = resultsPath+topo+"_DemandTrace_"+layer_no+".trace" #args.mem_trace_in
    mem_trace_out = resultsPath+topo+"_RamulatorTrace_"+layer_no+".trace" #args.mem_trace_out
    ramulatorExtraction = dataExtraction(
                                        ifmapFile=ifmap_file,
                                        ofmapFile=ofmap_file,
                                        filterFile=filter_file,
                                        traceMap=mem_trace_in,
                                        ramulatorOut=mem_trace_out
                                    )

    #--- call the ramulator functions

    print("starting layer {}".format(layer_path))
    ramulatorExtraction.extractAddress(ifmap_file,ofmap_file,filter_file,layer_no,shaper)

    prefix = topo+layer_no
    print("Starting ramulator for layer {}".format(layer_path))
    ramulatorExtraction.runRamulator(prefix)

    print("Finished ramulator for layer {}".format(layer_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-topology', metavar='Topology file', type=str,
                        default="",
                        help="Directory path for the layers"
                        )
    parser.add_argument('-run_name', metavar='Config run name', type=str,
                        default="GoogleTPU_v1_ws",
                        help="Directory path for the layers"
                        )
    parser.add_argument('-shaper', metavar='Memory shaping logic', type=bool,
                        default=False,
                        help="Enable fake transactions"
                        )


    args = parser.parse_args()
    topology = args.topology
    run = args.run_name
    shaper = args.shaper
    layers_path = []
    if not os.listdir(resultsPath+topology):
        assert "Generate DRAM demand transactions before running Ramulator"
    filepath = resultsPath+topology+'/'+run+'/'
    for file in os.listdir(filepath):
        if file.startswith('layer') and os.path.isdir(filepath+file):
            layers_path.append(filepath+file)
    print(layers_path)
    #worker(layers_path[0],topology,run,shaper)
    for layer_path in layers_path:
        p = mp.Process(target=worker, args=(layer_path, topology, shaper))
        p.start()
