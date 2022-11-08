import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER
import torch
def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")
    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument('--no_hyperthreads', action='store_true',
                        help='Flag to disable binding to hyperthreads')
    parser.add_argument('--no_membind', action='store_true',
                        help='Flag to disable memory binding')
    # non-optional arguments for binding
    parser.add_argument("--nsockets_per_node", type=int, required=True,
                        help="Number of CPU sockets on a node")
    parser.add_argument("--ncores_per_socket", type=int, required=True,
                        help="Number of CPU cores per socket")
    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")
    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()
def main():
    args = parse_args()
    # variables for numactrl binding
    #NSOCKETS = args.nsockets_per_node
    #NGPUS_PER_SOCKET = args.nproc_per_node // args.nsockets_per_node
    #NCORES_PER_GPU = args.ncores_per_socket // NGPUS_PER_SOCKET
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes
    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    processes = []
    #custom binding for A100 GPUs on AMD ROME 2 socket systems
    cpu_ranges = []
    cpu_ranges.append([0,3,80,83])  #local_rank=0
    cpu_ranges.append([4,7,84,87])  #local_rank=0
    cpu_ranges.append([8,10,88,90])  #local_rank=0
    cpu_ranges.append([11,14,91,94])  #local_rank=0
    cpu_ranges.append([15,17,95,97])  #local_rank=0
    cpu_ranges.append([18,20,98,100])  #local_rank=0
    cpu_ranges.append([21,23,101,103])  #local_rank=0
    cpu_ranges.append([24,26,104,106])  #local_rank=0
    cpu_ranges.append([27,30,107,110])  #local_rank=0
    cpu_ranges.append([31,32,111,112])  #local_rank=0
    cpu_ranges.append([33,36,113,116])  #local_rank=0
    cpu_ranges.append([37,39,117,119])  #local_rank=0
    cpu_ranges.append([40,43,120,123])  #local_rank=1
    cpu_ranges.append([44,47,124,127])  #local_rank=1
    cpu_ranges.append([48,50,128,130])  #local_rank=1
    cpu_ranges.append([51,53,131,133])  #local_rank=1
    cpu_ranges.append([54,57,134,137])  #local_rank=1
    cpu_ranges.append([58,60,138,140])  #local_rank=1
    cpu_ranges.append([61,63,141,143])  #local_rank=1
    cpu_ranges.append([64,66,144,146])  #local_rank=1
    cpu_ranges.append([67,70,147,150])  #local_rank=1
    cpu_ranges.append([71,73,151,153])  #local_rank=1
    cpu_ranges.append([74,76,154,156])  #local_rank=1
    cpu_ranges.append([77,79,157,159])  #local_rank=1
    
    memnode = []
    memnode.append(0)  #local_rank=0
    memnode.append(0)  #local_rank=1
    memnode.append(0)  #local_rank=2
    memnode.append(0)  #local_rank=3
    memnode.append(0)  #local_rank=4
    memnode.append(0)  #local_rank=5
    memnode.append(0)  #local_rank=6
    memnode.append(0)  #local_rank=7
    memnode.append(0)  #local_rank=8
    memnode.append(0)  #local_rank=9
    memnode.append(0)  #local_rank=10
    memnode.append(0)  #local_rank=11
    memnode.append(1)  #local_rank=12
    memnode.append(1)  #local_rank=13
    memnode.append(1)  #local_rank=14
    memnode.append(1)  #local_rank=15
    memnode.append(1)  #local_rank=16
    memnode.append(1)  #local_rank=17
    memnode.append(1)  #local_rank=18
    memnode.append(1)  #local_rank=19
    memnode.append(1)  #local_rank=20
    memnode.append(1)  #local_rank=21
    memnode.append(1)  #local_rank=22
    memnode.append(1)  #local_rank=23
    


    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        # form numactrl binding command
        #cpu_ranges = [local_rank * NCORES_PER_GPU,
        #             (local_rank + 1) * NCORES_PER_GPU - 1,
        #             local_rank * NCORES_PER_GPU + (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS),
        #             (local_rank + 1) * NCORES_PER_GPU + (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS) - 1]
        numactlargs = []
        if args.no_hyperthreads:
            numactlargs += [ "--physcpubind={}-{}".format(*cpu_ranges[local_rank][0:2]) ]
        else:
            numactlargs += [ "--physcpubind={}-{},{}-{}".format(*cpu_ranges[local_rank]) ]
        if not args.no_membind:
            #memnode = local_rank // NGPUS_PER_SOCKET
            numactlargs += [ "--membind={}".format(memnode[local_rank]) ]
        # spawn the processes
        cmd = [ "/usr/bin/numactl" ] \
            + numactlargs \
            + [ sys.executable,
                "-u",
                args.training_script,
                "--local_rank={}".format(local_rank)
              ] \
            + args.training_script_args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
        #print(local_rank,cmd)
    for process in processes:
        process.wait()
if __name__ == "__main__":
    main()

