# Adaptive Message Quantization and Parallelization for Distributed Full-graph GNN Training

Borui Wan (The University of Hong Kong), Juntao Zhao (The University of Hong Kong), Chuan Wu (The University of Hong Kong)

Accepted by MLSys 2023



## Directory Hierarchy

```
|-- AdaQP                 # source code of AdaQP
|   `-- assigner
|   `-- communicator
|   `-- config            # offline configurations of experiments
|   `-- helper
|   `-- manager
|   `-- model             # customized PyTorch modules
|   `-- trainer
|   `-- util
|       `-- quantization  # quantization/de-quantization extensions
|-- data                  # raw datasets and graph partitions
|   `-- dataset
|   `-- part_data
|-- exp                   # experiment results
|-- graph_degrees         # global degrees of original graphs
|-- gurobi_license        # license file for Gurobi solver
`-- scripts                # training scripts
```

Please note that we only provide `./gurobi_license/` for Artifact Evaluation purposes. We will remove the lisense file when releasing the code publicly due to privacy concerns.

## Setup

### Environment

#### Hardware Dependencies

- X86-CPU machines, each with 256 GB host memory (recommended).  
- Nvidia GPUs (at least 16 GB each)

#### Software Dependencies

- Ubuntu 20.04 LTS
- Python 3.8
- CUDA 11.3
- [PyTorch 1.11.0](https://github.com/pytorch/pytorch)
- [DGL 0.9.0](https://github.com/dmlc/dgl)
- [OGB 1.3.3](https://ogb.stanford.edu/docs/home/)
- [PuLP 2.6.0](https://github.com/coin-or/pulp)
- [Gurobi 9.5.2](https://anaconda.org/Gurobi/gurobi)

### Installation

#### Option 1: Run with Docker (Recommended)

We have prepared a [Docker package](https://hub.docker.com/r/raydarkwan/adaqp) for AdaQP.

```bash
docker pull raydarkwan/adaqp
docker run -it --gpus all --network=host raydarkwan/adaqp
```

#### Option 2: Install with Conda

Running the following command will install PuLP from pip, quant_cuda from source, and other prerequisites from conda.

```bash
bash setup.sh
```

### Datasets

We use Reddit, Yelp, ogbn-products and AmazonProducts for evaluation. All datasets are supposed to be stored in `./data/dataset` by default. Yelp is preloaded in the Docker environment, and is available [here](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) if you choose to set up the enviromnent by yourself. 


## Usage

### Partition the Graph

Before conducting training, run `script/partition/partition_<dataset>.sh` to partition the coressponding graph into several subgraphs, and store them into `./data/part_data/<dataset>` by default. Customized partitioning methods are also supported. For example, to partition the Reddit graph into 4 subgraphs, run

```bash
python graph_partition.py --dataset reddit --partition_size 4
```

For multi-node multi-GPU training, please make sure that the corresponding subgraphs are stored in the same directory on all machines. For example, directory hierarchy like `./data/part_data/reddit/4part/part0`, `./data/part_data/reddit/4part/part1` on **machine A** and `./data/part_data/reddit/4part/part2`, `./data/part_data/reddit/4part/part3` on **machine B** can support lauching two training processes on each machine. Besides, data in `./graph_degrees` should also be copied to all machines.

### Train the Model

Run `scripts/example/<dataset>_vanilla.sh` and `scripts/example/<dataset>_adaqp.sh` to see the performance of Vanilla and AdaQP under single-node multi-GPU settings. Note that variables like `IP`, `PORT`, `RANK` need to be speficied in the scripts before launching the training jobs. An example of training scripts is shown below:

```bash
# variables
NUM_SERVERS=1
WORKERS_PER_SERVER=2
RANK=0
# network configurations
IP=XX.XX.X.XX
PORT=8888
# run the script
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--dataset reddit \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method env:// \
--model_name gcn \
--mode Vanilla \
--logger_level INFO
```
Besides, the environment variable `GLOO_SOCKET_IFNAME` may need to be set for multi-node training. 

### Training Arguments

Core training arguments are listed below:

- `--dataset`: the training dataset
- `--num_parts`: the number of partitions
- `--model_name`: the GNN model (only GCN and GraphSAGE are supported at this moment)
- `--mode`: the training method (Vanilla, AdaQP or variants of AdaQP)
- `--assignment`: bit-width assignment scheme used by AdaQP's Assigner
- `--use_parallel`: whether to use parallelization

Please refer to `main.py` for more details. All offline default configurations for datasets, models, and bit-width assignment can be found in `./AdaQP/config/`, adjust them to customize your settings.

### Reproduce Experiments

Reproduce the core experiments (throughput, accuracy, time breakdown) by running `scripts/<dataset>_all.sh`. The experiment results will be saved to `./exp/<dataset>` directory.

## License

Copyright (c) 2023 ray wan. All rights reserved.

Licensed under the MIT license.
