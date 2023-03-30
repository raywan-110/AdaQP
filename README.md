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
|-- scripts               # training scripts
```

`data`, `exp`, `graph_degrees`, will be created after lauching the scripts. Beside, note that we only provide `./gurobi_license/` for Artifact Evaluation purposes. **The lisense file will be removed after releasing the code publicly due to privacy concerns.** Your can apply for your own lisence by following the instructions in [gurobi-licenses application](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Setup

### Environment

#### Hardware Dependencies

- X86-CPU machines, each with 256 GB host memory (recommended).  
- Nvidia GPUs (32 GB each)

#### Software Dependencies

- Ubuntu 20.04 LTS
- Python 3.8
- CUDA 11.3
- [PyTorch 1.11.0](https://github.com/pytorch/pytorch)
- [DGL 0.9.0](https://github.com/dmlc/dgl)
- [OGB 1.3.3](https://ogb.stanford.edu/docs/home/)
- [PuLP 2.6.0](https://github.com/coin-or/pulp)
- [Gurobi 9.5.2](https://anaconda.org/Gurobi/gurobi)
- Quant_cuda 0.0.0 (customized quantization/de-quantization kernels)

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

Before conducting training, run `script/partition/partition_<dataset>.sh` to partition the coressponding graph into several subgraphs, and store them into `./data/part_data/<dataset>` by default. Customized partitioning are also supported. For example, to partition the Reddit graph into 4 subgraphs, run

```bash
python graph_partition.py --dataset reddit --partition_size 4
```

For multi-node multi-GPU training, please make sure that the corresponding subgraphs are stored in the same directory on all machines. For example, directory hierarchy like `./data/part_data/reddit/4part/part0`, `./data/part_data/reddit/4part/part1` on **machine A** and `./data/part_data/reddit/4part/part2`, `./data/part_data/reddit/4part/part3` on **machine B** can support lauching two training processes on each machine. Besides, `./graph_degrees` should also be copied to all machines.

### Train the Model

Run `scripts/example/<dataset>_vanilla.sh` and `scripts/example/<dataset>_adaqp.sh` to see the performance of Vanilla and AdaQP under single-node multi-GPU settings. Note that variables like `IP`, `PORT`, `RANK` need to be speficied in the scripts before launching the training jobs. An example of training scripts is shown below:

```bash
# variables
NUM_SERVERS=1
WORKERS_PER_SERVER=4
RANK=0
# network configurations
IP=127.0.0.1
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
The output in the console will be like:

```bash
INFO:trainer:Epoch 00010 | Loss 0.0000 | Train Acc 85.77% | Val Acc 87.00% | Test Acc 86.75%
Worker 0 | Total Time 1.1635s | Comm Time 0.8469s | Quant Time 0.0000s | Agg Time 0.2060s | Reduce Time 0.0466s
INFO:trainer:Epoch 00020 | Loss 0.0000 | Train Acc 92.37% | Val Acc 93.00% | Test Acc 92.98%
Worker 0 | Total Time 1.0919s | Comm Time 0.7889s | Quant Time 0.0000s | Agg Time 0.1786s | Reduce Time 0.0690s
INFO:trainer:Epoch 00030 | Loss 0.0000 | Train Acc 93.33% | Val Acc 93.81% | Test Acc 93.91%
```

For multi-node multi-GPU training, copy the source code to all machines and launch all the scripts respectively. Besides, the environment variable `GLOO_SOCKET_IFNAME` may need to be set as the inferfaces name.

### Training Arguments

Core training arguments are listed below:

- `--dataset`: the training dataset
- `--num_parts`: the number of partitions
- `--model_name`: the GNN model (only GCN and GraphSAGE are supported at this moment)
- `--mode`: the training method (Vanilla, AdaQP or variants of AdaQP)
- `--assignment`: bit-width assignment scheme used by AdaQP's Assigner

Please refer to `main.py` for more details. All offline default configurations for datasets, models, and bit-width assignment can be found in `./AdaQP/config/`, adjust them to customize your settings.

### Reproduce Experiments

Reproduce the core experiments (throughput, accuracy, time breakdown) by running `scripts/<dataset>_all.sh`. The experiment results will be saved to `./exp/<dataset>` directory. The variable `RANK` in the scripts should be adjusted on different machines accordingly.

### Experiment Customization

Adjust configurations in `AdaQP/config/*yaml` to customize dataset, model, training hyperparameter, bit-width assignment settings or add new configurations; adjust runtime arguments in `scripts/*` to customize graph partitions numbers, optional GPUs or machines for training, bit-width assignment strategies and training methods (AdaQP and its variants). For example, set argument `--assignment` to `random` to use random bit-width assignment, or set argument `--mode` to `AdaQP-q` to use AdaQP with only message quantization.

## License

Copyright (c) 2023 ray wan. All rights reserved.

Licensed under the MIT License.
