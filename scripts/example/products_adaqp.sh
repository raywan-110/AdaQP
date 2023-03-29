# variables
NUM_SERVERS=1
WORKERS_PER_SERVER=4
RANK=0
# network configurations
IP=127.0.0.1
PORT=8888
# run the script
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--dataset ogbn-products \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method env:// \
--model_name gcn \
--mode AdaQP \
--assign_scheme adaptive \
--logger_level INFO