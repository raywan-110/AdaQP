export GLOO_SOCKET_IFNAME=enp94s0
# define exp config
model_name="gcn"
servers=1
workers_per_server_arr=4
IP="10.28.1.27"
PORT="1234"
RANK=0
# run the scripts
export NUM_SERVERS=$servers
export WORKERS_PER_SERVER=$workers_per_server_arr
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--dataset reddit \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method env:// \
--model_name $model_name \
--mode AdaQP-q \
--assign_scheme adaptive \
--logger_level INFO