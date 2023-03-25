export GLOO_SOCKET_IFNAME=enp94s0
# define exp config
models=(gcn)
servers=(1)
workers_per_server_arr=(4)
IP="10.28.1.27"
PORT="1234"
RANK=0
# run the scripts
for ((i=0;i<${#servers[@]};i++))
do
for ((j=0;j<${#models[@]};j++))
do
export NUM_SERVERS=${servers[i]}
export WORKERS_PER_SERVER=${workers_per_server_arr[i]}
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--dataset reddit \
--mode AdaQP-q \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method env:// \
--logger_level INFO \
--model_name ${models[j]}
done
done