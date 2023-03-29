# environment variables
export GLOO_SOCKET_IFNAME=enp94s0
# define exp config
declare -a all_models=(gcn sage)
declare -a servers_configs=(2 2)
declare -a workers_per_server_configs=(2 4)
# netowrk configurations
IP=127.0.0.1 # (use your own IP address)
PORT=1234
RANK=0
# loop through all experiment configurations
for ((i=0;i<${#servers_configs[@]};i++)); do
  NUM_SERVERS=${servers_configs[$i]}
  WORKERS_PER_SERVER=${workers_per_server_configs[$i]}
  # traing with all models
  for model in ${all_models[@]}; do
    torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
      --dataset ogbn-products \
      --num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
      --backend gloo \
      --init_method env:// \
      --model_name $model \
      --mode AdaQP \
      --assign_scheme adaptive \
      --logger_level INFO
  done
done
