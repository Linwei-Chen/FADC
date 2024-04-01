CONFIG=$1
GPUS=$2
# PORT=${PORT:-29500}
# PORT=${PORT:-29500}
PORT=${PORT:-$((1 + RANDOM % 10000))}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} 