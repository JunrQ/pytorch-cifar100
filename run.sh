
net=$1
gpus=$2

export CUDA_VISIBLE_DEVICES=${gpus}

python train.py -net ${net} -gpu > log/${net}.log 2>&1 &

