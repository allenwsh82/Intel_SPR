#export OMP_NUM_THREADS=224
export OMP_NUM_THREADS=112

numactl -C 0-55,112-167 python fine_tuning.py --bf16 True --use_ipex True --max_seq_length 512
