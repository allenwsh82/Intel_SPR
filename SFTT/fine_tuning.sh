export OMP_NUM_THREADS=224
python fine_tuning.py --bf16 True --use_ipex True --max_seq_length 512
