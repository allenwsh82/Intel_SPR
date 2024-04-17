numactl -C 0-55,112-167  python inference_with_INT4.py --prompt "What are some unique things about the 37th largest city in Japan?" --n-predict 200
