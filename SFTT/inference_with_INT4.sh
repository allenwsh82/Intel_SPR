#numactl -C 0-55,112-167  python inference_with_INT4.py --prompt "What are some unique things about the 37th largest city in Japan?" --n-predict 200
numactl -C 0-55,112-167  python inference_with_INT4.py --prompt "what is the total Nike earnings before interest and taxes in fiscal year 2022 in Europe, Middle East and Africa?
" --n-predict 200
