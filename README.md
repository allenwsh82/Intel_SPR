# Intel-4th-Gen
This LLM AI demo is running on Intel 4th Gen Xeon processor (Codename: Sapphire Rapid)


Introduction:

Supervised fine-tuning, involves adapting a pre-trained Language Model (LLM) to a specific downstream task using labeled data. In supervised fine-tuning, the finetuning data is collected from a set of responses validated before hand. That’s the main difference to the unsupervised techniques, where data is not validated before hand. While LLM training is (usually) unsupervised, Finetuning is (usually) supervised.

Installation: 

pip install -r requirements.txt


SFTT (Supervised Fine Tuning Trainer) contains 3 scripts to execute the following task:
- Supervised Fine Tuning with HuggingFace Trainer API

How to run: ./fine_tuning.sh

CPU: Intel 8480+ (56 cores), Memory usage: 140GB 
Model: Llama-2-7b-chat-hf
Dataset: mlabonne/guanaco-llama2-1k [1000 Samples]
Configuration: Bfloat16, use_ipex, max_seq_length=512, num_of_epochs=1
Time to tune: 1 hour 45 minutes 
  
![SFT_Llama2](https://github.com/allenwsh82/Intel-4th-Gen/assets/44453417/b09ae9d8-9cc0-49c9-bf30-a5cd8e8d6388)

- Inference Fine Tuned Llama-2-7b with IPEX

How to run: ./inference_with_IPEX.sh
  
![Inference_Llama2_IPEX](https://github.com/allenwsh82/Intel-4th-Gen/assets/44453417/3965cbe7-46de-4b6e-a282-9abd679e97bb)
  
- Inference Fine Tuned Llama-2-7b with INT4

How to run: ./inference_with_INT4.sh
  
![Inference_Llama2_INT4](https://github.com/allenwsh82/Intel-4th-Gen/assets/44453417/276be93e-c924-4102-b0c1-76f5b6e7f042)


