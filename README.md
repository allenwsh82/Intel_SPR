# Supervised Fine Tuning with Intel-4th-Gen
This LLM AI demo is running on Intel 4th Gen Xeon processor (Codename: Sapphire Rapid)


Introduction:

Supervised fine-tuning, involves adapting a pre-trained Language Model (LLM) to a specific downstream task using labeled data. In supervised fine-tuning, the finetuning data is collected from a set of responses validated before hand. That’s the main difference to the unsupervised techniques, where data is not validated before hand. While LLM training is (usually) unsupervised, Finetuning is (usually) supervised.

Installation: 
```
pip install -r requirements.txt
```

SFTT (Supervised Fine Tuning Trainer) contains 3 scripts to execute the following task:
- Supervised Fine Tuning with HuggingFace Trainer API

How to run: 
```
./fine_tuning.sh
```

CPU: Intel 8480+ (56 cores), Memory usage: 140GB 
Model: Llama-2-7b-chat-hf
Dataset: mlabonne/guanaco-llama2-1k [1000 Samples]
Configuration: Bfloat16, use_ipex, max_seq_length=512, num_of_epochs=1
Time to tune: 1 hour 45 minutes 
  
![SFT_Llama2](https://github.com/allenwsh82/Intel_SPR/assets/44453417/64330f04-9f66-438e-83eb-be28b25c89cd)

![SFT_Llama2_2](https://github.com/allenwsh82/Intel_SPR/assets/44453417/1bea45bd-9f4a-493c-8798-9d7fdcaafbf6)


- Inference Fine Tuned Llama-2-7b with IPEX

How to run: 
```
./inference_with_IPEX.sh
```
  
![Inference_Llama2_IPEX](https://github.com/allenwsh82/Intel_SPR/assets/44453417/1a123121-d301-40c3-ac88-c3b2de4b7e33)

  
- Inference Fine Tuned Llama-2-7b with INT4

How to run: 
```
./inference_with_INT4.sh
```

  
![Inference_Llama2_INT4](https://github.com/allenwsh82/Intel_SPR/assets/44453417/95b409e5-94de-4e02-a3a9-822ddbcd7102)

Inference Performance Comparison:

<img width="810" alt="Inference_Performance_Comparison_Intel_4th_Gen" src="https://github.com/allenwsh82/Intel_SPR/assets/44453417/693d77a8-690b-4984-b493-0bd507e1d346">


