from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import transformers
import torch
import argparse
import time
from threading import Thread

model_path="./fine_tuned_llama2-7B-hf-chat"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

###############################################################################################################
# With only 2 lines of additional lines if you are running your inference on PyTorch Framework #

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True, level="O1", auto_kernel_selection=True)

###############################################################################################################
#prompt = "Who is Leonardo Da Vinci?"
#prompt = "What is a virtual machine ?"
#prompt= "What are some unique things about the 37th largest city in Japan?"

LLAMA2_PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="./fine_tuned_llama2-7B-hf-chat",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'                ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What are some unique things about the 37th largest city in Japan?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=200,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print()
        print("Inference with Fine Tuned Llama2-7B with IPEX") 
        print()
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        token_num = len(output[0])
        print('Number of tokens:', token_num)
        print(f'Token/s: {token_num/(end-st)}')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
        print()
        
        
        
        
        
        
