# Create or load any Pytorch model, take Llama-2-7b-chat-hf as an example

from transformers import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import transformers
import torch
import argparse
import time
import torch
import time
import argparse

from transformers import AutoModelForCausalLM, LlamaTokenizer
from ipex_llm import optimize_model

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
LLAMA2_PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--model-path', type=str, default="./fine_tuned_llama2-7B-hf-chat")
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=200,
                        help='Max tokens to predict')

    args = parser.parse_args()
    #model_path = args.model_path
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
 
    # With only one line to enable BigDL-LLM optimization on model
    model = optimize_model(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Inference with Fine Tuned Llama2-7B with INT4 Precision") 
        print()
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        token_num = len(output[0])
        print('Number of tokens:', token_num)
        print(f'Token/s: {token_num/(end-st)}')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
        print()
