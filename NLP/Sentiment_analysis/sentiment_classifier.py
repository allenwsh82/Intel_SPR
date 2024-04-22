import psutil
import gradio as gr
import shutil
from pathlib import Path
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
import time

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


title = "NLP Sentiment Classifier by Intel Xeon 4th Gen Sapphire Rapid"

# The following model has been quantized, sparsified using Optimum-Intel 1.7 which is enabled by OpenVINO and NNCF
# for reproducibility, refer https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80
model_id = "OpenVINO/bert-base-uncased-sst2-int8-unstructured80"

# The following two steps will set up the model and download them to HF Cache folder
#model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
model = OVModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the exported model
save_directory = "openvino_bert-base-uncased-sst2-int8-unstructured80"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Let's take the model for a spin!
sentiment_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

#input = text
# Doing batching job, batch_size = 8

print("Executing 1 prompt with batch size = 16")
print()
text_0 = "He's a dreadful magician"
text_1 = "Intel is the best processor in the world"
text_2 = "Nasi lemak is one of the best food in Malaysia"
text_3 = "The toilet in that mall is dirty"
text_4 = "Intel AI PC is the best in the world"
text_5 = "I hate the hot weather"
text_6 = "I don't like his atttitude"
text_7 = "I love to play badminton"
text_8 = "He's a dreadful magician"
text_9 = "Intel is the best processor in the world"
text_10 = "Nasi lemak is one of the best food in Malaysia"
text_11 = "The toilet in that mall is dirty"
text_12 = "Intel AI PC is the best in the world"
text_13 = "I hate the hot weather"
text_14 = "I don't like his atttitude"
text_15 = "I love to play badminton"

#Define inputs 

inputs=[text_0, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_10, text_11, text_12, text_13, text_14, text_15]   


print()
# get the start time
st = time.time()

outputs = sentiment_classifier(inputs)

print("Now we are going to run inferencing with Batch_Size=16 for inputs for Sentiment Analysis..........")
print()
print()
print(text_0, outputs[0])
print()
print(text_1, outputs[1])
print()
print(text_2, outputs[2])
print()
print(text_3, outputs[3])
print()
print(text_4, outputs[4])
print()
print(text_5, outputs[5])
print()
print(text_6, outputs[6])
print()
print(text_7, outputs[7])
print()
print(text_8, outputs[8])
print()
print(text_9, outputs[9])
print()
print(text_10, outputs[10])
print()
print(text_11, outputs[11])
print()
print(text_12, outputs[12])
print()
print(text_13, outputs[13])
print()
print(text_14, outputs[14])
print()
print(text_15, outputs[15])
print()


    
# get the end time
et = time.time()
print()
print()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print()
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
print()
