from transformers import AutoTokenizer, pipeline
from optimum.intel import OVModelForSeq2SeqLM
import time


title="NLP Translation running on Intel OpenVINO runtime powered by Intel Xeon 4th Gen"

model_id = "google-t5/t5-small"
model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
translation_pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)

#Create Batch size = 8

text_0 = "He never went out without a book under his arm, and he often came back with two."
text_1 = "Can you show me the way to the National Musuem?"
text_2 = "Can I have one plate of noodles?"
text_3 = "AI is not about GPU but also CPU, we have to make this fact clear to everyone"
text_4 = "Intel has wide range of product which includes Core, Xeon, IGPU, discrete GPU and Habana Gaudi"
text_5 = "Intel annouce 14th Gen processor during CES back in Jan 2024."
text_6 = "Mohan is one of the best ITS in Intel"
text_7 = "Ashton is a good boy who lives in Kuala Lumpur Malaysia"

inputs = [text_0, text_1, text_2, text_3, text_4, text_5, text_6, text_7]         

print("Below are the text which will be translated to French: ")
print()
for item in inputs:

    print (item)


print()
# get the start time
st = time.time()

result = translation_pipe(inputs)
print()
# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print()
print('Execution time:', elapsed_time, 'seconds')
print()
print("Original Text:", text_0, result[0])
print()
print("Original Text:", text_1, result[1])
print()
print("Original Text:", text_2,result[2])
print()
print("Original Text:", text_3,result[3])
print()
print("Original Text:", text_4,result[4])
print()
print("Original Text:", text_5,result[5])
print()
print("Original Text:", text_6,result[6])
print()
print("Original Text:", text_7,result[7])
print()

# Save the exported model
save_directory = "openvino_t5"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

samples = [ 
    ["translate English to French: Malaysia has the best food in the world"],
    ["translate English to French: Malaysia has some incredible wildlife outside of the rainforest"],
    ["translate English to French: May I know your name?"],
    ["translate English to French: Can you show me the direction to the hospital?"],
    ["translate English to French: Malaysia has 13 States"],
    ["translate English to Romanian: Malaysia has the best food in the world"],
    ["translate English to Romanian: Malaysia has some incredible wildlife outside of the rainforest"],
    ["translate English to Romanian: May I know your name?"],
    ["translate English to Romanian: Can you show me the direction to the hospital?"],
    ["translate English to Romanian: Malaysia has 13 States"]
