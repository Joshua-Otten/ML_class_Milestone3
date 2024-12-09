# Test the model on the test set
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, LoraConfig
from trl import SFTTrainer
import os
import gc
import torch
import json
#from nltk.translate.bleu_score import sentence_bleu
import sacrebleu


if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda")
else:
    print("GPU not available")
    device = torch.device("cpu")


#model = AutoModelForCausalLM.from_pretrained('./llama-2-7b-chat-guanaco')
base_model_path = 'NousResearch/Llama-2-7b-chat-hf'
compute_dtype = getattr(torch, 'float16')
quant_config = BitsAndBytesConfig(load_in_4bits=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

text_generator = pipeline("text-generation",model=model, tokenizer=tokenizer)#, max_new_tokens=10)

#test_set = ['[INST]Translate this English code into French Python: """print("Hello World")"""[/INST]imprimer("Hello World")']
prompt = '<s>[INST]Translate this English code into French Python: """print("hello world")\nx=2\nfor i in range(x):\n\tif x==2:\n\t\tprint("2")\n\telse:\n\t\tprint("x != 2")"""[/INST]'
reference = """imprimer("hello world")\nx=2\npour i dans port\u00E9e(x):\n\tsi x==2:\n\t\timprimer("2")\n\tautre:\n\t\timprimer("x != 2")""" 
result = text_generator(prompt, max_new_tokens = 100)
#print('Baseline:\n'+str(result[0]["generated_text"].strip().split('[/INST]')[1]))
cand = result[0]["generated_text"].strip().split('[/INST]')[1]
print('Baseline:\n'+cand)
bleu = sacrebleu.corpus_bleu([cand],[reference])
chrf = sacrebleu.corpus_chrf([cand],[reference])
print('Baseline BLEU:',bleu)
print('Baseline chrF:',chrf)

# now for the 1-epoch model
model = PeftModel.from_pretrained(model, './llama-2-7b-py-trans')
text_generator = pipeline("text-generation",model=model, tokenizer=tokenizer)#, max_new_tokens=10)
result = text_generator(prompt, max_new_tokens = 100)
#print('1-epoch model:\n'+str(result[0]["generated_text"].strip().split('[/INST]')[1]))
cand = result[0]["generated_text"].strip().split('[/INST]')[1].strip()
print('1-epoch model:\n'+cand)
bleu = sacrebleu.corpus_bleu([cand],[reference])
chrf = sacrebleu.corpus_chrf([cand],[reference])
print('1-epoch model BLEU:',bleu)
print('1-epoch model chrF:',chrf)
# finally for the second model

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', quantization_config=quant_config)
model = PeftModel.from_pretrained(model, './llama-2-7b-py-trans2-small')
text_generator = pipeline("text-generation",model=model, tokenizer=tokenizer)#, max_new_tokens=10)
result = text_generator(prompt, max_new_tokens = 100)
#print('Second model:\n'+str(result[0]["generated_text"].strip().split('[/INST]')[1]))
cand = result[0]["generated_text"].strip().split('[/INST]')[1].strip()
print('Second model:\n'+cand)
bleu = sacrebleu.corpus_bleu([cand],[reference])
chrf = sacrebleu.corpus_chrf([cand],[reference])
print('Second model BLEU:',bleu)
print('Second model chrF:',chrf)


print('reference:',reference)
