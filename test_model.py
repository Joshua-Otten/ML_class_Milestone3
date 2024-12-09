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


model = PeftModel.from_pretrained(model, './llama-2-7b-py-trans')

text_generator = pipeline("text-generation",model=model, tokenizer=tokenizer)#, max_new_tokens=10)

with open('final_test_set.txt','r') as f:
    test_set = json.load(f)

total_bleu = 0
total_chrf = 0
output = list()
count = 0
indiv_chrf_avgs = list()
indiv_chrf = 0
indiv_bleu_avgs = list()
indiv_bleu = 0
#test_set = ['[INST]Translate this English code into French Python: """print("Hello World")"""[/INST]imprimer("Hello World")']
# take out examples that are too long
t = list()
for example in test_set:
    #print(i)
    #print(test_set[i])
    prompt = '[INST]'+ str(example.split('[/INST]')[0].split('[INST]')[1].strip()) + '[/INST]'
    ref = example.split('[/INST]')[1].strip()
    if len(prompt) < 200 and prompt != '' and ref != None and ref != '': # 4096
        t.append(example)

    '''
    if len(prompt) > 4096:
        test_set.pop(test_set.index(example))
    elif prompt == None or prompt=='':
        test_set.pop(test_set.index(example))
    ref = example.split('[/INST]')[1].strip()
    if ref == None or ref=='':
        test_set.pop(test_set.index(example))
    '''
test_set = t

def data():
    for example in test_set:
        prompt = '[INST]'+ str(example.split('[/INST]')[0].split('[INST]')[1].strip()) + '[/INST]'
        yield prompt

j = 0
print("length of test set:",len(test_set))
for out in text_generator(data(), max_new_tokens=50):
    count += 1
    print(count)
    cand = out[0]["generated_text"].strip()
    ref = test_set[j].split('[/INST]')[1].strip()
    j += 1
    #print('cand:',cand)
    #print('ref:',ref)
    #print('complete:',test_set[j])
    bleu = sacrebleu.corpus_bleu([cand], [ref])
    total_bleu += bleu.score

    output.append({'cand':cand, 'ref':ref})
    indiv_bleu += bleu.score
    #print('bleu score:',bleu)
    
    chrf = sacrebleu.corpus_chrf([cand], [ref])
    total_chrf += chrf.score
    indiv_chrf += chrf.score
    #print('chrf score:',chrf)
    if count%100==0:
        print(count)
    if len(test_set) / 8 <= count:
        print('Finished with a type...')
        indiv_chrf_avgs.append(indiv_chrf / count)
        print('AVG chrF over this one type:',indiv_chrf / count)
        indiv_bleu_avgs.append(indiv_bleu / count)
        print('AVG bleu over this one type:',indiv_bleu / count)
        indiv_chrf = 0
        indiv_bleu = 0
        count = 0


'''
for example in test_set:
    count += 1
    #print(example)
    #prompt = example['text'].split('[/INST]')[0].split('[INST]')[1].strip()
    prompt = '[INST]'+ str(example.split('[/INST]')[0].split('[INST]')[1].strip()) + '[/INST]'
    #ref = example['text'].split('[/INST]')[1].strip()
    ref = example.split('[/INST]')[1].strip()
    print(len(prompt))
    if len(prompt) < 4096: # avoids generations that are too large for the model's predefined max length
        result = text_generator(prompt, max_length=len(prompt), do_sample=False, top_k=50)
        #print(result[0]["generated_text"])
        output.append({'cand':result[0]["generated_text"].strip(), 'ref':ref})
    
        ### evaluate based on BLEU and chrF score
        #b_score = sentence_bleu(ref, cand) # alt method for bleu score
        cand = result[0]["generated_text"].strip()
        cand = cand.split('[/INST]')[-1]
        print('generated text:',cand)
        bleu = sacrebleu.corpus_bleu([cand], [ref])
        total_bleu += bleu.score
        indiv_bleu += bleu.score
        #print('bleu score:',bleu)
    
        chrf = sacrebleu.corpus_chrf([cand], [ref])
        total_chrf += chrf.score
        indiv_chrf += chrf.score
        #print('chrf score:',chrf)
    if len(test_set) / 8 <= count:
        indiv_chrf_avgs.append(indiv_chrf / count)
        indiv_bleu_avgs.append(indiv_bleu / count)
        indiv_chrf = 0
        indiv_bleu = 0
        count = 0
'''
with open('/scratch/jotten4/python_translation/llm_test_translations.json','w') as outfile:
    json.dump(output, outfile)

print('Average BLEU Score:', total_bleu / len(test_set))
print('Average CHRF Score:', total_chrf / len(test_set))

print('Individual BLEU averages (in order of en-fr, fr-en, en-es, es-en, en-el, el-en, en-hi, hi-en)')
print(indiv_bleu_avgs)
print('Individual CHRF averages (in same order)')
print(indiv_chrf_avgs)


    
