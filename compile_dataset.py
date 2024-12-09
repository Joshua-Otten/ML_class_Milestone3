# make complete dataset from English + multilingual code examples
import random
import json

languages = ['French', 'Spanish', 'Greek', 'Hindi']

with open('python_dataset.json','r') as f:
    en_examples = json.load(f)

with open('French_code_examples.json','r') as f:
    fr_examples = json.load(f)


with open('Spanish_code_examples.json','r') as f:
    es_examples = json.load(f)

with open('Greek_code_examples.json','r') as f:
    el_examples = json.load(f)

with open('Hindi_code_examples.json','r') as f:
    hi_examples = json.load(f)


en_fr_dataset = list()
en_es_dataset = list()
en_el_dataset = list()
en_hi_dataset = list()
fr_en_dataset = list()
es_en_dataset = list()
el_en_dataset = list()
hi_en_dataset = list()
          
for key in fr_examples:

    ### English --> Other language
    to_append = '[INST]Translate this English code into French Python: ' + en_examples[key] + '[/INST] ' + fr_examples[key]
    en_fr_dataset.append(to_append)

    to_append = '[INST]Translate this English code into Spanish Python: ' + en_examples[key] + '[/INST] ' + es_examples[key]
    en_es_dataset.append(to_append)

    to_append = '[INST]Translate this English code into Greek Python: ' + en_examples[key] + '[/INST] ' + el_examples[key]
    en_el_dataset.append(to_append)

    to_append = '[INST]Translate this English code into Hindi Python: ' + en_examples[key] + '[/INST] ' + hi_examples[key]
    en_hi_dataset.append(to_append)


    ### Other --> English
    to_append = '[INST]Translate this French code into standard English Python: ' + fr_examples[key] + '[/INST] ' + en_examples[key]
    fr_en_dataset.append(to_append)

    to_append = '[INST]Translate this Spanish code into standard English Python: ' + es_examples[key] + '[/INST] ' + en_examples[key]
    es_en_dataset.append(to_append)

    to_append = '[INST]Translate this Greek code into standard English Python: ' + el_examples[key] + '[/INST] ' + en_examples[key]
    el_en_dataset.append(to_append)

    to_append = '[INST]Translate this Hindi code into standard English Python: ' + hi_examples[key] + '[/INST] ' + en_examples[key]
    hi_en_dataset.append(to_append)


### create training/test splits, and randomly permute training examples
train_num = int(0.8 * len(en_fr_dataset)) # ~80% of the entire dataset

train_en_fr = en_fr_dataset[:train_num]
test_en_fr = en_fr_dataset[train_num:]
random.shuffle(test_en_fr)
en_fr_dataset = None # get rid of extra memory usage

train_en_es = en_es_dataset[:train_num]
test_en_es = en_es_dataset[train_num:]
random.shuffle(test_en_es)
en_es_dataset = None

train_en_el = en_el_dataset[:train_num]
test_en_el = en_el_dataset[train_num:]
random.shuffle(test_en_el)
en_el_dataset = None

train_en_hi = en_hi_dataset[:train_num]
test_en_hi = en_hi_dataset[train_num:]
random.shuffle(test_en_hi)
en_hi_dataset = None
          
# don't forget reverse direction
train_fr_en = fr_en_dataset[:train_num]
test_fr_en = fr_en_dataset[train_num:]
random.shuffle(test_fr_en)
fr_en_dataset = None

train_es_en = es_en_dataset[:train_num]
test_es_en = es_en_dataset[train_num:]
random.shuffle(test_es_en)
es_en_dataset = None

train_el_en = el_en_dataset[:train_num]
test_el_en = el_en_dataset[train_num:]
random.shuffle(test_el_en)
el_en_dataset = None

train_hi_en = hi_en_dataset[:train_num]
test_hi_en = hi_en_dataset[train_num:]
random.shuffle(test_hi_en)
hi_en_dataset = None

final_training_set = list()
for i in range(train_num):
    options = ['en_fr', 'en_es', 'en_el', 'en_hi', 'fr_en', 'es_en', 'el_en', 'hi_en']
    for j in range(8):
        choose = random.choice(options)
        if choose == 'en_fr':
            final_training_set.append({'text': train_en_fr[0]})
            train_en_fr.pop(0)
        elif choose == 'en_es':
            final_training_set.append({'text': train_en_es[0]})
            train_en_es.pop(0)
        elif choose == 'en_el':
            final_training_set.append({'text': train_en_el[0]})
            train_en_el.pop(0)
        elif choose == 'en_hi':
            final_training_set.append({'text': train_en_hi[0]})
            train_en_hi.pop(0)
        elif choose == 'fr_en':
            final_training_set.append({'text': train_fr_en[0]})
            train_fr_en.pop(0)
        elif choose == 'es_en':
            final_training_set.append({'text': train_es_en[0]})
            train_es_en.pop(0)
        elif choose == 'el_en':
            final_training_set.append({'text': train_el_en[0]})
            train_el_en.pop(0)
        else:
            final_training_set.append({'text': train_hi_en[0]})
            train_hi_en.pop(0)
        options.pop(options.index(choose)) # ensures no example type is shown twice in a row


# write the final training file
with open('final_training_set.txt','w') as outfile:
    json.dump(final_training_set, outfile)


# write final test file
final_test_set = test_en_fr + test_fr_en + test_en_es + test_es_en + test_en_el + test_el_en + test_en_hi + test_hi_en
with open('final_test_set.txt','w') as outfile:
    json.dump(final_test_set, outfile)


