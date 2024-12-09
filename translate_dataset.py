####
# Translates code examples to other languages and creates a file for each
####
import subprocess
import json

languages = ['French', 'Spanish', 'Greek', 'Hindi']

for language in languages:
    print('starting',language)
    count = 1
    with open('python_dataset.json','r') as f:
        examples = json.load(f)
    trans_examples = {}
    for key in examples:
        code = examples[key]
        #print(code)
        # gotta write this to code1.unipy
        with open('code1.unipy','w') as f:
            f.write(code)
        result = subprocess.run(['python','StringCodeTranslator.py','code1.unipy','English', language, 'unipy'], capture_output=True, text=True)
        output = str(result.stdout)
        to_save = ''
        i = 0
        while i < len(output)-1:
            if output[i]=='\u202a' or output[i]=='\u202c':
                i += 1
                #print('found something')
            else:
                to_save += output[i]
                i += 1
        #print(output)
        #print(to_save.strip())
        #print('\n')
        trans_examples[count] = to_save.strip()
        count += 1
        if count%100 == 0: #if count > 100000:
            print(count)
        if count > 20000:
            #print(trans_examples)
            with open(language+'_code_examples.json','w') as outfile:
                json.dump(trans_examples, outfile)
            break
