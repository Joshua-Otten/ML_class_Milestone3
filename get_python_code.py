
## first filter a dataset of code for Python code, with only the specified import modules
from datasets import load_dataset
import json

ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
count = 0
holder = {}
try:
    for example in ds:
        if example.get("language")=="Python":
            # the example is Python, now we need to make sure it doesn't have other imports
            code = example["code"]
            supported = True
            lines = code.split()
            for line in lines:
                if "import" in line:
                    if "pandas" not in line and "torch" not in line and "numpy" not in line and "random" not in line and "tensorflow" not in line:
                        # must be a lib we don't support
                        supported = False
            if supported:
                print(code)
                count += 1
                holder[count] = code
                print(count)
                print('\n')
                if count >= 100000:
                    # write it all to a file
                    with open("python_dataset.json", "w") as outfile:
                        json.dump(holder, outfile)
                    break
        
except:
    with open("python_dataset.json", "w") as outfile:
        json.dump(holder, outfile)

