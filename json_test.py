import json

with open('untitled.json', 'r') as myfile:
    f = myfile.read()

data = json.loads(f)
print(str(data))
