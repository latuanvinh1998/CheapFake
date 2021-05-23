import json

f = open('mmsys_anns/public_test_mmsys_final.json')

label = []
for line in f:
	label.append(json.loads(line))

print(len(label))