# import pandas as pd

# label = pd.read_csv("objectInfo150.csv")

# print(label['Name'][1])

f = open('label.txt', 'r')

txt = f.readlines()

for lines in txt:

	line = str(lines.rstrip('\n').split(';'))

	print(line)