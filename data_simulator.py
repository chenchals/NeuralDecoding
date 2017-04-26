import random
import csv
num_samples = 20000
counter = 0
idx = 0
reset = 0
num_features = 2
num_classes = 2
with open('dat.csv', 'w+') as f:
	spam_writer = csv.writer(f, delimiter=' ')
	for _ in range(num_samples):
		if reset == 0:
			idx += 1
			reset = random.randint(1,70)
		counter += 1
		reset -= 1
		features = [random.normalvariate(0, 1) for _ in range(num_features)]
		label = random.randint(0, num_classes-1)
		sample = [idx]+features+[label]
		spam_writer.writerow(sample)
