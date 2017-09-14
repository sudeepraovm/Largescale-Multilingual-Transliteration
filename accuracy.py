

f = open('test_res2_1', 'r').read()
ctr = 0
tot = 0
for line in f.strip().split('\n'):
	tot += 1
	pred = line.split('?')[0][1:]
	true = line.split('?')[1]
	if(pred == true):
		ctr += 1

print float(ctr)/tot		