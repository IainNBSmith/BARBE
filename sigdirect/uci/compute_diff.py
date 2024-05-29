import sys



def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(min(len(a), len(b)) ))



def get_all_diffs(name1, name2):
	with open(name1, 'r') as f1:
		d1 = f1.read().strip().split('\n')
	with open(name2, 'r') as f2:
		d2 = f2.read().strip().split('\n')

	print('len f1:',len(d1),'len f2:', len(d2))
	counter_1 = 0
	for i, (l1,l2) in enumerate(zip(d1,d2)):
		diff = diff_letters(l1,l2)
		if diff==0:
			continue
		if diff==1:
			counter_1 +=1
		if diff>1:
			print('line:',i,l1,l2)
	print(counter_1, 'lines had one different letter')


name = sys.argv[1]
index = sys.argv[2]

for i in range(1,11):
	name1 = name + '_tr' + str(i) + '.txt_my_output_' + str(index) + '.sorted'
	name2 = name + '_tr' + str(i) + 'rule.txt.sorted'
	print()
	print(i)
	get_all_diffs(name1, name2)
	