file1 = open('UCSD_train.txt','r')
file2 = open('p1.txt','a')
file3 = open('p2.txt','a')
for line in file1.readlines():
	tmp1 = ''
	tmp2 = ''
	length =len(line)
	for i in range(length):
		if line[i] == 'g':
			tmp1 = '"'+line[:i+1]+'"'+':'
			tmp2 = line[i+2:]
			file2.write(tmp1+'\n')
			file3.write(','+tmp2)
			
file4 = open('p2.txt','r')
file5 = open('p2_1.txt','a')
str = ''
for line in file4.readlines():
	# 所有空格替换成空格+逗号
	str = line.replace(' ',', ')
	file5.write(str)
	
file6 = open('p2_1.txt','r')
file7 = open('p2_2.txt','a')
for line in file6.readlines():
	length = len(line)
	tmp = ''
	k = 0
	line = list(line)
	for i in range(length):
		if line[i] == ',':
			k = k+1
			if k%5 ==0: 
			#插入右括号')'
				line.insert(i,')')	
	tmp = ''.join(line)
	file7.write(tmp)
	
file8 = open('p2_2.txt','r')
file9 = open('p2_3.txt','a')
for line in file8.readlines():
	tmp = ''
	tmp = '('+line[1:]
	file9.write(tmp)
	
file10 = open('p2_3.txt','r')
file11 = open('p2_4.txt','a')
for line in file10.readlines():
	length = len(line)
	tmp = ''
	k = 0
	line = list(line)
	for i in range(length):
		if line[i] == ',':
			k = k+1
			if k%4 ==0: 
			#插入左括号')'
				line.insert(i+2,'(')	
	tmp = ''.join(line)
	file11.write(tmp)

file12 = open('p1.txt','r')
file13 = open('p2_4.txt','r')
file14 = open('p3.txt','a')

list1 = []
n1 = 0
for linex in file12.readlines():
	n1 = n1+1
	s1 = linex.strip()
	list1.append(s1)
file12.close()

list2 = []
n2 = 0
for liney in file13.readlines():
	n2 = n2+1
	s2 = liney.strip()
	list2.append(s2)
file13.close()

for i in range(n1):
	s3 = list1[i]+' '+list2[i]
	file14.write(s3+'\n')
file14.close()
