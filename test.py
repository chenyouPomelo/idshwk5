from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist = []
class Domain:
	def __init__(self,_name,_label,_length,_numbers,_entropy):
		self.name = _name
		self.label = _label
		self.length=_length
		self.numbers=_numbers
		self.entropy=_entropy
		

	def returnData(self):
		return [self.length, self.numbers, self.entropy]

	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1
		
def initData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = tokens[1]
			length= len(name)
			num=0
			entro=0     #熵
			result={}   #每个字符出现的次数
			frequency={}#每个字符出现的频率
			for i in name:
                            result[i]=name.count(i)
                            frequency[i]=float(result[i])/length
                            entro-=1*(frequency[i])*math.log(frequency[i],2)
                            if i.isdigit():
                                num = num + 1
			numbers= num  
			entropy = entro
			domainlist.append(Domain(name,label,length,numbers,entropy))

def main():
	initData("train.txt")
	featureMatrix = []
	labelList = []
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())

	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)

	file1=open("test.txt")
	file2=open("result.txt","w")
	for line in file1:
		line = line.strip()
		if line.startswith("#") or line =="":
			continue
		tokens_test = line.split(",")
		name_test = tokens_test[0]
		length_test = len(name_test)
		num_test=0
		entro_test=0     #熵
		result_test={}   #每个字符出现的次数
		frequency_test={}#每个字符出现的频率
		for i in name_test:
                	result_test[i]=name_test.count(i)
                	frequency_test[i]=float(result_test[i])/length_test
                	entro_test-=1*(frequency_test[i])*math.log(frequency_test[i],2)
                	if i.isdigit():
                	    num_test +=1
		numbers_test= num_test  
		entropy_test = entro_test
		if clf.predict([[length_test,numbers_test,entropy_test]])==0:
		    label_test1 = "notdga"
		    print(name_test,label_test1,file=file2)
		if clf.predict([[length_test,numbers_test,entropy_test]])==1:
		    label_test2 = "dga"
		    print(name_test,label_test2,file=file2)

	file1.close
	file2.close

if __name__ == '__main__':
	main()

