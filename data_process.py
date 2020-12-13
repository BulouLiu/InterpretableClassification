import os
import numpy as np

lines = open('./data/connect-4.data', 'r')
feature_l = []
label = []

for line in lines:
    feature = []
    str_list = line.split(',')
    for item in str_list[0:42]:
        if item == 'x':
            feature.append(1)
        elif item == 'o':
            feature.append(-1)
        else:
            feature.append(0)
    if str_list[42][0] == 'w':
        label.append(1)
    elif str_list[42][0] == 'l':
        label.append(-1)
    else:
        label.append(0)
    feature_l.append(feature)

fold_l = ['0', '1', '2', '3', '4']
feature_split = []
label_split = []
for i in range(5):
    feature_split.append([])
    label_split.append([])

for j in range(len(label)):
    feature_split[j%5].append(feature_l[j])
    label_split[j%5].append(label[j])

for i in range(5):
    if not os.path.exists('./data/' + fold_l[i]):
        os.makedirs('./data/' + fold_l[i])
    np.savetxt('./data/' + fold_l[i] + '/feature.txt', feature_split[i])
    np.savetxt('./data/' + fold_l[i] + '/label.txt', label_split[i])

# 读取数据方法
print(np.loadtxt('./data/' + fold_l[0] + '/feature.txt'))
print(np.loadtxt('./data/' + fold_l[0] + '/label.txt'))
