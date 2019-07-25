import numpy as np 
from glob import glob 
import os
import random

label_para = []
kind = '4'  #train, test, val, aug_train, aug_test, aug_val
img_list1 = os.listdir(r'./aug/aug'+kind+'/inf/') 
img_path1 = './aug/aug'+kind+'/inf/'

img_list2 = os.listdir(r'./aug/aug'+kind+'/nor/') 
img_path2 = './aug/aug'+kind+'/nor/'

img_list3 = os.listdir(r'./aug/aug'+kind+'/poly/') 
img_path3 = './aug/aug'+kind+'/poly/'

img_list4 = os.listdir(r'./aug/aug'+kind+'/vas/') 
img_path4 = './aug/aug'+kind+'/vas/'

img_list1_0 = [img_path1+img_list1[i] for i in range(len(img_list1))]
img_list2_0 = [img_path2+img_list2[i] for i in range(len(img_list2))]
img_list3_0 = [img_path3+img_list3[i] for i in range(len(img_list3))]
img_list4_0 = [img_path4+img_list4[i] for i in range(len(img_list4))]
###inflammatory : 2   normal : 1   polyp : 4   vascularlesion : 3

train_path = open('./aug'+kind+'_txt.txt','w')

print(len(img_list1),len(img_list2),len(img_list3),len(img_list4))

img_label1 = np.array([int(2) for i in range(len(img_list1))])
img_label2 = np.array([int(1) for i in range(len(img_list2))])
img_label3 = np.array([int(4) for i in range(len(img_list3))])
img_label4 = np.array([int(3) for i in range(len(img_list4))])

img_list_0  = np.concatenate((img_list1_0,img_list2_0,img_list3_0,img_list4_0),axis=0)
labels = np.concatenate((img_label1,img_label2,img_label3,img_label4),axis=0)
index = [i for i in range(len(img_list1)+len(img_list2)+len(img_list3)+len(img_list4))]
random.shuffle(index)
    
counter = 0  
for i in index[0:int(len(index))]:    
    train_path.write(img_list_0[i]+' '+str(labels[i])+' \n')
    counter += 1
print(counter)
train_path.close()
