import csv
import numpy as np
import pandas as pd
import re
import json




with open("/home/aev21/data/CN/new_life_cos/images_with_uncertainty.json") as test:
      images_uncertainty_dict_read = json.load(test)

with open("/home/aev21/data/CN/new_life_cos/images_with_uncertainty2.json") as test:
      images_uncertainty_dict_read2 = json.load(test)

all_images = images_uncertainty_dict_read + images_uncertainty_dict_read2
all_images.sort(key=lambda x: x["avg"], reverse = True) 
with open("/home/aev21/data/CN/new_life_cos/sorted.json", "w") as test:
    json.dump(all_images, test)
    
with open("/home/aev21/data/CN/new_life_cos/sorted.json") as test:
      all_sorted = json.load(test)

for entry in all_sorted:
	print("new one   ")
	print(entry["name"])
"""
with open(csv2, "r") as f2:
	list2 = f2.readlines()
f2.close()


with open(csv3, "r") as f3:
	list3 = f3.readlines()
f3.close()


with open(csv4, "r") as f4:
	list4 = f4.readlines()
f4.close()"""

#list_all = np.concatenate((list1, list2, list3, list4), axis=0)
#list_all = np.concatenate((list1, list2), axis=0)
"""list_all = list1
separated = []
for img in list_all:

	img_path=img.split(" avg ",1)[0] 

	u_avg = img.split(" avg ",1)[1].split(" sum",1)[0]
	u_sum = img.split(" sum ",1)[1].split(" bbs ",1)[0]

	BBS = img.split(" bbs ",1)[1][:-1]

	separated.append([float(u_avg), BBS, img_path])


separated.sort(key = lambda x: x[0], reverse = True) 

#for i in separated:#
#	print(i)


with open("/home/aev21/data/CN/new_life_cos/avg_sorted.txt", "w") as f:
	for img in separated:
		f.write(img[2]+"\n")
f.close()

with open("/home/aev21/data/CN/new_life_cos/avg_sorted_with_uncertainties.txt", "w") as f:
	for img in separated:
		f.write(img[2] +" uncertainty "+ str(img[0])+" bbs "+ str(img[1])+"\n")
f.close()"""