import csv
import numpy as np
import pandas as pd
import re


csv1 = "/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/images_with_uncertainty.txt"
csv2 = "/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/images_with_uncertainty2.txt"
csv3 = "/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/images_with_uncertainty3.txt"
csv4 = "/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/images_with_uncertainty4.txt"


with open(csv1, "r") as f1:
	list1 = f1.readlines()
f1.close()


with open(csv2, "r") as f2:
	list2 = f2.readlines()
f2.close()


with open(csv3, "r") as f3:
	list3 = f3.readlines()
f3.close()


with open(csv4, "r") as f4:
	list4 = f4.readlines()
f4.close()

list_all = np.concatenate((list1, list2, list3, list4), axis=0)
#list_all = np.concatenate((list1, list2), axis=0)

separated = []
for img in list_all:

	img_path=img.split(" max",1)[0] 
	uncertainties=img.split("max ",1)[1] 
	u_max = uncertainties.split(" min",1)[0]
	#u_min = uncertainties.split(" min ",1)[1].split(" avg",1)[0]
	#u_avg = uncertainties.split(" avg ",1)[1].split(" sum",1)[0]
	#u_sum = uncertainties.split(" sum ",1)[1].split(" max_trace",1)[0]

	#u_max_trace = uncertainties.split(" max_trace ",1)[1].split(" min_trace",1)[0]
	#u_min_trace = uncertainties.split(" min_trace ",1)[1].split(" avg_trace",1)[0]
	#u_avg_trace = uncertainties.split(" avg_trace ",1)[1].split(" sum_trace",1)[0]
	#u_sum_trace = uncertainties.split(" sum_trace ",1)[1]


	separated.append([float(u_max), img_path])


separated.sort(key = lambda x: x[0], reverse = True) 

#for i in separated:#
#	print(i)


with open("/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/max_sorted.txt", "w") as f:
	for img in separated:
		f.write(img[1]+"\n")
f.close()

with open("/home/aev21/data/CN/txt_files/images_with_uncertainty_after_init/max_sorted_with_uncertainties.txt", "w") as f:
	for img in separated:
		f.write(img[1] +" "+ str(img[0])+"\n")
f.close()