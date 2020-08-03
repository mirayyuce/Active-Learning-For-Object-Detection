import os
import numpy as np
path = "/media/aev21/d15d0282-a376-47ba-bdf5-855a03695f32/data/CN/"
path_out = "/home/aev21/data/CN/2sequences_init.txt"
path_out_train = "/home/aev21/data/CN/2sequences_train.txt"
path_out_rest = "/home/aev21/data/CN/2sequences_rest.txt"
path_out_val = "/home/aev21/data/CN/2sequences_val.txt"
cut = "/home/aev21/data/CN/"


seq_to_use = []
seq_to_use_val = []
seq_to_use_init = []
seq_lef_over = []
num_frames = 0
all_sequences = np.random.permutation(os.listdir(path))

for seq in all_sequences:
	seq_folder = path + "/" + str(seq) + "/images/"
	num_frames += len(os.listdir(seq_folder))
	if num_frames < 68000:
		seq_to_use.append(seq)
	elif 68000 <= num_frames < 75000:
		seq_to_use_init.append(seq)
	elif num_frames <= 83000:
		seq_to_use_val.append(seq)
	else:
		seq_lef_over.append(seq)
print(len(seq_to_use_val), len(seq_to_use_init), len(seq_to_use), len(seq_lef_over))

# for init set

with open(path_out, "a+") as f_out:
	for s in seq_to_use_init:
		
		f_out.write("%s\n"%s)
f_out.close()

num_frames = 0
with open(path_out, "r") as f:

	x = f.readlines()
	for line in x:
		seq_name = line[:-1]
		print(seq_name)
		seq_folder =  path + "/" + str(seq_name) + "/images/"
		#print(os.listdir(seq_folder))
		num_frames += len(os.listdir(seq_folder))
f.close()
print(num_frames)

# for train set
with open(path_out_train, "a+") as f_out:
	for s in seq_to_use:
		
		f_out.write("%s\n"%s)
f_out.close()

num_frames = 0
with open(path_out_train, "r") as f:

	x = f.readlines()
	for line in x:
		seq_name = line[:-1]
		#print(seq_name)
		seq_folder = path + "/" + str(seq_name) + "/images/"
		#print(os.listdir(seq_folder))
		num_frames += len(os.listdir(seq_folder))
f.close()
print(num_frames)

with open(path_out_val, "a+") as f_out:
	for s in seq_to_use_val:
		
		f_out.write("%s\n"%s)
f_out.close()

num_frames = 0
with open(path_out_val, "r") as f:

	x = f.readlines()
	for line in x:
		seq_name = line[:-1]
		#print(seq_name)
		seq_folder = path + "/" + str(seq_name) + "/images/"
		#print(os.listdir(seq_folder))
		num_frames += len(os.listdir(seq_folder))
f.close()
print(num_frames)

with open(path_out_rest, "a+") as f_out:
	for s in seq_lef_over:
		
		f_out.write("%s\n"%s)
f_out.close()
num_frames = 0
with open(path_out_rest, "r") as f:

	x = f.readlines()
	for line in x:
		seq_name = line[:-1]
		#print(seq_name)
		seq_folder = path + "/" + str(seq_name) + "/images/"
		#print(os.listdir(seq_folder))
		num_frames += len(os.listdir(seq_folder))
f.close()
print(num_frames)
"""for line in x:
				seq_name = line[:-1]
				#print(seq_name)
				seq_folder = "/media/mirayyuce/Elements/CN/train/" + str(seq_name) + "/images/"
				#print(os.listdir(seq_folder))
				num_frames += len(os.listdir(seq_folder))
				if num_frames < 75000:
					seq_to_use.append(seq_folder)
				elif 75000 < num_frames < 80000:
					seq_to_use_init.append(seq_folder)
				elif num_frames <= 88000:
					seq_to_use_val.append(seq_folder)
			print(len(seq_to_use_val), len(seq_to_use_init), len(seq_to_use))
		f.close()
		
		with open(path_out, "a+") as f_out:
			for s in seq_to_use_init:
				
				f_out.write("%s\n"%s[len(cut):-8])
		f_out.close()
		
		num_frames = 0
		with open(path_out, "r") as f:
		
			x = f.readlines()
			for line in x:
				seq_name = line[:-1]
				#print(seq_name)
				seq_folder = "/media/mirayyuce/Elements/CN/train/" + str(seq_name) + "/images/"
				#print(os.listdir(seq_folder))
				num_frames += len(os.listdir(seq_folder))
		f.close()
		print(num_frames)
		
		with open(path_out_train, "a+") as f_out:
			for s in seq_to_use:
				
				f_out.write("%s\n"%s[len(cut):-8])
		f_out.close()
		
		num_frames = 0
		with open(path_out_train, "r") as f:
		
			x = f.readlines()
			for line in x:
				seq_name = line[:-1]
				#print(seq_name)
				seq_folder = "/media/mirayyuce/Elements/CN/train/" + str(seq_name) + "/images/"
				#print(os.listdir(seq_folder))
				num_frames += len(os.listdir(seq_folder))
		f.close()
		print(num_frames)
		
		with open(path_out_val, "a+") as f_out:
			for s in seq_to_use_val:
				
				f_out.write("%s\n"%s[len(cut):-8])
		f_out.close()
		
		num_frames = 0
		with open(path_out_val, "r") as f:
		
			x = f.readlines()
			for line in x:
				seq_name = line[:-1]
				#print(seq_name)
				seq_folder = "/media/mirayyuce/Elements/CN/train/" + str(seq_name) + "/images/"
				#print(os.listdir(seq_folder))
				num_frames += len(os.listdir(seq_folder))
		f.close()
		print(num_frames)"""