import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random

feat_dim = 76
SPLIT = 1 # 1-5 for LOSO,  1-8 for LOUO

cross_val_mode = 'LOUO' #'LOSO' # or 'LOUO'

labels_table = {
  'Knot_Tying': {'G1':0, 'G11':1, 'G12':2, 'G13':3, 'G14':4, 'G15':5},
  'Suturing': {'G1':0, 'G2':1, 'G3':2, 'G4':3, 'G5':4, 'G6':5,'G8':6,'G9':7,'G10':8,'G11':9},
  'Needle_Passing': {'G1':0, 'G2':1, 'G3':2, 'G4':3, 'G5':4, 'G6':5,'G8':6,'G11':7},
}
num_classes = {
  'Knot_Tying': 6,
  'Suturing':  10,
  'Needle_Passing': 8
}
max_lengths = {
  'Knot_Tying': 900,
  'Suturing':  1200,
  'Needle_Passing': 3000
}


def train_gen(split=1,cross_val_mode='LOSO',task='Knot_Tying'):
  max_length = max_lengths[task]
  n_classes = num_classes[task]
  kinematics_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/'+task+'/kinematics/AllGestures'
  if cross_val_mode == 'LOSO':
    label_file_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/Experimental_setup/'+task+'/unBalanced/GestureClassification/SuperTrialOut/'
  elif cross_val_mode == 'LOUO':
    label_file_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/Experimental_setup/'+task+'/unBalanced/GestureClassification/UserOut/'
  fold_one = os.path.join(label_file_root,str(split)+'_Out/itr_1')

  files_to_data = {}
  kinematic_files = os.listdir(kinematics_root)
  for kf in kinematic_files:
    kinematics_file = open(os.path.join(kinematics_root, kf), 'r')
    X = []
    for k_line in kinematics_file:
      timestep_feat = [float(k) for k in k_line.split()]
      X.append(timestep_feat)
    X = np.array(X)

    kinematics_file.close()
    files_to_data[kf] = X

  train_split = open(os.path.join(fold_one,'Train.txt'),'r')
  train_data = []

  lengths = []
  for line in train_split:
    data, label = line.split()
    indices = data[-17:-4].split('_')
    from_ = int(indices[0])
    to_ = int(indices[1])

    file_name = data[:-18]+'.txt'
    y = labels_table[task][label]
    x_sample = files_to_data[file_name][from_:to_]

    X = np.zeros((1,max_length,feat_dim ))
    X[0,:x_sample.shape[0]] = x_sample
    Y = np.zeros((1,n_classes))
    Y[0, y] = 1
    train_data.append((X, Y))

  train_split.close()

  while True:
    random.shuffle(train_data)
    for x,y in train_data:
      yield (x,y)

def test_gen(split=1,cross_val_mode='LOSO',task='Knot_Tying'):
  max_length = max_lengths[task]
  n_classes = num_classes[task]
  kinematics_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/'+task+'/kinematics/AllGestures'
  if cross_val_mode == 'LOSO':
    label_file_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/Experimental_setup/'+task+'/unBalanced/GestureClassification/SuperTrialOut/'
  elif cross_val_mode == 'LOUO':
    label_file_root = '/home-4/tkim60@jhu.edu/scratch/data/JIGSAWS/Experimental_setup/'+task+'/unBalanced/GestureClassification/UserOut/'
  fold_one = os.path.join(label_file_root,str(split)+'_Out/itr_1')

  files_to_data = {}
  kinematic_files = os.listdir(kinematics_root)
  for kf in kinematic_files:
    kinematics_file = open(os.path.join(kinematics_root, kf), 'r')
    X = []
    for k_line in kinematics_file:
      timestep_feat = [float(k) for k in k_line.split()]
      X.append(timestep_feat)
    X = np.array(X)

    kinematics_file.close()
    files_to_data[kf] = X

  train_split = open(os.path.join(fold_one, 'Test.txt'), 'r')
  train_data = []

  lengths = []
  for line in train_split:
    data, label = line.split()
    indices = data[-17:-4].split('_')
    from_ = int(indices[0])
    to_ = int(indices[1])

    file_name = data[:-18]+'.txt'
    y = labels_table[task][label]
    x_sample = files_to_data[file_name][from_:to_]

    X = np.zeros((1,max_length,feat_dim ))
    X[0,:x_sample.shape[0]] = x_sample
    Y = np.zeros((1,n_classes))
    Y[0, y] = 1
    train_data.append((X, Y))

  train_split.close()

  while True:
    random.shuffle(train_data)
    for x,y in train_data:
      yield (x,y)



if __name__ == "__main__":
  train_gen(split=1, cross_val_mode='LOSO')
