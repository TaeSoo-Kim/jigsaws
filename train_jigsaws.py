import numpy as np
import os
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils import to_categorical
from keras.regularizers import l2,l1
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasRegressor
import Models


from data_utils import train_gen, test_gen
import sys
import random
import pdb


SPLIT = 8 # 1-5 for LOSO,  1-8 for LOUO
cross_val_mode ='LOUO'  #'LOSO' # or 'LOUO'
task = 'Suturing' # ['Knot_Tying','Suturing','NeedlePassing']

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

n_classes = num_classes[task]

SAMPLE_NUMS = {
  'Knot_Tying': {
    'LOSO' : {
      1: (300,71),
      2: (302,69),
      3: (292,79),
      4: (293,78),
      5: (297,74)
    },
    'LOUO' : {
      1: (334,37),
      2: (319,52),
      3: (316,55),
      4: (317,54),
      5: (321,50),
      6: (319,52),
      7: (343,28),
      8: (328,43)
    }
  },
  'Needle_Passing': {
    'LOSO' : {
      1: (412,106),
      2: (440,78),
      3: (402,116),
      4: (398,120),
      5: (420,98)
    },
    'LOUO' : {
      1: (451,67),
      2: (411,107),
      3: (408,110),
      4: (441,77),
      5: (464,54),
      6: (518,0), # train only?
      7: (490,28),
      8: (443,75)
    }
  },
  'Suturing': {
    'LOSO' : {
      1: (603,190),
      2: (660,133),
      3: (635,158),
      4: (635,158),
      5: (639,154)
    },
    'LOUO' : {
      1: (691,102),
      2: (697,96),
      3: (679,114),
      4: (701,92),
      5: (703,90),
      6: (677,116), # train only?
      7: (707,86),
      8: (696,97)
    }
  }
  

}


def train():
  lr = 0.001
  loss = "categorical_crossentropy"
  out_dir_name = task+'_'+cross_val_mode+'_split'+str(SPLIT)
  dropout = 0.5
  epochs = 100
  plateau_threhsold = 30
  feature_dim = 76
  max_len = max_lengths[task]
  num_train  = SAMPLE_NUMS[task][cross_val_mode][SPLIT][0]
  num_test = SAMPLE_NUMS[task][cross_val_mode][SPLIT][1]
  
  model = Models.ResTCN_Gesture(
           n_classes,
           feature_dim,
           max_len,
           gap=1,
           dropout=dropout,
           kernel_regularizer=l2(1.e-4),
           activation="relu")

  
  #optimizer = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=True)
  optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  #optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
  model.compile(optimizer,
                loss=loss,
                metrics=['accuracy'])

  if not os.path.exists('JIGSAWS_GESTURE_weights/'+out_dir_name):
    os.makedirs('JIGSAWS_GESTURE_weights/'+out_dir_name)
  
  weight_path = 'JIGSAWS_GESTURE_weights/'+out_dir_name+'/{epoch:03d}_{val_acc:0.3f}.hdf5'
  checkpoint = ModelCheckpoint(weight_path, 
                               monitor='val_acc', 
                               verbose=1, 
                               save_best_only=True, mode='max')
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1,
                                patience=30, 
                                verbose=1,
                                mode='auto',
                                cooldown=3,
                                min_lr=0.00001)
  

  callbacks_list = [checkpoint,reduce_lr]

  model.fit_generator(train_gen(split=SPLIT, cross_val_mode=cross_val_mode, task=task),
                      num_train,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[checkpoint,reduce_lr], 
                      validation_data=test_gen(split=SPLIT, cross_val_mode=cross_val_mode, task=task),
                      validation_steps=num_test, 
                      class_weight=None, 
                      workers=1,  
                      initial_epoch=0)


if __name__ == "__main__":
  train()
