# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:50:02 2020

@author: Roma
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd 
import seaborn as sns


#%%

def draw_matrix(t, figsize=10, drop_nulss=True, save=None, classes=['Патология', 'Норма'], norm=True):
  if type(t) != np.array:
    t = np.array(t)
  if len(classes) != t.shape[0] and len(classes) != t.shape[0] :
    print('Lenght of classe less than matrix')
    classes = range(t.shape[0])

  fig, ax = plt.subplots(figsize=(figsize, figsize))
  cm = pd.DataFrame(data=t, index=classes, columns=classes)
  cm.index.name = 'Actual'
  cm.columns.name = 'Predicted'
  
  sumed = np.sum(t, axis=0)
  per = ["{0:.2%}".format(v) for v in (t / sumed).flatten()]
  
  co = t.flatten().astype(np.str)
  if drop_nulss:
    co = list(map(lambda x: '' if x == '0.0' else x, co))
    per = list(map(lambda x: '' if x == '0.00%' else x, per))

  labels = np.asarray([f"{v2}\n{v1}" for v1, v2 in zip(co, per)]).reshape(t.shape)   
  #fig = 
  sns.heatmap(t / sumed, annot=labels, vmin=0.0, vmax=1.0, fmt='', cmap='Blues', ax=ax)   
  #print(save, isinstance(save, str))
  if save != None:
    if not isinstance(save, str):
      raise Exception('Need get name or path with name')
    else:
      ax.get_figure().savefig(save)
  return fig, ax 

#%%
##############
##  HOW TO USE
##############
  
#
#t = draw_matrix(ma1, figsize=10, drop_nulss=True, save='C:\\Users\\Mi\\matrix.jpg')
#
#t[1].get_figure().savefig('C:\\Users\\Mi\\matrix.jpg')

