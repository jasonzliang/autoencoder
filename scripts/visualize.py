#!/usr/bin/python
'''
Created on Nov 17, 2014

@author: jason
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,7))

def comparePerformance(myFile='script_data/ga_comparison.txt', myLabels=["SGD", "HGA", "CGA"], 
  x = [(1,17), (18,34), (35,51)], outFile="ga_comparison.png",
  xlog=False, ylog=False):
  linestyles = ['-', '--', ':', '-.'] * 5
  f = open(myFile)
  raw_data = f.readlines()
#   for item in raw_data[0:17]:
#     print item.rstrip()
  myList = []
  for s,e in x:
    time = [float(line.rstrip().split()[5]) for line in raw_data[s:e]]
    error = [float(line.rstrip().split()[8]) for line in raw_data[s:e]]
    myList.append((time, error))
  
  for i, d in enumerate(myList):
    time, error = d
    plt.plot(time, error, linewidth=1.5, linestyle=linestyles[i], label=myLabels[i])
  plt.xlabel("time in seconds")
  plt.ylabel("total error")
  plt.ylim(0,100000)
  ax = plt.gca()
  if xlog:
    ax.set_xscale('log')
  if ylog:
    ax.set_yscale('log')
  plt.legend(loc=1)
  plt.grid()
  plt.savefig(outFile, bbox_inches="tight", dpi=200)
  plt.clf()
  
def compareNumHiddenUnits(myFile='script_data/ga_comparison3_bigpopsize.txt', n=20
  ,myHidden = [100, 200, 400, 800, 1600], outfile='ga_comparison3_big.png'):
  f = open(myFile)
  raw_data = f.readlines()
  x = [(3*i, 3*i + 2) for i in xrange(n)]
  myList = []
  for s,e in x:
    time = [float(line.rstrip().split()[5]) for line in raw_data[s:e]]
#     print time[-1]
#     diff = [time[i+1] - time[i] for i in xrange(10)]
    myList.append(time[-1])

  plt.plot(myHidden, myList[0::4], marker='x', linewidth=2, linestyle='-', label='1 thread')
  plt.plot(myHidden, myList[1::4], marker='x', linewidth=2, linestyle='--', label='4 threads')
  plt.plot(myHidden, myList[2::4], marker='x', linewidth=2, linestyle=':', label='8 threads')
  plt.plot(myHidden, myList[3::4], marker='x', linewidth=2, linestyle='-.', label='16 threads')
  plt.xlabel("number of hidden units")
  plt.ylabel("time in seconds per outer iteration")
  plt.legend(loc=2)
  plt.xlim(0,myHidden[-1])
  plt.grid()
  plt.savefig(outfile, bbox_inches="tight", dpi=200)
  plt.clf()

def visualizeWeights():
  print "visualizing weights"
  f = open("script_data/experiment3.txt")
  data = f.readlines()
  plt.figure(figsize=(16,16))
  for i in xrange(1,101):
    plt.subplot(10,10,i)
    values = np.array([float(x) for x in data[i].rstrip().split()])
    values = np.subtract(values, min(values))
    values = np.divide(values, max(values))
#     print values
    values = np.reshape(values, (28,28))
    plt.axis('off')
    plt.imshow(values, cmap='binary', interpolation='nearest')
#   plt.tight_layout()
  plt.savefig("experiment3_1.png", bbox_inches="tight")
  plt.clf()

def visualizeReconstructedDigits():
  print "visualizing reconstructed digits"
  f = open("script_data/experiment3.txt")
  data = f.readlines()
  plt.figure(figsize=(16,16))
  for i in xrange(102,202):
    plt.subplot(10,10,i-101)
    values = np.array([float(x) for x in data[i].rstrip().split()])
    values = np.subtract(values, min(values))
    values = np.divide(values, max(values))
#     print values
    values = np.reshape(values, (28,28))
    plt.axis('off')
    plt.imshow(values, cmap='binary', interpolation='nearest')
#   plt.tight_layout()
  plt.savefig("experiment3_2.png", bbox_inches="tight")
  plt.clf()

  
if __name__ == '__main__':
  # comparePerformance(myFile="script_data/ga_comparison2.txt", 
  #   myLabels=["1 thread", "4 threads", "8 threads", "16 threads"], 
  #   x = [(1,17), (18,34), (35,51), (52,68)], 
  #   outFile="ga_comparison2.png", xlog=False, ylog=False)
  comparePerformance(myFile='script_data/ga_comparison4.txt', myLabels=["HGA-2", "HGA-4", "HGA-8", "HGA-16", "HGA-32"], 
  x = [(1,17), (18,34), (35,51), (52,68), (69,85)], outFile="ga_comparison4_1.png",
  xlog=False, ylog=False)
  comparePerformance(myFile='script_data/ga_comparison4.txt', myLabels=["CGA-2", "CGA-4", "CGA-8", "CGA-16", "CGA-32"], 
  x = [(86,102), (103,119), (120,136), (137,153), (154,170)], outFile="ga_comparison4_2.png",
  xlog=False, ylog=False)
