'''
Created on Nov 17, 2014

@author: jason
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,7))

def experiment1():
  linestyles = ['-', '--', ':']
  f = open('script_data/experiment1.txt')
  raw_data = f.readlines()
#   for item in raw_data[0:17]:
#     print item.rstrip()
  x = [(0,16), (17,33), (34,50)]
  myLabels = ["1 thread", "4 threads", "8 threads"]
  myList = []
  for s,e in x:
    time = [float(line.rstrip().split()[5]) for line in raw_data[s:e]]
    error = [float(line.rstrip().split()[8]) for line in raw_data[s:e]]
    myList.append((time, error))
  
  for i, d in enumerate(myList):
    time, error = d
    plt.plot(time, error, marker='x', linewidth=2, linestyle=linestyles[i], label=myLabels[i])
  plt.xlabel("time in seconds")
  plt.ylabel("total error")
  plt.legend(loc=1)
  plt.grid()
  plt.savefig("experiment1.png", bbox_inches="tight", dpi=200)
  plt.clf()

  
def experiment2():
  f = open('script_data/experiment2_v2.txt')
  raw_data = f.readlines()
  x = [(3*i, 3*i + 2) for i in xrange(12)]
  myHidden = [100, 200, 500, 800]
  myList = []
  for s,e in x:
    time = [float(line.rstrip().split()[5]) for line in raw_data[s:e]]
#     print time[-1]
#     diff = [time[i+1] - time[i] for i in xrange(10)]
    myList.append(time[-1])

  plt.plot(myHidden, myList[0::3], marker='x', linewidth=2, linestyle='-', label='1 thread')
  plt.plot(myHidden, myList[1::3], marker='x', linewidth=2, linestyle='--', label='4 threads')
  plt.plot(myHidden, myList[2::3], marker='x', linewidth=2, linestyle=':', label='8 threads')
  plt.xlabel("number of hidden units")
  plt.ylabel("time in seconds per outer iteration")
  plt.legend(loc=2)
  plt.xlim(0,900)
  plt.grid()
  plt.savefig("experiment2.png", bbox_inches="tight", dpi=200)
  plt.clf()

def experiment3():
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

def experiment3_v2():
  f = open("script_data/experiment3.txt")
  data = f.readlines()
  plt.figure(figsize=(16,16))
  for i in xrange(0,100):
    plt.subplot(10,10,i+1)
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
  experiment1()
  experiment2()
