import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

def visualize_channel():
    
    x = cPickle.load(open("../data/s01"+".dat", 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
      field_names.append(key)

    labels = x[field_names[0]]
    data = x[field_names[1]]
    
    lst = [0, 16, 2, 24]
    dat = []
    for i in range(40):
        tmp = []
        for j in lst:
            tmp.append(data[i][j])
        dat.append(tmp)
    dat = np.array(dat)
    feature = []
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("High - Arousal || High - Valence", fontsize=15 ,color = 'purple')
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    lst = [ax1,ax2,ax3,ax4]
    cls = ['r','g','b','k']
    chan = ['FP1','FP2','F3','C4']
    i = 0
    channelNames = []
    for ch_idx in range(dat.shape[1]):
        channel_data = dat[:, ch_idx, :]  # select data for current channel
        lst[i].plot(channel_data[19][1500:1601],cls[i])
        lst[i].set_title(chan[i],color=cls[i])
        i = i + 1
    
    plt.show();
        

def visualize_activity(X_train,Y_train):
    
    hv_ha = []
    hv_la = []
    lv_ha = []
    lv_la = []
    
    
    for i in range(X_train.shape[0]):
        if Y_train[i][0] == 1 and Y_train[i][1] == 1:
            hv_ha.append(X_train[i][0])
        elif Y_train[i][0] == 1 and Y_train[i][1] == 0 :
            hv_la.append(X_train[i][0])
        elif Y_train[i][0] == 0 and Y_train[i][1] == 1 :
            lv_ha.append(X_train[i][0])
        else:
            lv_la.append(X_train[i][0])
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("Activity", fontsize=15 ,color = 'purple')
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(lv_ha,'r')
    ax1.set_title("Low Valence | High Arousal")
    ax2.plot(hv_ha,'g')
    ax2.set_title("High Valence | High Arousal")
    ax3.plot(lv_la,'k')
    ax3.set_title("Low Valence | Low Arousal")
    ax4.plot(hv_la,'b')
    ax4.set_title("High Valence | Low Arousal")
    plt.show()
    
    
def visualize_mobility(X_train,Y_train):
    
    hv_ha = []
    hv_la = []
    lv_ha = []
    lv_la = []
    
    
    for i in range(X_train.shape[0]):
        if Y_train[i][0] == 1 and Y_train[i][1] == 1:
            hv_ha.append(X_train[i][1])
        elif Y_train[i][0] == 1 and Y_train[i][1] == 0 :
            hv_la.append(X_train[i][1])
        elif Y_train[i][0] == 0 and Y_train[i][1] == 1 :
            lv_ha.append(X_train[i][1])
        else:
            lv_la.append(X_train[i][1])
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("Mobility", fontsize=15 ,color = 'purple')
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(lv_ha,'r')
    ax1.set_title("Low Valence | High Arousal")
    ax2.plot(hv_ha,'g')
    ax2.set_title("High Valence | High Arousal")
    ax3.plot(lv_la,'k')
    ax3.set_title("Low Valence | Low Arousal")
    ax4.plot(hv_la,'b')
    ax4.set_title("High Valence | Low Arousal")
    plt.show()
    
    
def visualize_complexity(X_train,Y_train):
    
    hv_ha = []
    hv_la = []
    lv_ha = []
    lv_la = []
    
    
    for i in range(X_train.shape[0]):
        if Y_train[i][0] == 1 and Y_train[i][1] == 1:
            hv_ha.append(X_train[i][2])
        elif Y_train[i][0] == 1 and Y_train[i][1] == 0 :
            hv_la.append(X_train[i][2])
        elif Y_train[i][0] == 0 and Y_train[i][1] == 1 :
            lv_ha.append(X_train[i][2])
        else:
            lv_la.append(X_train[i][2])
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("Complexity", fontsize=15 ,color = 'purple')
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(lv_ha,'r')
    ax1.set_title("Low Valence | High Arousal")
    ax2.plot(hv_ha,'g')
    ax2.set_title("High Valence | High Arousal")
    ax3.plot(lv_la,'k')
    ax3.set_title("Low Valence | Low Arousal")
    ax4.plot(hv_la,'b')
    ax4.set_title("High Valence | Low Arousal")
    plt.show()