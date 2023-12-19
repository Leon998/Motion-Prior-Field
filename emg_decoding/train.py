import pytrigno
import numpy as np
import matplotlib.pyplot as plt
import socket
import _thread
import time
import keyboard
import csv
import os
from process import EMGAnalysis
import argparse


class Emg_record():
    def __init__(self, subject, lowCh, highCh, fs, host):
        self.fs = fs
        self.dev = pytrigno.TrignoEMG(channel_range = (lowCh, highCh), samples_per_read = 27, host = host, buffered = True)
        self.channel_num = highCh - lowCh + 1
        self.save_path = 'emg_decoding/' + str(subject) + '_data/'
        self.dataFileName = self.save_path + 'data'
        self.triggerFileName = self.save_path + 'trigger'
        self.triggers = {}
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def record_with_plot(self):
        """
        检测emg采集是否正常
        """
        self.plot_time = 10  # 多久刷新一次画面
        t = 0
        self.ax = [0]*self.channel_num
        line = [0]*self.channel_num
        fig=plt.figure()
        for i in range(self.channel_num):
            self.ax[i] = fig.add_subplot(2,int(self.channel_num/2),i+1)
            self.ax[i].set_xlabel('time/s')
            self.ax[i].set_title('channel'+str(i+1))
            line[i] = self.ax[i].plot(0,0)[0]
        plt.ion()  #interactive mode on
        LOOP_FLAG = True
        while LOOP_FLAG:
            print(t)
            t0 = time.time()
            while self.dev.buffer.shape[1]<self.fs*self.plot_time*(t+1): # dev.buffer.shape[1]代表sample数量
                self.trigger_listener('0')
                self.trigger_listener('1')
                self.trigger_listener('2')
                self.trigger_listener('3')
                self.trigger_listener('4')
                self.trigger_listener('5')
                if keyboard.is_pressed('esc'):
                    LOOP_FLAG = False
                    break
                for i in range(self.channel_num):
                    line[i].set_ydata(self.dev.buffer[i])
                    line[i].set_xdata(np.arange(self.dev.buffer.shape[1])/self.fs)
                    self.ax[i].set_xlim((t*10,t*10+10))
                    self.ax[i].set_ylim((-0.0005,0.0005))
                plt.pause(0.001)
            t += 1
            print(time.time() - t0)
        self.save()
        self.dev.stop()

    def trigger_listener(self, action):
        if keyboard.is_pressed(action):
            self.ax[-2].set_xlabel('TRIGGERING' + action)
            time.sleep(0.5)
            triggerTime = self.dev.buffer.shape[1]
            self.triggers[triggerTime] = action
            print(self.triggers)


    def save(self):
        np.savetxt(self.dataFileName + '.csv', self.dev.buffer)
        print(self.dev.buffer.shape)
        with open(self.triggerFileName + '.csv','w') as f:
            w=csv.writer(f)
            w.writerow(self.triggers.keys())
            w.writerow(self.triggers.values())


def readData(dataFileName, triggerFileName):
    data = np.loadtxt(dataFileName + '.csv')
    trigger = np.loadtxt(triggerFileName + '.csv', delimiter=',')
    trigger = trigger.astype(int)
    return data, trigger



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='emg server')
    parser.add_argument('--name','-n',type=str, default = "shixu",required=True,help="subject name")
    args = parser.parse_args()
    # subject name
    subject = args.name
    lowCh = 0
    highCh = 5
    fs = 2000
    class_num = 6
    emg_record = Emg_record(subject = subject, lowCh = lowCh, highCh = highCh, fs = fs, host = 'localhost')
    emg_record.dev.start()
    _thread.start_new_thread(emg_record.dev.read, ())
    emg_record.dev.recordFlag = True
    emg_record.record_with_plot()
    print("Record over")

    ### Decoding
    data, trigger = readData('emg_decoding/' + str(subject) + "_data/data", 'emg_decoding/' + str(subject) + "_data/trigger")
    print(data.shape)
    print(trigger.shape)
    
    trigger_idx = {}
    class_num = len(np.unique(trigger[1]))
    for i in range(class_num):
        trigger_idx[i] = trigger[0,np.where(trigger[1]==i)[0]]
    print("trigger_idx: ", trigger_idx)
    print("class_num: ", class_num)
    # 对动作进行分类
    class_list = [i for i in range(class_num)]
    trigger1 = {key:value for key,value in trigger_idx.items() if key in class_list}
    print("trigger1: ", trigger1)
    # =================================================== #
    classify1 = EMGAnalysis(class_num = class_num, channel_num = highCh-lowCh+1, st = 1, nd = 5,
                          model_name = 'emg_decoding/' + str(subject) + '_data/' + 'model', classifier = 0)
    classify1.data = data
    classify1.trigger = trigger1
    acc1, std1 = classify1.crossValidate()
    print(acc1, std1)
