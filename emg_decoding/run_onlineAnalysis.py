import pytrigno
import numpy as np
import joblib
import time
import math
import _thread
from process import EMGAnalysis
from functools import partial
import redis


class Emg_onlineAnalysis():
	def __init__(self, subject, class_num, lowCh, highCh, window_length, window_step, fs, host):
		# 0:rest 1/2:quzhou 3/4:jianqianqu 5/6:jianwaizhan
		self.window_length = int(window_length*fs)  # 窗长400个点
		self.window_step = int(window_step*fs)  # 间隔100个点
		self.dev = pytrigno.TrignoEMG(channel_range = (lowCh, highCh), samples_per_read = 27, host = host, buffered = True)
		self.window = np.zeros((highCh - lowCh + 1, 0))
		self.clf_lists = [joblib.load('emg_decoding/' + str(subject) + '_wrist.m')]
		self.onlineDecode = EMGAnalysis(class_num = class_num, channel_num = highCh-lowCh+1, st = 1, nd = 6,
						  model_name = str(subject) + '_wrist', classifier = 0)
		self.buffer = np.zeros((highCh - lowCh + 1,0))
		self.window_num = int((0.3-0.05)/window_step)#由大部分文章迟滞300ms，所以取为5
		self.predictBox = np.zeros((self.window_num), dtype = int)
		self.count = 0
		self.i = 0
		self.action_lists = [0]
		self.finished = True #来自机械臂控制
		self.EMG_time = []
		self.actual_label = 0
		self.time0 = time.time()

	def collectData(self, rds):
		print('Action server running...')
		while True:
			if self.dev.buffer.shape[1] >= (self.window_length + self.i*self.window_step):
				# print(self.dev.buffer.shape)
				self.window = self.dev.buffer[:,-self.window_length:]  # 取最后一个窗长的数据
				# _thread.start_new_thread(self.predict, (self.window,))
				# time.sleep(0.05)
				action = self.predict(self.window)
				rds.set('action', str(action), ex=1)

	def predict(self, window):
		"""
		多个（5个）窗的最大投票法预测
		"""
		action = 0
		if self.finished:
			method = self.clf_lists[0]
			if self.action_lists[-1] == self.actual_label and self.actual_label != 0:
				self.finished = False
		else:
			method = self.clf_lists[math.ceil(self.action_lists[-1]/2)]
		if self.count < self.window_num:
			self.count += 1
			self.predictBox[self.count - 1] = self.onlineDecode.predict(window, method)
		else:
			action = np.argmax(np.bincount(self.predictBox))
			# print(action)
			self.action_lists.append(action)
			self.EMG_time.append(time.time() - self.time0)
			self.count = 0
		return action


if __name__ == "__main__":
	# redis server
	pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
	rds = redis.Redis(connection_pool=pool)
	
	subject = 'test'
	lowCh = 0
	highCh = 3
	fs = 2000
	class_num = 5
	analysis = Emg_onlineAnalysis(subject=subject, class_num=class_num, lowCh = lowCh, highCh = highCh, 
							   window_length = 0.2, window_step = 0.05, fs = fs, host = 'localhost')
	analysis.dev.start()
	_thread.start_new_thread(analysis.dev.read, ())
	analysis.dev.recordFlag = True
	run_classify = partial(analysis.collectData, rds)
	run_classify()



	