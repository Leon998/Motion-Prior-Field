import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from EMGFeatures import EMGFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from scipy import io
from sklearn import metrics
from pdb import set_trace
from sklearn.pipeline import make_pipeline
import joblib
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
class EMGAnalysis():
	def __init__(self, fs = 2000, class_num = 2, channel_num = 4, st = 1, nd = 7.5, dir = '.', model_name = 'model', classifier = 0):
		self.data = None
		self.trigger = {}
		self.fs = fs
		self.class_num = class_num
		self.channel_num = channel_num
		self.dir = dir
		self.model_name = model_name
		self.st = st * fs  # 从1s开始，即第2000个点
		self.nd = nd * fs  # 到7.5s结束，即第15000个点
		self.window_length = int(0.2*self.fs)  # 窗长400个点
		self.window_step = int(0.05*self.fs)  # 间隔100个点
		lda = LDA()
		SVM = GridSearchCV(svm.SVC(kernel='rbf'), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
		forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
		clf_lists = [lda, SVM, forest]
		# print(classifier)
		self.clf = clf_lists[classifier]
		# print(self.clf)
		self.X = np.zeros((0, self.channel_num, self.window_length))
		self.y = np.zeros(0, dtype = int)
		self.features = None

	def readData(self):
		origin_data = io.loadmat(self.dir, mat_dtype = True,squeeze_me = True)['EMG_DATA']
		self.data = origin_data[:self.channel_num, :]
		# for i in range(0, self.class_num - 1):
		self.trigger[0] = np.where(origin_data[-1, :] == 7)[0]
		self.trigger[self.class_num - 1] = np.where(origin_data[-1, :] == 9)[0]
	def preprocessData(self):
		self.data = self._bandpass_filter()
		self.data = self._notch_filter()

	def reshapeData(self):
		"""
		将预处理后的数据根据滑动窗分段，整理成X和y的形式，其中X形状为（段数，4通道，窗长），y形状为（段数）
		"""
		keys = list(self.trigger.keys())
		trial_num = len(self.trigger[keys[0]])
		frame_num = self._getFrameNum()
		for i in keys: 
			for j in range(trial_num):
				for n in range(frame_num):
					# print(n)
					left = self.trigger[i][j] + self.st + self.window_step*n
					right = left + self.window_length
					if right < self.data.shape[1] or right == self.data.shape[1]:
						self.X = np.vstack((self.X, np.reshape(self.data[:, left: right], (1, self.channel_num, self.window_length))))
						self.y = np.hstack((self.y, np.array([i])))
	
		permutation = np.random.permutation(self.y.shape[0])
		self.X = self.X[permutation, :, :]
		self.y = self.y[permutation]
		print("Data shape: ", self.X.shape, self.y.shape)
	

	def getFeatures(self):
		feature_num = 3
		featureExtraction = EMGFeatures()
		self.features = np.zeros((self.X.shape[0], self.X.shape[1]*feature_num))
		for i in range(self.X.shape[0]):
			for j in range(self.X.shape[1]):
				self.features[i, j*feature_num: j*feature_num + feature_num] = np.hstack((featureExtraction.get_mav(self.X[i, j, :]), 
					featureExtraction.get_wl(self.X[i, j, :]), featureExtraction.get_wamp(self.X[i, j, :])))
					# EMGFeatures.get_wamp(self.X[i, j, :]),
					# EMGFeatures.get_rms(self.X[i, j, :]),
					# EMGFeatures.get_zc(self.X[i, j, :]),
					# EMGFeatures.get_ssc(self.X[i, j, :])))
				# self.mav_features[i, j] = self.X[i, j, 200]
	def crossValidate(self):
		# self.readData()
		self.preprocessData()
		self.reshapeData()
		self.getFeatures()
		cv = StratifiedKFold(5, shuffle=True, random_state=None)
		scorer = metrics.make_scorer(metrics.accuracy_score)
		# set_trace()
		print("Feature shape: ", self.features.shape)
		# features_train, features_test, y_train, y_test = train_test_split(self.features, self.y, test_size = 0.2, random_state = 0)
		# train
		# acc = cross_val_score(self.lda, features_train, y_train, cv = cv, scoring = scorer)
		# self.lda.fit(features_train, y_train)
		acc = cross_val_score(self.clf, self.features, self.y, cv = cv, scoring = scorer)
		self.clf.fit(self.features, self.y)
		joblib.dump(self.clf, self.model_name + ".m")
		# # test
		# y_pred = self.lda.predict(features_test)
		# labels = list(set(self.y))
		# conf_mat = confusion_matrix(y_test, y_pred, labels = labels)

		# # print("confusion_matrix(left labels: y_true, up labels: y_pred):")
		# # print("labels\t")
		# # for i in range(len(labels)):
		# # 	print(labels[i],"\t")
		# # print()
		# # for i in range(len(conf_mat)):
		# # 	print(i,"\t")
		# # 	for j in range(len(conf_mat[i])):
		# # 		print(conf_mat[i][j],'\t'),
		# # 	print()
		# # print()
		# print(conf_mat)
		return np.mean(acc), np.std(acc)
	
	def predict(self, data, clf):
		"""
		单个窗的预测
		"""
		# data: channel * samples
		self.data = data
		self.preprocessData()
		self.X = np.reshape(self.data, (1, self.data.shape[0], self.data.shape[1]))
		# self.reshapeData()
		self.getFeatures()
		return clf.predict(self.features)

	def _bandpass_filter(self, order = 5, lowcut = 20., highcut = 500.):
		""" Forward-backward band-pass filtering (IIR butterworth filter) """

		Wn=[tmp*2/self.fs for tmp in [lowcut, highcut]]
		filter_b,filter_a = butter(order, Wn, btype = 'band')
		return filtfilt(filter_b, filter_a, self.data)

	def _notch_filter(self, stop_fz = 60, Q = 30):
		filter_b, filter_a = iirnotch(stop_fz, Q, self.fs)
		return filtfilt(filter_b, filter_a, self.data)

	def _getFrameNum(self):
		return int((self.nd - self.st - self.window_length)/self.window_step) + 1



if __name__ == "__main__":
	analysis = EMGAnalysis(dir = '/Users/lwre/Downloads/毕业设计/EMG数据分析/fengTrainData.mat')
	# analysis.readData('/Users/lwre/Downloads/毕业设计/EMG数据分析/fengTrainData.mat')
	# analysis.readData('/Users/lwre/Downloads/毕业设计/EMG数据分析/EmgData_CHENGHONGYUAN_20191122.mat')
	# analysis.readData('/Users/lwre/Downloads/毕业设计/EMG数据分析/EmgData_DINGHAOHAO_20191024.mat')
	# analysis.readData('/Users/lwre/Downloads/毕业设计/EMG数据分析/EmgData_HANWENQIANG-20191021.mat')

	# analysis.preprocessData()
	# analysis.reshapeData()
	# analysis.shuffleData()
	# analysis.getFeatures()
	acc = analysis.crossValidate()
	print(acc)
	dataDecoding = EMGAnalysis()
	origin_data = io.loadmat('/Users/lwre/Downloads/毕业设计/EMG数据分析/fengTestData.mat', mat_dtype = True,squeeze_me = True)['EMG_DATA']
	data = origin_data[:6, :]
	label = np.zeros((8637))
	for n in range(8637):
		testData = data[:, analysis.window_step*n: analysis.window_step*n + analysis.window_length]
		label[n] = dataDecoding.predict(testData, analysis.lda)
	print(label)
	set_trace()





