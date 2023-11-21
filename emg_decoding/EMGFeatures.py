import numpy as np
class EMGFeatures():
	def __init__(self):
		pass
	def get_mav(self, x): #mean_absolute_value
		return np.mean(np.absolute(x), axis=0)

	def get_wl(self, x):  #waveform_length
		return np.sum(np.absolute(np.diff(x, axis = 0)), axis = 0)
		#WAMP
	def get_wamp(self, x): #Wilson amplitude
		threshold = np.std(x)*0.05
		return np.sum(np.absolute(np.diff(x, axis = 0)) > threshold, axis=0)
	# def get_ar_coeffs(self, x, order = 4): #Auto-regressive coefficient
	# 	return 
	# def get_mavs(self, x): 

		#MAVS
	def get_rms(self, x):
		return np.sqrt(np.mean(np.square(x), axis=0))
	def get_zc(self, x):
		threshold = np.std(x)*0.05
		return np.sum(np.logical_and(np.diff(np.signbit(x), axis = 0),
        	np.absolute(np.diff(x, axis = 0)) > threshold), axis = 0)
	def get_ssc(self, x):
		threshold = np.std(x)*0.05
		return np.sum(
			np.logical_and(np.diff(np.signbit(np.diff(x, axis = 0)), axis = 0), 
			np.logical_or(np.diff(x[:-1], axis = 0) > threshold, 
				np.diff(x[1:], axis = 0) > threshold)))