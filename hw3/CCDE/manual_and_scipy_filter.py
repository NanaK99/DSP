'''
This script contains manual implementation of scipy filters
in real time fashion and compares the implementation with scipy one. 
The task is to 
1) Study scipy.signal.iirfilter and scipy.signal.lfilter functions (see below) 
2) Write a filter amplutude repsonse plotting function (in db scal) which will take filter coefficents (a,b) as input and plot according filter amplitude response. To do so you need to use transfer function definition and obtain frequency response formula for given filter, then take discrete values for angular frequency and plot the amplitude response. 
3) Plot various filters amplitude spectrum obtaining filter coefficients (a,b) using scipy.signal.iirfilter functino. Vary filter order, band type (lowpass, bandpass, highpass) cut-off frequency(ies) and filter type (e.g. Butterworth : ‘butter’, Chebyshev I : ‘cheby1’, Chebyshev II : ‘cheby2’ etc, see the iirfilter function web page)
4) Take audio recordings (in .wav format and preferably with 16khz sampling rate) 
filter obtaining coefficients for various filters (highpass, bandpass, lowpass) see example in get_filter_coefs()
and filter audio examples with that filters saving the results. 
'''
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as file


def get_filter_coefs(filt_order, cut_off, sample_rate):
	nyq_freq = sample_rate / 2
	cut_off_norm = cut_off / nyq_freq # normalized cut-off frequency
	filter_coefs = signal.iirfilter(filt_order, cut_off_norm, btype='highpass', ftype='butter', output='ba')
	b, a = filter_coefs
	return b, a

def get_filter_coefs_bandpass(filt_order, cut_off1, cut_off2, sample_rate):
	nyq_freq = sample_rate / 2
	cut_off_norm1 = cut_off1 / nyq_freq # normalized cut-off frequency
	cut_off_norm2 = cut_off2 / nyq_freq # normalized cut-off frequency
	filter_coefs = signal.iirfilter(filt_order, [cut_off_norm1, cut_off_norm2], btype='bandpass', ftype='butter', output='ba')
	b, a = filter_coefs
	return b, a

def get_filter_coefs_cheby(filt_order, cut_off1, sample_rate):
	nyq_freq = sample_rate / 2
	cut_off_norm = cut_off1 / nyq_freq # normalized cut-off frequency
	filter_coefs = signal.iirfilter(filt_order, cut_off_norm, rs=60, btype='highpass', ftype='cheby2', output='ba')
	b, a = filter_coefs
	return b, a

def scipy_lin_filt(x, b, a):
	out_ = signal.lfilter(b, a, x)
	return out_

def lin_filt_short(x, b, a, x_state, y_state):
	# x_state = [x[-n], ... x[-1]],
	# y_state = [y[-n], ... y[-1]]
	l = len(x)
	order_ = len(b) - 1
	x = np.concatenate((x_state, x))
	y = np.concatenate((y_state, np.zeros(l)))
	flip_b = np.flip(b[1:])
	flip_a = np.flip(a[1:])
	for n in range(order_,len(x)):
		y[n] = b[0] * x[n] + np.sum(x_state * flip_b) -  np.sum(y_state * flip_a)
		y[n] = y[n] / a[0]
		x_state = x[n-order_+1:n+1]; y_state = y[n-order_+1:n+1]
	return y[order_:], x_state, y_state


def manual_lin_filt(x, seg_len, b, a):
	'''
	applying linear filter by dividing x into seg_len segments
	and applying filter on each segment using filter states of previous segment
	this kind of implementation is suitable for real time applications
	'''
	order_ = len(b) - 1
	assert seg_len > order_, "seg len should be greater 1"
	out_list = []
	x_state = np.zeros(order_)
	y_state = np.zeros(order_)
	pad_len = seg_len - len(x) % seg_len
	x = np.concatenate((x, np.zeros(pad_len)))
	num_seg = len(x) // seg_len
	for i in range(num_seg):
		curr_seg = x[i*seg_len: (i+1)*seg_len]
		curr_out, x_state, y_state = lin_filt_short(curr_seg, b, a, x_state, y_state)
		out_list.append(curr_out)
	return np.hstack(out_list)[:-pad_len]

if __name__ == '__main__':
	# let initialize parameters of filter used in project
	w_path = 'some_audio_path'
	out_path = 'path_for_saving_output'
	filter_order = 5
	cut_off1 = 100 # hz
	cut_off2 = 4000
	sample_rate = 16000
	# b, a = get_filter_coefs(filter_order, cut_off, sample_rate)
	# b, a = get_filter_coefs_bandpass(filter_order, cut_off1, cut_off2, sample_rate)
	b, a = get_filter_coefs_cheby(filter_order, cut_off1, sample_rate)
	# print(b,a)
	# print(np.flip(b[1:]))
	# print(b[1:])
	'''
	here we compare scipy filtering with our manual filtering
	you can vary signal_len and seg_len to compare scipy and manual algorithm outputs
	'''
	signal_len = 16000
	seg_len = 100 # here we can take 0.015s * sr which is segments in our nc algorithms
	# initializing random signal
	# x = np.random.rand(signal_len) - 1/2
	sr, x = file.read(w_path)
	x = x.astype('float')
	# x = np.zeros(signal_len, dtype=np.float)
	# x[0]=1.
	y_scipy = scipy_lin_filt(x, b, a)
	y_manual = manual_lin_filt(x, seg_len, b, a)
	print('y_scipy: {}'.format(y_scipy))
	print('y_manual: {}'.format(y_manual))
	# file.write('input_.wav', sample_rate, x)
	file.write(out_path, sample_rate, y_manual.astype('int16'))