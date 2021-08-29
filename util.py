import numpy as np
import scipy.signal as sgn
import pywt 




def high_pass_filter(x, low_cutoff=1000, SAMPLE_RATE=186666677):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = sgn.butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = sgn.sosfilt(sos, x)

    return filtered_sig


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sgn.butter(order, normal_cutoff, btype='low', analog=False)
    y = sgn.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sgn.butter(order, normal_cutoff, btype='high', analog=False)
    y = sgn.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, cutoff_low, cuttoff_high, fs, order):
    nyq = 0.5*fs
    normal_cutoff_low = cutoff_low / nyq
    normal_cutoff_high = cuttoff_high / nyq    
    # Get the filter coefficients 
    b, a = sgn.butter(order, [normal_cutoff_low,normal_cutoff_high], btype='band', analog=False)
    y = sgn.filtfilt(b, a, data)
    return y

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def ProcessSignal(sig,fs,lowcutoff=1*10**6,wavelet='db4'):
    x = sgn.resample(denoise_signal(butter_lowpass_filter(sig,lowcutoff,fs,5),wavelet),64*64)
    x_min ,x_max = x.min(0),x.max(0)
    return (x-x_min)/(x_max-x_min)

def ProcessSignal16x16(sig,fs,lowcutoff=1*10**6,wavelet='db4'):
    x = sgn.resample(denoise_signal(butter_lowpass_filter(sig,lowcutoff,fs,5),wavelet),16*16)
    x_min ,x_max = x.min(0),x.max(0)
    return (x-x_min)/(x_max-x_min)

def ResampleSignal64x64(sig):
    x = sgn.resample(sig,64*64)
    x_min ,x_max = x.min(0),x.max(0)
    return (x-x_min)/(x_max-x_min)

def ResampleSignal16x16(sig):
    x = sgn.resample(sig,16*16)
    x_min ,x_max = x.min(0),x.max(0)
    return (x-x_min)/(x_max-x_min)