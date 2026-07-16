# General Read Me
### **Author**: Matteo Houston
### **Last Updated**: 16 July 2026

##  -- General Overview -- 

As stated in the beginning of the code, the code (which is rather lengthy) is less applicable for actual pipeline use, and more so a general experimental ground to figure out which approach to signal processing would be best. In this read me, I cover the very basic approaches to signal processing, as well as the various packages and methods I utilized and as to why. While I don't have an explicit favorite, I do lay down a few suggestions, and any future paths I recommend investigating.

## -- First Approach -- 

The first thing LIGO does upon receiving a signal is to whiten and band pass it. This is to mitigate the noise and bring out the signal hidden underneath. When working with any transform, the process of whitening ( + windowing) and band passing is normally a necessity, however, in our work over the past six weeks, given a high enough SNR, whitening and band passing maybe enough to identify the presence of a chirp.

Here, I recommend finding the lowest SNR possible before whitening and band passing becomes pointless. I tried finding this, however, it does seem a bit more complicated than just SNR. For instance, the SNR of GW170817 is higher than GW150914, however, the chirp cannot be identified in the graph after whitening and band passing. It's likely that BNS coalescence events may not have the necessary strength in strain to be picked up, even if they are very much present.

## -- Packages -- 

The most frequent packages I used were ssqueezepy and GWpy. Some other packages I'd recommend giving a more in-depth look into are FFTW (Fastest Fourier Transform in the West), Wavelet Qp Transform, and the Hilbert-Huang Transform (including any modifications of this transform). An important aspect to any transform is that it can be inverted, so that we can reconstruct the signal. Not every transform is invertible.

The reason behind using ssqueezepy and GWpy is that ssqueezepy contains both the CWT and STFT, as well as its own special version (squeezed CWT/STFT), as well as its own ridge extraction and signal reconstruction. I chose to work with GWpy most often because it had rather robust whiten and bandpassing methods. I have not looked too in-depth to packages like pyCBC, however would recommend doing so anyway.

So far, the methods I've had success deploying are the following: 
  1. The q-transform (from GWpy)
  2. The Short-time Fourier Transform (from ssqueezepy)
  3. The Continuous Wavelet Transform (also ssqueezepy)
  4. The Fast Fourier Transform (from numpy/GWpy. Recommend looking at FFTW)

(Will update more in the future! Currently also writing guide)
