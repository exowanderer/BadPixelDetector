# HxRG Bad Pixel Simulator and Detector

1. Generate bad pixels from the IR suite of detectors from the Hubble Space Telescope (WFC3; H1RG), James Webb Space Telescope (NIRISS, NIRSpec, NIRCam; H2RGs), and Nancy Grace Roman Space Telescope (H4RGs): [Colab Notebook Example](https://colab.research.google.com/drive/19rlGf9JmCdjtxK4HMUUkLMdYoQi1Rdqo?usp=sharing)
2. Train a suite of deep learning algorithms to detect and classify these anomalous pixels. Using the IR-Bad Pixel generator in (1), train a neural network as either an anomaly detector, or a feature classifier. This is a semi-supervised approached, where we trained the simulator on real data (using linear regression) and then use that simulated data to train our deep learning suite.

With the HxRG series of IR detectors, all 'images' are generated as time-series cubes of "voltage" over time. The final science image is generated as the non-linear calibration of these 'ramp' files into 'slope' files (i.e. images in unites of electrons/s).

Athough 'bad pixels' are anomalous signals in the final image, they are generated via time-series phenomena in the IR ramp files. Such as an exponential rise for `hot pixels`; an exponential fall for `cold pixels`; a step function for `cosmic rays`; or a hat function for popcorn (RTN) pixels.

We developed this package to detect these time-series anomalies; but found that the popcorn (RTN) pixels are not well populated enough to detect. As such, we further built the bad pixel generator to train our neural networks what a 'bad pixel' looks like. After applying to JWST calibration data, hosted on MAST, we were able to accurately detect 20 popcorn pixels out of 4 million total pixels.

Further steps in development are 
A. Use clustering analysis to quantify the separation between the 9 classes of bad pixels
B. Apply both 1d convolution and LSTM/GRU time-series anomaly detection system (trained on simulated data) to the JWST calibration data on MAST.
