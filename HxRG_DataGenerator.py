from keras.utils import Sequence, to_categorical
import numpy as np
from matplotlib import pyplot as plt

# from sklearn.externals import joblib
# from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


class Pixel(object):
    """
    Pixel class is used to generate a single pixel time series from an HxRG 
    dark current. This can be done iteratively; but it canonically expected to
    be used as a subclass to the HxRG object below.

    HxRG detectors include HST/WFC3-IR, JWST/NIRISS, JWST/NIRcam, JWST/NIRSpec, 
    Roman Space Telescope suite of detectors, and dozens of ground-based, IR 
    detectors.

    Args:
        object (object): standard Python class inheritance

    Returns:
        pixel (instance): instance of the Pixel class, with methods to generate 
        numerous pixel time series categories for an HxRG detector
    """
    __all__ = ['normal_pixel', 'hot_pixel', 'saturated_hot_pixel', 'cold_pixel',
               'saturated_cold_pixel', 'cosmic_ray_pixel', 'popcorn_pixel',
               'noisy_pixel']

    def __init__(self, xarr=None, n_reads=100,
                 max_normal_pixel=120, normal_growth_rate=0.01,
                 darkcurrent=0.0856, bias=10000, noise=1000,
                 max_hotcold_pixel=4e4, hotcold_growth_rate=0.05,
                 saturated_delta=1e4, saturated_growth_rate=1,
                 cosmic_ray_strength=3e4, popcorn_strength=1000,
                 cosmic_ray_hit_idx=None, popcorn_pixel_up=None,
                 popcorn_pixel_down=None, noisy_pixel_noise=2000):
        """
        __init__ Initial configuration for the Pixel class instance.

        [See Pixel Docstring for Description of This Object]

        Parameters
        ----------
        xarr : 1Darray, optional
            indices associated with the IR reads up the ramp, by default None
        n_reads : int, optional
            number of reads up the ramp, if `xarr` is not provided,
            by default 100
        max_normal_pixel : int, optional
            averager number of electrons on top of the bias for `normal` 
            pixels, by default 120
        normal_growth_rate : float, optional
            exponenital growth rate up the ramp for normal pixels (depends on 
            illumination), by default 0.01
        darkcurrent : float, optional
            Dark current rate in electrons per hour, by default 0.0856
        bias : int, optional
            baseline threshold for initial voltage (in electrons) on the 
            detector, by default 10000
        noise : int, optional
            Expected std-dev of noise level in electrons for all signals,
            by default 1000
        max_hotcold_pixel : int, optional
            maximum electrons read out by hot pixels (+\- noise), by default 4e4
        hotcold_growth_rate : float, optional
            average rate of growth for hot pixels, by default 0.05
        saturated_delta : int, optional
            level below `max_hotcold_pixels` for saturated pixels to end,
            by default 1e4
        saturated_growth_rate : float, optional
            growth reate in (electrons per frame) for saturated hot/cold 
            pixels, by default 1.0
        cosmic_ray_strength : int, optional
            average electrons deposited by cosmic rays (+\- noise),
            by default 3e4
        popcorn_strength : int, optional
            average extra electrons read out by detector in popcorn event 
            [might be a factor of 2], by default 1000
        cosmic_ray_hit_idx : int, optional
            index where cosmic rays hit the detector (index ~ time),
            by default None
        popcorn_pixel_up : int, optional
            index where popcorn effect initiates, by default None
        popcorn_pixel_down : int, optional
            index where popcorn effect completes, by default None
        noisy_pixel_noise : int, optional
            std-dev in electrons for `noisy` pixels to attain on average,
            by default 2000
        """
        self.n_reads = n_reads
        self.xarr = np.arange(self.n_reads) if xarr is None else xarr

        self.max_normal_pixel = max_normal_pixel
        self.normal_growth_rate = normal_growth_rate
        self.darkcurrent = darkcurrent
        self.bias = bias
        self.noise = noise
        self.max_hotcold_pixel = max_hotcold_pixel
        self.hotcold_growth_rate = hotcold_growth_rate
        self.saturated_delta = saturated_delta
        self.saturated_growth_rate = saturated_growth_rate
        self.cosmic_ray_strength = cosmic_ray_strength
        self.popcorn_strength = popcorn_strength
        self.cosmic_ray_hit_idx = cosmic_ray_hit_idx
        self.popcorn_pixel_up = popcorn_pixel_up
        self.popcorn_pixel_down = popcorn_pixel_down
        self.noisy_pixel_noise = noisy_pixel_noise

    def normal_pixel(self):
        """
        normal_pixel creates a non `bad pixel` with slight illumination and 
        measureable dark current

        This is the base `Pixel` behaviour, where 99% of all HxRG pixels have 
        been to shown to be `normal`. All other classes of pixels have this 
        pixel behaviour as their background signal; e.g. a "hot pixel" is an 
        exponential ramp on top of a `normal pixel`; and a "cosmic ray" is a 
        step function on top of a `normal pixel`

        Returns
        -------
        1Darray
            `normal pixel` time series for dark frame of HxRG detectors
        """
        ramp_up = np.exp(self.normal_growth_rate * np.log(self.darkcurrent))
        ramp_up = ramp_up * self.xarr
        growth_curve = (1 - ramp_up)

        signal = self.max_normal_pixel * growth_curve
        bias = np.random.normal(self.bias, self.noise)

        return np.random.normal(signal + bias, self.noise)

    def hot_pixel(self):
        """
        hot_pixel Simulates the capacitor malfuction associated with a hot pixel

        Creates an exponential ramp effect on top of a normal pixel, as is 
        physically expected (and measured) to be associated with a hot pixel, 
        time-series with HxRG detectors

        Returns
        -------
        1Darray
            hot pixel time series over successive frame reads
        """
        background = self.normal_pixel()

        growth_curve = 1 - np.exp(-self.hotcold_growth_rate * self.xarr)

        signal = self.max_hotcold_pixel * growth_curve + background

        return np.random.normal(signal, self.noise)

    def saturated_hot_pixel(self):
        """
        saturated_hot_pixel Simulates a saturated hot pixel from an HxRG 
        detector

        Some hot pixels have a dramatic rise time, with a saturated shorter 
        than that of max flux or even other hot pixels. This category of bad 
        pixel is associated with a malfunctioning capacitor

        Returns
        -------
        1Darray
            saturated hot pixel time seres over successive frame reads
        """
        background = self.normal_pixel()
        signal_size = self.max_hotcold_pixel - self.saturated_delta

        growth_curve = (1 - np.exp(-self.saturated_growth_rate * self.xarr))
        signal = signal_size * growth_curve + background

        return np.random.normal(signal, self.noise)

    def cold_pixel(self):
        """
        cold_pixel Simulates a cold pixel from an HxRG detector

        Anomalous bad pixels with HxRG detectors can sometimes start very high, 
        and then dramatically fall. These are often referred to as "inverted 
        hot pixels"; here we name them "cold pixels" because the effect is that 
        the number of electrons read out dramatically decreases, like a drop in 
        'temperature'. This bad pixel is associated with capacitor malfunction.

        Returns
        -------
        1Darray
            cold pixel time-seres forover successive frame reads
        """
        background = self.normal_pixel()
        growth_curve = np.exp(self.hotcold_growth_rate * self.xarr)
        signal = self.max_hotcold_pixel * growth_curve + background
        return np.random.normal(signal, self.noise)

    def saturated_cold_pixel(self):
        """
        saturated_cold_pixel Simulates a saturated cold pixel from an HxRG 
        detector

        Some cold pixels have a dramatic decay time, with a saturated shorter 
        than that of max flux or even other cold pixels. This category of bad 
        pixel is associated with a malfunctioning capacitor

        Returns
        -------
        1Darray
            saturated cold pixel time seres over successive frame reads
        """
        background = self.normal_pixel()
        growth_curve = np.exp(-self.saturated_growth_rate * self.xarr)
        signal_size = self.max_hotcold_pixel - self.saturated_delta

        signal = signal_size * growth_curve + background
        return np.random.normal(signal, self.noise)

    def cosmic_ray_pixel(self):
        """
        cosmic_ray_pixel Simulates the cosmic ray effect of spontanously large 
        number of electrons being detected in the dark frame

        When a high energy charged particle (proton, electron, positron, 
        magnetron (hehe)) interact with the HxRG pixel walls, they can 
        dramatically deposit energy in the form of excess electrons. This 
        effect has the apperance in the IR dark frame stepping the electrons up 
        by 1000s - 10000s electrons in a single frame.

        Returns
        -------
        1Darray
            cosmic ray time series over successive frames
        """
        if self.cosmic_ray_hit_idx is None:
            options = range(self.xarr.size // 6, 5 * self.xarr.size // 6)
            cosmic_ray_hit_idx = np.random.choice(options)

        else:
            cosmic_ray_hit_idx = np.copy(self.cosmic_ray_hit_idx)

        signal = self.normal_pixel()
        cosmic_ray_level = np.random.normal(self.cosmic_ray_strength,
                                            self.noise / 10)
        signal[cosmic_ray_hit_idx:] += cosmic_ray_level
        return np.random.normal(signal, self.noise)

    def popcorn_pixel(self):
        """
        popcorn_pixel Simulates the Random Telegraph Noise (RTN) or "Popcorn 
        Pixel" effect of spontanously increasing and (often) later decreasing 
        the number of electrons in a given pixel.

        This effect is associated with a read-out error, such that a bit is 
        randomly flipped in the read out electronics or above, which has the 
        effect of stepping up the electrons read out by 1000s - 10000s in a 
        single frame; very often, the effect will invert, which thus 
        spontaneously decreases the electrons read out by 1000s - 10000s in a 
        single frame. The number of electrons reduced (after) is almost always 
        the same number as the number of electrons gained (first). This effect 
        looks like a top-hat function. It is not a temporally stationary 
        effect; but some pixels are much more likely than others to induce this 
        signal. Moreover, because the effect is associated with a bit flip, it 
        is expected to be a factor 2 +\- noise.

        Returns
        -------
        1Darray
            popcorn pixel (RTN) time series over successive frames
        """
        if self.popcorn_pixel_up is None:
            options = range(self.xarr.size // 3, self.xarr.size // 2)
            popcorn_pixel_up = np.random.choice(options)
        else:
            popcorn_pixel_up = self.popcorn_pixel_up
        if self.popcorn_pixel_down is None:
            options = range(popcorn_pixel_up, self.xarr.size)
            popcorn_pixel_down = np.random.choice(options)
        else:
            popcorn_pixel_down = self.popcorn_pixel_down

        signal = self.normal_pixel()
        signal[popcorn_pixel_up:popcorn_pixel_down] += self.popcorn_strength
        return np.random.normal(signal, self.noise)

    def noisy_pixel(self):
        """
        noisy_pixel Simulates the effect of an excessively noisuy pixel

        Some pixels are just more noisy than others. Here we simulate a `normal 
        pixel` and then increase its average noise properties (~2x)

        Returns
        -------
        1Darray
            noisy pixel time series over successive frames
        """
        signal = self.normal_pixel()
        med_signal = np.median(signal)
        signal = (signal - med_signal) / self.noise * self.noisy_pixel_noise

        return signal


class HxRG(Pixel):
    """
    HxRG class is used to generate a large number of pixel time series to 
    simulate a full (or subframe) HxRG dark current data cube. This can be done 
    independently or as a subclass to the HxRGDataGenerator object below.

    HxRG detectors include HST/WFC3-IR, JWST/NIRISS, JWST/NIRcam, JWST/NIRSpec, 
    Roman Space Telescope suite of detectors, and dozens of ground-based, IR 
    detectors.

    Args:
        Pixel (object): inherits the Pixel class (above) to generate a 
        well-defined distribution of `normal` and `bad` pixels

    Returns:
        HxRG (instance): instance of the HxRG class, with methods to distribute 
        the percent of good + bad pixels over the detector; as well as 
        `populate` the detector with thousands to millions of pixel time series 
        with respect to HxRG detectors' behaviours.
    """

    def __init__(self, n_norm, n_bad, populate=False,
                 percent_per_class=None):
        """
        __init__ Initiates the configuration of an HxRG detector, as a set of 
        thousands to millions of IR time series ('pixels')

        Create the underlying distribution and number of good + bad pixels over 
        the full (or subframe) HxRG image, data cube

        Parameters
        ----------
        n_norm : int
            Number of 'good' pixels
        n_bad : int
            Number of 'bad' pixels
        populate : bool, optional
            Whether to generate `n_pixels` time-series as a list below, by 
            default False
        percent_per_class : 1Darray (8 elements), optional
            the percent of each bad pixel as a function of number of total 
            pixels. The order *must* be [normal, hot, hot saturated, cold, cold 
            saturated, cosmic ray, popcorn, noisy], by default None
        """
        super().__init__()

        self.n_norm = n_norm
        self.n_bad = n_bad
        self.n_pixels = self.n_norm + self.n_bad
        self.percent_per_class = percent_per_class

        # Compute the fraction of pixels
        self.configure_pixel_percentage()

        # Compute the number of pixels in addition to the fraction of pixels
        self.configure_pixel_numbers()

        if populate:
            self.populate_detector()

    def configure_pixel_percentage(self):
        """
        configure_pixel_percentage Convert from percent_per_class to fraction 
        of each category of bad pixel and normal pixel

        The inputs for `percent_per_class` are in percetn of pixels for each 
        category of good + bad pixels. This method reorganizes that 
        information, as wll as establishes both the default 
        (percent_per_class=None) and special example 
        (percent_per_class='nircam') distribution. The default behaviour places 
        all pixels in the 'normal' bin with `self.frac_normal = 1`. The special 
        example (percent_per_class='nircam') option distributes the percent of 
        each bad pixel as described in Raucher et al 2017. If 
        `percent_per_class` is provided as an 8-element array, then this 
        function will renormalize it to integrate to 1.0.
        """
        percent_per_class_given = False
        if self.percent_per_class is None:
            self.frac_normal = 1.0
            self.frac_hot = 0.
            self.frac_sat_hot = 0.
            self.frac_cold = 0.
            self.frac_sat_cold = 0.
            self.frac_cosmic = 0.
            self.frac_popcorn = 0.
            self.frac_noisy = 0.
        elif self.percent_per_class == 'nircam':
            # Fraction of `bad pixels` that are each class
            self.frac_normal = 0.0  # place holder
            self.frac_hot = 0.5
            self.frac_sat_hot = 0.1
            self.frac_cold = 0.1
            self.frac_sat_cold = 0.1
            self.frac_cosmic = 0.1
            self.frac_popcorn = 0.01
            self.frac_noisy = 0.09
        else:
            percent_per_class_given = True
            self.frac_normal = self.percent_per_class[0]
            self.frac_hot = self.percent_per_class[1]
            self.frac_sat_hot = self.percent_per_class[2]
            self.frac_cold = self.percent_per_class[3]
            self.frac_sat_cold = self.percent_per_class[4]
            self.frac_cosmic = self.percent_per_class[5]
            self.frac_popcorn = self.percent_per_class[6]
            self.frac_noisy = self.percent_per_class[7]

        if not percent_per_class_given:
            # Repopulate and Re-calibrate from `percent of bad pixels`
            #   to `percent of pixels`
            self.percent_per_class = np.array([
                self.frac_normal, self.frac_hot, self.frac_sat_hot,
                self.frac_cold, self.frac_sat_cold, self.frac_cosmic,
                self.frac_popcorn, self.frac_noisy
            ])

            # Rescale the percent of bad pixels from the percet `n_bad`
            #   to percent `n_pixels`
            percent_bad_pixels = self.n_bad / self.n_pixels
            self.percent_per_class = self.percent_per_class * percent_bad_pixels

            # Redefine the fraction of normal pixels as the remaining
            #   percentage of pixels other than `bad` pixels
            self.percent_per_class[0] = 1 - self.percent_per_class.sum()

        # Rebalance all elements to integrate to exactly 1.0
        #   This is a catch to ensure that the distribution is valid
        self.percent_per_class /= self.percent_per_class.sum()

    def configure_pixel_numbers(self, assign_remainder=0):
        """
        configure_pixel_numbers convert from percentage of pixels to actual 
        number of pixels, and assigns the remainder to the normal pixel 
        category (or other assigned through kwarg above)

        Parameters
        ----------
        assign_remainder : int, optional
            index associated with the category of good or bad pixel that the 
            remainder of all pixels on the detector should be placed within,
            by default 0:normal
        """
        self.n_hot = int(self.n_bad * self.frac_hot)
        self.n_sat_hot = int(self.n_bad * self.frac_sat_hot)
        self.n_cold = int(self.n_bad * self.frac_cold)
        self.n_sat_cold = int(self.n_bad * self.frac_sat_cold)
        self.n_cosmicray = int(self.n_bad * self.frac_cosmic)
        self.n_popcorn = int(self.n_bad * self.frac_popcorn)
        self.n_noisy = int(self.n_bad * self.frac_noisy)

        self.n_per_class = np.array([self.n_norm,
                                     self.n_hot,
                                     self.n_sat_hot,
                                     self.n_cold,
                                     self.n_sat_cold,
                                     self.n_cosmicray,
                                     self.n_popcorn,
                                     self.n_noisy])

        # Set/take leftovers to/from `normal`:0, `hot pixel`:1, etc
        n_leftover = self.n_bad - self.n_per_class[1:].sum()
        print(n_leftover)
        print(self.n_per_class[assign_remainder])
        self.n_per_class[assign_remainder] += n_leftover

    def populate_detector(self):
        """
        populate_detector Create a simualtion fo the HxRG detector in question

        Iterate over the number of pixels to create a simulated HxRG dark frame 
        (image cube) of 'mostly' normal pixels with ~1% of bad pixels.
        """
        self.pixels = []
        for method_name, n_samples in zip(self.__all__, self.n_per_class):
            if n_samples > 0:
                method_ = self.__getattribute__(method_name)
                for _ in range(n_samples):
                    self.pixels.append(method_())

        indices = np.arange(self.n_pixels)
        indices = np.random.choice(indices, replace=False)
        self.pixels = self.pixels[indices]


class HxRGDataGenerator(Sequence):  # , HxRG
    """
    HxRGDataGenerator Keras Data Generator for HxRG time-series dark pixels

    This Keras Data Generator can be used to train a neural network on the HxRG 
    dark current time-series dark pixels to classify, regress, or anomaly 
    detect bad pixels from normal pixels. HxRG dark frame pixels are a 
    time-series of 10-1000+ 'read up the ramp' The generator will samples 
    `batch_size` number of pixel time series, based on the prescribed 
    `percent_per_class` distribution of good + bad pixels (e.g. 99% vs 1%, 
    respectively).

    Parameters
    ----------
    Sequence : Keras Data Generator class
        Super class for using a generator when training a keras neural network.
    """

    def __init__(self, xarr=None, percent_per_class=None, batch_size=32,
                 dim=100, n_channels=1, shuffle=True):
        """
        __init__ Initialize the Keras Data generator with a specific 
        distribution of good + bad pixels

        The Keras Data Generator creates a set of batches of size `batch_size`, 
        where each sample of pixels has a lenght of dark frame reads per 
        integration `dim`. This will generate `batch_size` number of pixels to 
        be used in training a neural network to classify, regress, or anomaly 
        detect good+ bad pixels from our simulated HxRG detector behaviour.

        Parameters
        ----------
        xarr : 1Darray, optional
            An input axis over the index of each read, by default None
        percent_per_class : 1Darray (8 elements), optional
            Set of occurance rates per bad pixel category to establish the 
            distribution of good + bad pixels, by default None; corresponds to 
            all `normal` pixels.
        batch_size : int, optional
            Number of samples to be generated per batch, by default 32
        dim : int, optional
            Number of frames to be generated per sample, by default 100
        n_channels : int, optional
            Number of pixels to be associate per sample, by default 1
        shuffle : bool, optional
            Whether to randomly shuffle the samples (True) or read over a 
            predefined set of pixels (False), by default True
        """
        super(Sequence, self).__init__()
        # super(HxRG, self).__init__()

        self.percent_per_class = percent_per_class
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        __len__ Denotes the number of batches per epoch

        Returns
        -------
        int
            number of batches per epoch
        """
        return self.batch_size

    def __getitem__(self, index):
        """
        __getitem__ Generate one batch of data

        Parameters
        ----------
        index : int, optional
            starting position for a predefined HxRG simulated dark frame

        Returns
        -------
        2Darray
            batch of samples, sized (batch_size, dim, n_channels)
        """
        detector = HxRG(n_norm=batch_size, n_bad=0)

        # Return the list of IR pixel time series the input/output
        #   data for the autoencoder (input == output)
        return detector.pixels

    def on_epoch_end(self):
        """
        on_epoch_end Updates indexes after each epoch

        Our base code is to generate `batch_size` new pixel time series. In the 
        case of a pre-defined HxRG sample (real world or simulated dark frame)
        this method randomly resamples over the indices of that dark frame.
        """
        # self.samples = HxRG(percent_per_class=self.percent_per_class)
        pass
