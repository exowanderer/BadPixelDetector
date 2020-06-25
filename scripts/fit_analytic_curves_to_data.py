import numpy as np
from matplotlib import pyplot as plt
plt.ion()

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def normal_pixel(xarr, end_growth=120, growth_rate=0.01,
                 darkcurrent=0.0856, bias=10000, noise=1000):
    # normal pixel
    signal = end_growth * \
        (1 - np.exp(growth_rate * np.log(darkcurrent) * xarr))
    signal = signal + np.random.normal(bias, noise)
    return np.random.normal(signal, noise)


def hot_pixel(xarr, max_growth=4e4, growth_rate=0.05, noise=1000):
    # hot pixel
    background = normal_pixel(xarr)
    signal = max_growth * (1 - np.exp(-growth_rate * xarr)) + background
    return np.random.normal(signal, noise)


def saturated_hot_pixel(xarr, max_growth=4e4, sat_delta=1e4, growth_rate=1, noise=1000):
    # satruated hot pixel
    background = normal_pixel(xarr)
    signal = (max_growth - sat_delta) * \
        (1 - np.exp(-growth_rate * xarr)) + background
    return np.random.normal(signal, noise)


def cold_pixel(xarr, max_growth=4e4, growth_rate=0.05, noise=1000):
    # cold pixel
    background = normal_pixel(xarr)
    signal = max_growth * (np.exp(-0.05 * xarr)) + background
    return np.random.normal(signal, noise)


def saturated_cold_pixel(xarr, max_growth=4e4, growth_rate=1, noise=1000):
    # saturated cold pixel
    background = normal_pixel(xarr)
    signal = max_growth * (np.exp(-growth_rate * xarr)) + background
    return np.random.normal(signal, noise)


def cosmic_ray_pixel(xarr, strength=3e4, where_hit=None, noise=1000):
    # cosmic ray
    if where_hit is None:
        where_hit = np.random.choice(range(xarr.size // 6, 5 * xarr.size // 6))

    signal = normal_pixel(xarr)
    signal[where_hit:] += np.random.normal(strength, noise / 10)
    return np.random.normal(signal, noise)


def popcorn_pixel(xarr, strength=1000, where_up=None, where_down=None, noise=1000):
    # popcorn
    if where_up is None:
        where_up = np.random.choice(range(xarr.size // 3, xarr.size // 2))
    if where_down is None:
        where_down = np.random.choice(range(xarr.size // 4, xarr.size))

    signal = normal_pixel(xarr, noise=noise)
    signal[where_up:where_down] += strength
    return np.random.normal(signal, noise)


def noisy_pixel(xarr, noise_mult=2, noise=1000):
    # noisy pixel -- extra scatter
    return normal_pixel(xarr, noise=noise_mult * noise)


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-pn', '--plot_now', action='store_true',
                    help='Toggle to plot one of each class')
    ap.add_argument('-sn', '--save_name', type=str, default=None,
                    help='Toogle to save the pickle file')

    clargs = ap.parse_args()

    plot_now = clargs.plot_now
    save_name = clargs.save_name

    classes = np.array([normal_pixel,
                        hot_pixel,
                        saturated_hot_pixel,
                        cold_pixel,
                        saturated_cold_pixel,
                        cosmic_ray_pixel,
                        popcorn_pixel,
                        noisy_pixel])

    n_classes = len(classes)
    n_samples = n_norm + n_bad

    labels = np.zeros(n_samples, dtype=int)

    if non_uniform:
        n_per_class = non_uniform_distribution(n_norm, n_bad,
                                               frac_hot=0.5,
                                               frac_sat_hot=0.1,
                                               frac_cold=0.1,
                                               frac_sat_cold=0.1,
                                               frac_cosmic=0.1,
                                               frac_popcorn=0.01,
                                               frac_noisy=0.09)
        n_complete = 0
        for class_, n_now in enumerate(n_per_class):
            labels[n_complete:n_complete + n_now] = class_
            n_complete += n_now
    else:
        # Uniform distribution of bad pixel classes
        labels[-n_bad:] = np.random.randint(1, n_classes, n_bad)

    # Shuffle all labels to randomize samples
    np.random.shuffle(labels)

    # Create features per class based on label
    features = np.zeros((n_samples, n_timesteps))
    for k, label_ in tqdm(enumerate(labels), total=n_samples):
        features[k] = classes[label_](xarr)

    if plot_now:
        norm_samp = np.random.choice(np.where(labels == 0)[0])
        hot_samp = np.random.choice(np.where(labels == 1)[0])
        sathot_samp = np.random.choice(np.where(labels == 2)[0])
        cold_samp = np.random.choice(np.where(labels == 3)[0])
        satcold_samp = np.random.choice(np.where(labels == 4)[0])
        cosmic_samp = np.random.choice(np.where(labels == 5)[0])
        popcorn_samp = np.random.choice(np.where(labels == 6)[0])
        noisy_samp = np.random.choice(np.where(labels == 7)[0])

        plt.figure()
        plt.plot(features[norm_samp], label='Clean')
        plt.plot(features[hot_samp], label='Hot Pixel')
        plt.plot(features[sathot_samp], label='Saturated Hot Pixel')
        plt.plot(features[cold_samp], label='Cold Pixel')
        plt.plot(features[satcold_samp], label='Saturated Cold Pixel')
        plt.plot(features[cosmic_samp], label='Cosmic Ray')
        plt.plot(features[popcorn_samp], label='Popcorn Pixel')
        plt.plot(features[noisy_samp], label='Noisy Pixel')
        plt.xlim(0, n_timesteps)

        plt.ylabel('Electrons Read Off Detetor', fontsize=30)
        plt.xlabel(f'Group Number [0,{n_timesteps-1}]', fontsize=30)
        plt.legend(loc=0, fontsize=15, framealpha=.9)
        plt.title('Raw Pixels Values Read Per Group', fontsize=30)

        ax = plt.gcf().get_axes()[0]
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        plt.figure()
        std_sclr = StandardScaler()
        features_std = std_sclr.fit_transform(features)

        plt.plot(features_std[norm_samp], label='Clean')
        plt.plot(features_std[hot_samp], label='Hot Pixel')
        plt.plot(features_std[sathot_samp], label='Saturated Hot Pixel')
        plt.plot(features_std[cold_samp], label='Cold Pixel')
        plt.plot(features_std[satcold_samp], label='Saturated Cold Pixel')
        plt.plot(features_std[cosmic_samp], label='Cosmic Ray')
        plt.plot(features_std[popcorn_samp], label='Popcorn Pixel')
        plt.plot(features_std[noisy_samp], label='Noisy Pixel')
        plt.xlim(0, n_timesteps)

        plt.ylabel('Zero Mean Weighted Electrons', fontsize=30)
        plt.xlabel(f'Group Number [0,{n_timesteps-1}]', fontsize=30)
        plt.legend(loc=4, fontsize=15, framealpha=.9)
        plt.title('Zero Mean Weighted Pixel Values Per Group', fontsize=30)

        ax = plt.gcf().get_axes()[0]
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)

    if save_name is not None:
        # default = 'simulated_4M_bad_pixels_df.joblib.save'
        out_dict = {}
        out_dict['features'] = features
        out_dict['labels'] = labels

        joblib.dump(out_dict, save_name)
