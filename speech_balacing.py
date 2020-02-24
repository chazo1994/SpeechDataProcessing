#__author__ = thinhnv20
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys, os
import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("Speech_Balancing")

type_of_transforms = {"or": "Original", "bc": "Box-cox Transform", "yj": "Yeo-Johnson Transform", "qt": "Quantile Transform (GAUSS)"}

"""
Transform speech data into normal by using three method: Box-cox, Yeo-johnson and Quantile transform
- inputfile: this is input of amplitudes of data files
- outputs: there are amplitudes of data, box-cox transformed, yeo-johnson transformed, quantitle transformed
- if scale=True, transformed amplitudes will be scale into range 0-1
"""
def transform_amplitude(inputfile, scale = True):
    amplitudes = np.fromfile(inputfile, dtype=np.float)
    n_samples = amplitudes.shape[0]
    amplitudes = amplitudes.reshape((n_samples,-1))

    bc = PowerTransformer(method='box-cox')
    yj = PowerTransformer(method='yeo-johnson')
    qt = QuantileTransformer(n_quantiles=n_samples, output_distribution='normal')
    min_max_scaler = MinMaxScaler()

    bc_amplitudes = bc.fit_transform(amplitudes)
    yj_amplitudes = yj.fit_transform(amplitudes)
    qt_amplitudes = qt.fit_transform(amplitudes)

    if scale:
        bc_amplitudes = min_max_scaler.fit_transform(bc_amplitudes)
        yj_amplitudes = min_max_scaler.fit_transform(yj_amplitudes)
        qt_amplitudes = min_max_scaler.fit_transform(qt_amplitudes)


    return amplitudes, bc_amplitudes, yj_amplitudes, qt_amplitudes

def draw_amplitudes(id, amplitudes, type_of_transform, output_file = None, bins = 100):
    plt.figure(id)
    f, ax = plt.subplots(2)
    f.suptitle(type_of_transform)
    ax[0].plot(amplitudes)
    ax[0].set_ylabel("Amplitudes")

    ax[1].hist(x=amplitudes, bins= bins)
    ax[1].set_ylabel("Fequencies")
    ax[1].set_xlabel("Vol")

    plt.savefig(output_file)



if __name__=="__main__":
    inputfile = sys.argv[1].rstrip(os.path.sep)

    log.info("Start transform")
    amplitudes, bc_amplitudes, yj_amplitudes, qt_amplitudes = transform_amplitude(inputfile, scale=False)
    log.info("Transformation completed")
    intput_dir = os.path.dirname(inputfile)

    or_file = os.path.join(intput_dir, "Amplitudes_original.png")
    bc_file = os.path.join(intput_dir, "Amplitudes_bc.png")
    yj_file = os.path.join(intput_dir, "Amplitudes_yj.png")
    qt_file = os.path.join(intput_dir, "Amplitudes_qt.png")

    log.info("Draw normal output")
    draw_amplitudes(1, amplitudes, type_of_transforms["or"], or_file)
    draw_amplitudes(2, bc_amplitudes, type_of_transforms["bc"], bc_file)
    draw_amplitudes(3, yj_amplitudes, type_of_transforms["yj"], yj_file)
    draw_amplitudes(4, qt_amplitudes, type_of_transforms["qt"], qt_file)

    log.info("Start transform and scaling")
    _, bc_scaled_amplitudes, yj_scaled_amplitudes, qt_scaled_amplitudes = transform_amplitude(inputfile, scale=True)
    log.info("Transform and scaling completed")

    bc_scaled_file = os.path.join(intput_dir, "Amplitudes_bc_scaled.png")
    yj_scaled_file = os.path.join(intput_dir, "Amplitudes_yj_scaled.png")
    qt_scaled_file = os.path.join(intput_dir, "Amplitudes_qt_scaled.png")

    log.info("Draw scaled output")
    draw_amplitudes(5, bc_scaled_amplitudes, type_of_transforms["bc"], bc_scaled_file)
    draw_amplitudes(6, yj_scaled_amplitudes, type_of_transforms["yj"], yj_scaled_file)
    draw_amplitudes(7, qt_scaled_amplitudes, type_of_transforms["qt"], qt_scaled_file)

    log.info("Done")

