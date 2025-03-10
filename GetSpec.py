import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import scipy
from scipy import signal

plt.rcParams['legend.fancybox'] = False


class GetSpec:

    def __init__(self):
        self.cm_data = [[0.2081, 0.1663, 0.5292],
                        [0.2116238095, 0.1897809524, 0.5776761905],
                        [0.212252381, 0.2137714286, 0.6269714286],
                        [0.2081, 0.2386, 0.6770857143],
                        [0.1959047619, 0.2644571429, 0.7279],
                        [0.1707285714, 0.2919380952, 0.779247619],
                        [0.1252714286, 0.3242428571, 0.8302714286],
                        [0.0591333333, 0.3598333333, 0.8683333333],
                        [0.0116952381, 0.3875095238, 0.8819571429],
                        [0.0059571429, 0.4086142857, 0.8828428571],
                        [0.0165142857, 0.4266, 0.8786333333],
                        [0.032852381, 0.4430428571, 0.8719571429],
                        [0.0498142857, 0.4585714286, 0.8640571429],
                        [0.0629333333, 0.4736904762, 0.8554380952],
                        [0.0722666667, 0.4886666667, 0.8467],
                        [0.0779428571, 0.5039857143, 0.8383714286],
                        [0.079347619, 0.5200238095, 0.8311809524],
                        [0.0749428571, 0.5375428571, 0.8262714286],
                        [0.0640571429, 0.5569857143, 0.8239571429],
                        [0.0487714286, 0.5772238095, 0.8228285714],
                        [0.0343428571, 0.5965809524, 0.819852381],
                        [0.0265, 0.6137, 0.8135],
                        [0.0238904762, 0.6286619048, 0.8037619048],
                        [0.0230904762, 0.6417857143, 0.7912666667],
                        [0.0227714286, 0.6534857143, 0.7767571429],
                        [0.0266619048, 0.6641952381, 0.7607190476],
                        [0.0383714286, 0.6742714286, 0.743552381],
                        [0.0589714286, 0.6837571429, 0.7253857143],
                        [0.0843, 0.6928333333, 0.7061666667],
                        [0.1132952381, 0.7015, 0.6858571429],
                        [0.1452714286, 0.7097571429, 0.6646285714],
                        [0.1801333333, 0.7176571429, 0.6424333333],
                        [0.2178285714, 0.7250428571, 0.6192619048],
                        [0.2586428571, 0.7317142857, 0.5954285714],
                        [0.3021714286, 0.7376047619, 0.5711857143],
                        [0.3481666667, 0.7424333333, 0.5472666667],
                        [0.3952571429, 0.7459, 0.5244428571],
                        [0.4420095238, 0.7480809524, 0.5033142857],
                        [0.4871238095, 0.7490619048, 0.4839761905],
                        [0.5300285714, 0.7491142857, 0.4661142857],
                        [0.5708571429, 0.7485190476, 0.4493904762],
                        [0.609852381, 0.7473142857, 0.4336857143],
                        [0.6473, 0.7456, 0.4188],
                        [0.6834190476, 0.7434761905, 0.4044333333],
                        [0.7184095238, 0.7411333333, 0.3904761905],
                        [0.7524857143, 0.7384, 0.3768142857],
                        [0.7858428571, 0.7355666667, 0.3632714286],
                        [0.8185047619, 0.7327333333, 0.3497904762],
                        [0.8506571429, 0.7299, 0.3360285714],
                        [0.8824333333, 0.7274333333, 0.3217],
                        [0.9139333333, 0.7257857143, 0.3062761905],
                        [0.9449571429, 0.7261142857, 0.2886428571],
                        [0.9738952381, 0.7313952381, 0.266647619],
                        [0.9937714286, 0.7454571429, 0.240347619],
                        [0.9990428571, 0.7653142857, 0.2164142857],
                        [0.9955333333, 0.7860571429, 0.196652381],
                        [0.988, 0.8066, 0.1793666667],
                        [0.9788571429, 0.8271428571, 0.1633142857],
                        [0.9697, 0.8481380952, 0.147452381],
                        [0.9625857143, 0.8705142857, 0.1309],
                        [0.9588714286, 0.8949, 0.1132428571],
                        [0.9598238095, 0.9218333333, 0.0948380952],
                        [0.9661, 0.9514428571, 0.0755333333],
                        [0.9763, 0.9831, 0.0538]]

        self.parula = LinearSegmentedColormap.from_list('parula', self.cm_data)


    def plot_wave(self, samples, fs, outfile):

        fx, tv, spectrogram = scipy.signal.spectrogram(samples, fs=fs, window='hanning',
                                                      nperseg=1024, noverlap=1024 - 100,
                                                      detrend=False, scaling='spectrum')

        D = 10 * np.log10(spectrogram)

        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('axes', labelsize=9)

        width = 3.374
        height = width / 1.6

        fig, ax = plt.subplots(1, 1, sharey=False)
        fig.subplots_adjust(left=.08, bottom=.11, right=.97, top=.97)

        plt.imshow(D, extent=[0, tv[-1], fx[0], fx[-1]],
                   vmin=0, vmax=45, origin='lowest', aspect='auto', cmap=self.parula)

        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')

        plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
        plt.ylim([0, 1200])

        ax2_divider = make_axes_locatable(ax)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        clb = plt.colorbar(cax=cax2, orientation="horizontal")  # format='%+2.0f dB'
        cax2.xaxis.set_ticks_position("top")
        clb.set_label('[dB]', labelpad=-25, x=-0.132, rotation=90)

        fig.set_size_inches(width, height)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(outfile)

        return

    def plot_peaks(self, samples, fs, outfile, peaks, amp, tx, amp_min):

        fx, tv, spectrogram = scipy.signal.spectrogram(samples, fs=fs, window='hanning',
                                                      nperseg=1024, noverlap=1024 - 100,
                                                      detrend=False, scaling='spectrum')
        D = 10 * np.log10(spectrogram)

        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('axes', labelsize=9)

        width = 3.374
        height = width / 1.6

        fig, ax = plt.subplots(1, 1, sharey=False)
        fig.subplots_adjust(left=.08, bottom=.11, right=.97, top=.97)

        plt.imshow(D, extent=[0, tv[-1], fx[0], fx[-1]],
                   vmin=0, vmax=45, origin='lowest', aspect='auto', cmap=self.parula)

        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')

        plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
        plt.ylim([0, 1200])

        ax2_divider = make_axes_locatable(ax)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        clb = plt.colorbar(cax=cax2, orientation="horizontal")  # format='%+2.0f dB'
        cax2.xaxis.set_ticks_position("top")

        clb.set_label('[dB]', labelpad=-25, x=-0.132, rotation=90)

        for i, peak in enumerate(peaks):
            peak_n = [peak[idx] for idx, a in enumerate(amp[i]) if a > amp_min]  # 0.001 ]# 0.0001e+09]
            ax.scatter([tx[i]]*len(peak_n), peak_n, c='w', marker='x', alpha=1, s=1)

        fig.set_size_inches(width, height)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(outfile)

        return


    def plot_list_allsubset(self, samples, fs, outfile, list_allsubset, tx):

        fx, tv, spectrogram = scipy.signal.spectrogram(samples, fs=fs, window='hanning',
                                                      nperseg=1024, noverlap=1024 - 100,
                                                      detrend=False, scaling='spectrum')
        D = 10 * np.log10(spectrogram)

        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('axes', labelsize=9)

        width = 3.374
        height = width / 1.6

        fig, ax = plt.subplots(1, 1, sharey=False)
        fig.subplots_adjust(left=.08, bottom=.11, right=.97, top=.97)

        plt.imshow(D, extent=[0, tv[-1], fx[0], fx[-1]],
                   vmin=0, vmax=45, origin='lowest', aspect='auto', cmap=self.parula)

        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')

        plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
        plt.ylim([0, 1200])

        ax2_divider = make_axes_locatable(ax)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        clb = plt.colorbar(cax=cax2, orientation="horizontal")  # format='%+2.0f dB'
        cax2.xaxis.set_ticks_position("top")

        clb.set_label('[dB]', labelpad=-25, x=-0.132, rotation=90)

        for i, list in enumerate(list_allsubset):
            for subset in list:
                ax.scatter(tx[i], np.mean([f / (subset[id][2] + 1) for id, f in enumerate([x[0] for x in subset])]),
                       c='w', marker='x', alpha=1, s=0.2)

        fig.set_size_inches(width, height)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(outfile)

        return

    def plot_tracks(self, samples, fs, outfile, tracks, tx):

        fx, tv, spectrogram = scipy.signal.spectrogram(samples, fs=fs, window='hanning',
                                                      nperseg=1024, noverlap=1024 - 100,
                                                      detrend=False, scaling='spectrum')
        D = 10 * np.log10(spectrogram)

        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('axes', labelsize=9)

        width = 3.374
        height = width / 1.6

        fig, ax = plt.subplots(1, 1, sharey=False)
        fig.subplots_adjust(left=.08, bottom=.11, right=.97, top=.97)

        plt.imshow(D, extent=[0, tv[-1], fx[0], fx[-1]],
                   vmin=0, vmax=45, origin='lowest', aspect='auto', cmap=self.parula)

        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')

        plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
        plt.ylim([0, 1200])

        ax2_divider = make_axes_locatable(ax)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        clb = plt.colorbar(cax=cax2, orientation="horizontal")  # format='%+2.0f dB'
        cax2.xaxis.set_ticks_position("top")

        clb.set_label('[dB]', labelpad=-25, x=-0.132, rotation=90)

        for track in tracks:
            for scale in [1]:
                fx = [x[0][0] for x in track.get_past_states()]
                start_idx = track.get_start_index()
                ax.plot(tx[start_idx:(len(fx) + start_idx)], [scale * x for x in fx], c='w', linewidth=1.2)

        fig.set_size_inches(width, height)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(outfile)

        return

