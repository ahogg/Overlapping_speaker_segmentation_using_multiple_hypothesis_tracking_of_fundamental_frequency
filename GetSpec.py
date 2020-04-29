import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import librosa.display


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

    def plot_tracks(self, samples, outfile, tracks, tx):

        y = librosa.stft(samples)

        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('axes', labelsize=9)

        width = 3.374
        height = width / 1.6

        fig, ax = plt.subplots(1, 1, sharey=False)
        fig.subplots_adjust(left=.08, bottom=.11, right=.97, top=.97)

        librosa.display.specshow(librosa.amplitude_to_db(y, ref=np.max, amin=1e-10, top_db=100.0), cmap=self.parula,
                                 y_axis='linear', x_axis='time')

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
                ax.plot(tx[start_idx:(len(fx) + start_idx)], [scale * x for x in fx], c='k', linewidth=0.7)

        fig.set_size_inches(width, height)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(outfile)

        return

def plot_spec_all_same_as_paper(self, infile, outfile, tracks_m1, tracks_m2, freqs, amps, list_subset, list_allsubset, tx, pv, starts, stops, best_states, changes, changes_2, changes_3, changes_4, states):

    possible_freqs_amps = [[(frame_freqs[i], amp) if amp > 0.0001e+09 else (float('nan'), amp) for i, amp in enumerate(amps[index])] for index, frame_freqs in enumerate(freqs)]
    s_possible_freqs_amps = [[x for x in s_possible_freq_amp if ~np.isnan(x[0])] for s_possible_freq_amp in possible_freqs_amps] # remove nan's
    s_possible_freqs = [[x[0] for x in s_possible_freq_amp] for s_possible_freq_amp in s_possible_freqs_amps]
    s_possible_amps = [[x[1] for x in s_possible_freq_amp] for s_possible_freq_amp in s_possible_freqs_amps]

    y, sr = librosa.load(infile)
    D = librosa.stft(y)

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('axes', labelsize=9)

    width = 3.374
    height = width*2.4
    set_top_db = 100

    # fig, axs = plt.subplots(9, 1, sharey=False)
    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(nrows=53, ncols=1)

    gs.update(wspace=0.1, hspace=0.01)
    # fig.subplots_adjust(left=0.0, bottom=.00, right=0.01, top=.01)

    ax1 = plt.subplot(gs[0:1, :])

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula, y_axis='linear',
                         x_axis='time')
    plt.gca().set_visible(False)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    plt.xlim([0, 3.55])

    ax0_divider = make_axes_locatable(ax1)
    cax0 = ax0_divider.append_axes("top", size="500%", pad="7%")
    clb = plt.colorbar( cax=cax0, orientation="horizontal") # format='%+2.0f dB'
    cax0.xaxis.set_ticks_position("top")
    clb.set_label('[dB]', labelpad=-27, x=-0.132, rotation=90)


    ax1 = plt.subplot(gs[2:10, :])
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula, y_axis='linear',
                         x_axis='time')
    plt.ylabel('(a) \n Frequency [kHz]')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    plt.xlim([0, 3.55])

    for time, s_possible_freq in enumerate(s_possible_freqs):
         ax1.scatter([tx[time]]*len(s_possible_freq), s_possible_freq, color='k', marker='x', s=0.1)

    plt.subplot(gs[11:19, :])
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula, y_axis='linear',
                         x_axis='time')
    plt.ylabel('(b) \n Frequency [kHz]')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    for t_idx, subsets in enumerate(list_allsubset):
        for subset in subsets:
            cmap = np.linspace(0.0, 1.0, 12)
              # rgba_colors = cmap[len(subset)]
            # print(len(subset))
            rgba_colors = [cmap[len(subset)], cmap[len(subset)], cmap[len(subset)]]
            # print(rgba_colors)
            # ax.scatter(tx[t_idx], np.mean([f/(subset[id][2]+1) for id, f in enumerate([x[0] for x in subset])]), c=rgba_colors, marker='x', alpha=1, s=0.02)
            plt.scatter(tx[t_idx], np.mean([f / (subset[id][2] + 1) for id, f in enumerate([x[0] for x in subset])]), c='k', marker='x', alpha=1, s=0.2)

    plt.subplot(gs[20:28, :])
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula,
                             y_axis='linear', x_axis='time')
    plt.ylabel('(c) \n Frequency [kHz]')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    plt.xlim([0, 3.55])

    plt.axhline(1030, color='k', linestyle='-', label='', linewidth=8.0, alpha=0.3, zorder=1)
    plt.axhline(830, color='k', linestyle='-', label='', linewidth=8.0, alpha=0.3, zorder=1)

    for change in changes:
        plt.arrow(change[0] + 0.15, 1030, change[1]-change[0]-0.2, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w', length_includes_head=True)
        plt.arrow(change[0] + 0.15, 1030, -0.1, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w',
                  length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=1.0)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=1.0)

    for change in changes_2:
        plt.arrow(change[0] + 0.15, 830, change[1]-change[0]-0.2, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w', length_includes_head=True)
        plt.arrow(change[0] + 0.15, 830, -0.1, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w',
                  length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=1.0)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=1.0)

    for change in changes_3:
        plt.arrow(change[0]+ 0.2, 930, change[1]-change[0]-0.3, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.arrow(change[0]+ 0.2, 930, -0.1, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=0.7)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=0.7)

    for change in changes_4:
        plt.arrow(change[0]+ 0.2, 830, change[1]-change[0]-0.3, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.arrow(change[0]+ 0.2, 830, -0.1, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=0.7)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=0.7)

    plt.text(0.05, 1030, 'Speaker 1', fontsize=7, color='w', horizontalalignment='left', verticalalignment='center')
    plt.text(0.05, 830, 'Speaker 2', fontsize=7, color='w', horizontalalignment='left', verticalalignment='center')

    for track in tracks_m1:
        for scale in [1]:
            fx = [x[0][0] for x in track.get_past_states()]
            start_idx = track.get_start_index()
            plt.plot(tx[start_idx:(len(fx)+(start_idx))], [scale*x for x in fx], c='k', linewidth=1.0)


    plt.subplot(gs[29:37, :])
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula, y_axis='linear',
                         x_axis='time')
    plt.ylabel('(d) \n Frequency [kHz]')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    plt.xlim([0, 3.55])

    for t_idx, subsets in enumerate(list_subset):
        for subset in subsets:
            cmap = np.linspace(0.0, 1.0, 12)
              # rgba_colors = cmap[len(subset)]
            # print(len(subset))
            rgba_colors = [cmap[len(subset)], cmap[len(subset)], cmap[len(subset)]]
            # print(rgba_colors)
            # ax.scatter(tx[t_idx], np.mean([f/(subset[id][2]+1) for id, f in enumerate([x[0] for x in subset])]), c=rgba_colors, marker='x', alpha=1, s=0.02)
            plt.scatter(tx[t_idx], np.mean([f / (subset[id][2] + 1) for id, f in enumerate([x[0] for x in subset])]), c='k', marker='x', alpha=1, s=0.2)

    ax6 = plt.subplot(gs[39:41, :])
    plt.axhline(y=0.5, color='k', linestyle='-', linewidth=1.0)
    for change in changes+changes_2+changes_3+changes_4:
        plt.axvline(x=change[0], color='gray', linestyle='-', alpha=0.2, linewidth=20)
        plt.axvline(x=change[1], color='gray', linestyle='-', alpha=0.2, linewidth=20)
        plt.axvline(x=change[0], color='#0d5593', linestyle=':')
        plt.axvline(x=change[1], color='#0d5593', linestyle=':')

    plt.text(changes[0][0], 1.15, 'HIT', fontsize=7, color='k', horizontalalignment='center', verticalalignment='center')
    plt.text(changes[0][1], 1.15, 'HIT', fontsize=7, color='k', horizontalalignment='center', verticalalignment='center')
    plt.text(changes_2[0][0], 1.15, 'HIT', fontsize=7, color='k', horizontalalignment='center', verticalalignment='center')
    plt.text(changes_2[0][1], 1.15, 'MH', fontsize=7, color='k', horizontalalignment='center', verticalalignment='center')
    plt.text(tx[tracks_m2[0].get_start_index()], 1.15, 'FA', fontsize=7, color='k', horizontalalignment='center', verticalalignment='center')

    for track in tracks_m2:
        for scale in [1]:
            fx = [x[0][0] for x in track.get_past_states()]
            start_idx = track.get_start_index()
            plt.arrow(tx[start_idx], 1, 0, -0.43,  linewidth=1.5, color='#C61616', head_width=0.03, head_length=0.05, length_includes_head=True, zorder=4)
            plt.arrow(tx[len(fx)+(start_idx)], 1, 0, -0.43,  linewidth=1.5, color='#C61616', head_width=0.03, head_length=0.05, length_includes_head=True, zorder=4)

    # plt.axis('off')
    plt.setp(ax6.spines.values(), visible=False)
    ax6.xaxis.set_visible(False)
    plt.ylabel('(f)', labelpad=31.45)
    plt.xlabel('')
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1])
    plt.xlim([0, 3.55])



    plt.subplot(gs[42:51, :])
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max, amin=1e-10, top_db=set_top_db), cmap=self.parula,
                             y_axis='linear', x_axis='time')
    plt.ylabel('(e) \n Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.yticks(np.arange(200, 1100, 200), ('0.2', '0.4', '0.6', '0.8', '1'))
    plt.ylim([0, 1200])
    plt.xlim([0, 3.55])

    for track in tracks_m2:
        colour = np.random.rand(3,)
        for scale in [1]:
            fx = [x[0][0] for x in track.get_past_states()]
            start_idx = track.get_start_index()
            plt.plot(tx[start_idx:(len(fx)+(start_idx))], [scale*x for x in fx], c='k', linewidth=1.0)

    plt.axhline(1030, color='k', linestyle='-', label='', linewidth=8.0, alpha=0.3, zorder=1)
    plt.axhline(830, color='k', linestyle='-', label='', linewidth=8.0, alpha=0.3, zorder=1)

    for change in changes:
        plt.arrow(change[0] + 0.15, 1030, change[1]-change[0]-0.2, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w', length_includes_head=True)
        plt.arrow(change[0] + 0.15, 1030, -0.1, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w',
                  length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=1.0)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=1.0)

    for change in changes_2:
        plt.arrow(change[0] + 0.15, 830, change[1]-change[0]-0.2, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w', length_includes_head=True)
        plt.arrow(change[0] + 0.15, 830, -0.1, 0, head_width=20, head_length=0.05, linewidth=1.0, color='w',
                  length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=1.0)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=1.0)

    for change in changes_3:
        plt.arrow(change[0]+ 0.2, 930, change[1]-change[0]-0.3, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.arrow(change[0]+ 0.2, 930, -0.1, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=0.7)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=0.7)

    for change in changes_4:
        plt.arrow(change[0]+ 0.2, 830, change[1]-change[0]-0.3, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.arrow(change[0]+ 0.2, 830, -0.1, 0, head_width=15, head_length=0.1, linewidth=0.7, color='w', length_includes_head=True)
        plt.axvline(x=change[0], c='w', linestyle=':', linewidth=0.7)
        plt.axvline(x=change[1], c='w', linestyle=':', linewidth=0.7)

    plt.text(0.05, 1030, 'Speaker 1', fontsize=7, color='w', horizontalalignment='left', verticalalignment='center')
    plt.text(0.05, 830, 'Speaker 2', fontsize=7, color='w', horizontalalignment='left', verticalalignment='center')


    fig.set_size_inches(width, height)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0.01)

    return
