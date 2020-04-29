import numpy as np
import math
import matlab.engine
import librosa.display
from copy import deepcopy
from weighted_graph import WeightedGraph
from KalmanFilterHarmonic import KalmanFilterHarmonic as kf
from GetSpec import GetSpec as GS
import multiprocessing
import time


class Timer:
    def __init__(self):
        self.time = time.time()

    def lap(self, name):
        now = time.time()
        print(name + ': ' + str(now - self.time))
        self.time = now

def global_hypothesis(track_trees, t):

    MWIS_tracks = []
    gh_graph = WeightedGraph()

    tracks = [item for sublist in track_trees for item in sublist]

    if tracks:

        conflicting_tracks = []

        for track_index, track in enumerate(tracks):
            for conflict_index, conflict_track in enumerate(tracks[track_index:]):
                conflict_index += track_index

                if any(x in track.get_past_measurements() for x in conflict_track.get_past_measurements()):
                    if track_index != conflict_index:
                        conflicting_tracks.append((track_index, conflict_index))

        for index, kalman_filter in enumerate(tracks):
            gh_graph.add_weighted_vertex(str(index), kalman_filter.get_score(t))

        gh_graph.set_edges(conflicting_tracks)

        mwis_ids = gh_graph.mwis()

        for index in list(mwis_ids):
            MWIS_tracks.append([tracks[index]])
        
        for track_index, track in enumerate(MWIS_tracks):
            track = track[0]
            for conflict_index, conflict_track in enumerate(MWIS_tracks):
                conflict_track = conflict_track[0]
                if abs(track.get_state()-conflict_track.get_state())<10 and conflict_track.is_con() and track.is_con():
                    if track_index != conflict_index:
                        if track.get_score(t) > conflict_track.get_score(t):
                            track.kill()

    return MWIS_tracks


def update_tracks((t, node_tracks, subsets, transition_model, R)):
    updated_tracks = []

    for track in node_tracks:
        if track.isactive():
            if track.get_percentage_of_updates() < 0.55 and t - track.get_past_measurements()[-1][0] > 20:
                track_update = deepcopy(track)
                track_update.kill()
                updated_tracks.append(track_update)

            updated_tracks_observation = []
            for z_idx, subset in enumerate(subsets):
                h = np.array([np.array([harmonic + 1]) for (pitch, amp, harmonic, index) in subset])
                z = np.array([[pitch] for (pitch, amp, harmonic, index) in subset])
                r = np.diag(R * len(h))
                w = sum(np.array([[amp] for (pitch, amp, harmonic, index) in subset]))
                track_update = deepcopy(track)
                track_update.prediction(transition_model)
                track_update.update(t, z, z_idx, h, r, w)
                if np.mean(np.array([abs(x) for x in track_update.get_post_fit_residual()])) < 40:
                    updated_tracks_observation.append(track_update)

            updated_tracks += updated_tracks_observation

            track_update = deepcopy(track)
            track_update.prediction(transition_model)
            track_update.false_alarm(10)
            state = track_update.get_state()
            track_update.push_state(state)
            updated_tracks.append(track_update)
        else:
            updated_tracks.append(track)

    return updated_tracks



def harmonic_track_kalman_filter(list_subset):

    ###############################################################
    # Harmonic Track Kalman filter
    ###############################################################

    # initialise Kalman filter
    p = np.array([[100]])
    R = [math.sqrt(300)]
    q = math.sqrt(0.001)

    con_obs_num = 7

    current_tracks = []
    all_current_tracks = []

    ###############################################################

    last_measurement_index = len(list_subset)-1
    for idx, list in enumerate(reversed(list_subset)):
        if not list:
            last_measurement_index = len(list_subset)-idx-1
        else:
            break

    for t in np.arange(0, last_measurement_index):

        transition_model = np.array([[1]])

        subsets = list_subset[t]

        if not subsets:
            for node_tracks in current_tracks:
                for track in node_tracks:
                    track.prediction(transition_model)
                    state = track.get_state()
                    track.push_state(state)
        else:

            temp_tracks = deepcopy(current_tracks)
            current_tracks = []

            for node_tracks in temp_tracks:
                updated_tracks = update_tracks((t, node_tracks, subsets, transition_model, R))
                current_tracks = current_tracks + [updated_tracks]


            for z_idx, subset in enumerate(subsets):

                h = np.array([np.array([harmonic + 1]) for (pitch, amp, harmonic, index) in subset])
                z = np.array([[pitch] for (pitch, amp, harmonic, index) in subset])
                r = np.diag(R * len(h))
                w = sum(np.array([[amp] for (pitch, amp, harmonic, index) in subset]))

                x = np.mean([measurement / h[index][0] for index, measurement in enumerate(z)])
                new_track = kf(con_obs_num, x, p, q, r, t)
                new_track.set_start_index(t)
                new_track.false_alarm(0)  # penalty for small tracks
                new_track.prediction(transition_model)
                new_track.update(t, z, z_idx, h, r, w)
                current_tracks.append([new_track])

        continue_tracks = global_hypothesis(current_tracks, t)

        current_tracks = []
        for track in continue_tracks:
            if track[0].isactive():
                current_tracks.append(track)
            else:
                all_current_tracks.append(track)

    all_current_tracks += current_tracks

    current_tracks_list = [item for sublist in all_current_tracks for item in sublist]
    sorted_current_tracks = sorted(current_tracks_list, key=lambda obj: obj.get_score(t))

    track_scores = [track.get_score(t) for track in sorted_current_tracks]

    possible_tracks = [sorted_current_tracks[i] for i, score in enumerate(track_scores)]

    confirmed_tracks_m2 = []
    for track in possible_tracks:
        if track.is_con():  # remove non-confirmed tracks
            confirmed_tracks_m2.append(track)

    possible_tracks_m2 = confirmed_tracks_m2

    return possible_tracks_m2


def select_subsets(subsets):

    best_subsets = []

    if subsets:

        conflicting_subsets = []

        for subset_index, subset in enumerate(subsets[:-1]):
            subset_mean_f0 = np.mean([x[0] / (x[2] + 1) for x in subset])
            for conflict_index, conflict_subset in enumerate(subsets[subset_index+1:]):
                conflict_index += subset_index + 1
                conflict_subset_mean_f0 = np.mean([x[0] / (x[2] + 1) for x in conflict_subset])
                harmonics_of_conflict_subset = [conflict_subset_mean_f0 * x for x in [0.25, 0.5, 1.5] + list(np.arange(1, 10))]
                if True in [abs(subset_mean_f0-x) < 3 for x in harmonics_of_conflict_subset]:
                    conflicting_subsets.append((subset_index, conflict_index))

        conflicting_subsets = {tuple(item) for item in map(sorted, conflicting_subsets)}
        weighted_subsets = [(index, len(subset)) for index, subset in enumerate(subsets)]

        while weighted_subsets:
            max_weight = max(weighted_subsets, key=lambda item: item[1])[1]
            possible_best_candidates = [item[0] for item in weighted_subsets if item[1] == max_weight]

            mean_f0s = [(index, np.mean([x[0] / (x[2] + 1) for x in subsets[index]])) for index in possible_best_candidates]
            possible_best_candidate = max(mean_f0s, key=lambda item: item[1])[0]

            best_subsets.append(subsets[possible_best_candidate])

            subsets_to_remove = [item[1] for item in conflicting_subsets if item[0] == possible_best_candidate]
            subsets_to_remove += [item[0] for item in conflicting_subsets if item[1] == possible_best_candidate]
            subsets_to_remove.append(possible_best_candidate)

            weighted_subsets = [x for x in weighted_subsets if x[0] not in subsets_to_remove]

    return best_subsets


def find_subset((frame_freqs, frame_amps)):
    possible_freqs_amps = [(frame_freqs[i], amp) for i, amp in enumerate(frame_amps) if amp > 0.0001e+09]
    possible_subsets = []
    for (freq, amp) in possible_freqs_amps:
        n = 0
        f_0 = freq

        while 70 < f_0 and n < 3:
            n = n + 1
            f_0 = freq / n

        possible_f_0 = [freq/x for x in np.arange(1, n) if freq/x < 300]

        for f_0 in possible_f_0:
            if np.any(np.isclose(f_0, frame_freqs, atol=5)):
                possible_subset = []
                for idx, (est, est_amp) in enumerate(possible_freqs_amps):
                    if abs((round(est / f_0) * f_0) - est) < 10:
                        possible_subset.append((possible_freqs_amps[idx][0], possible_freqs_amps[idx][1], int(round(est/f_0))-1, idx))
                if (possible_subset not in possible_subsets) and len(possible_subset) > 1:
                    possible_subsets.append(possible_subset)

    return possible_subsets


def get_pitch_features(samples, fs):

    time = Timer()

    eng = matlab.engine.start_matlab()
    fx, tx, pv, amp = eng.gen_peak_track_large_array(
        matlab.double(list(samples)),
        matlab.double([fs]), nargout=4)
    eng.quit()

    time.lap('matlab')

    p = multiprocessing.Pool(4)
    list_allsubset = p.map(find_subset, zip(fx, amp))

    # list_allsubset_o = []
    # for idx in np.arange(0, len(fx)):
    #     list_allsubset_o.append(find_subset((fx[idx], amp[idx])))

    time.lap('subsets')

    # list_subset_o = []
    # for idx in np.arange(0, len(fx)):
    #     list_subset_o.append(select_subsets(list_allsubset[idx]))

    list_subset = p.map(select_subsets, list_allsubset)

    time.lap('selection')

    possible_tracks = harmonic_track_kalman_filter(list_subset)

    time.lap('tracking')

    fea = np.zeros((len(fx), 26))
    for track in possible_tracks:
        fx = [x[0][0] for x in track.get_past_states()]
        bins = np.linspace(50, 300, 26)
        digitized = np.digitize(fx, bins)
        start_idx = track.get_start_index()
        for index, t in enumerate(np.arange(start_idx, start_idx + len(fx))):
            pitch_bin = digitized[index]
            fea[t][pitch_bin] = 1

    time.lap('feature generation')

    outfile_pdf = 'images/TS3003b_mix_headset_snippet.pdf'
    spectrogram = GS()
    spectrogram.plot_tracks(samples, outfile_pdf, possible_tracks, tx)
    # spectrogram.plot_all(samples, outfile_pdf, [], [], fx, amp, list_subset, list_allsubset, tx)

    time.lap('plot')

    return fea


s, fs = librosa.load("audio/TS3003b_mix_headset_snippet.wav")
# fs, s = wavfile.read('audio/EN2002a_mix_headset_long_snippet.wav.wav')
# fxpefac_peak(s, fs)
fea = get_pitch_features(s, fs)

