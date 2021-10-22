import numpy as np
import matplotlib.pyplot as plt
from CaImaging.Miniscope import open_minian, get_transient_timestamps

def compare_transient_counts(folders, labels):
    data = {label: open_minian(folder) for folder, label in zip(folders, labels)}

    event_times = dict()
    event_mags = dict()
    bool_arrs = dict()
    transient_rate = dict()
    avg_event_mag = dict()
    for label in labels:
        event_times[label], event_mags[label], bool_arrs[label] = get_transient_timestamps(data[label].S)
        transient_rate[label] = [len(neuron) / bool_arrs[label].shape[1] / 30 for neuron in event_times[label]]
        avg_event_mag[label] = [np.mean(neuron) for neuron in event_mags[label]]

    fig, axs = plt.subplots(1,2)
    axs[0].boxplot([transient_rate[label] for label in labels])
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel('Transient rate [Hz]')

    axs[1].boxplot([avg_event_mag[label] for label in labels])
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel('Average transient magnitude [a.u.]')

    fig.tight_layout()

    return transient_rate, avg_event_mag

if __name__ == '__main__':
    compare_transient_counts([r'Z:\Will\IntranasalPeptide\Data\Grus\2021_10_11\10_16_06\Miniscope',
                              r'Z:\Will\IntranasalPeptide\Data\Grus\2021_10_11\11_25_42\Miniscope',
                              r'Z:\Will\IntranasalPeptide\Data\Grus\2021_10_11\12_31_46\Miniscope'],
                             ['baseline', 'saline', 'peptide'])
