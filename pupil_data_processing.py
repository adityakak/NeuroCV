import pandas as pd
import numpy as np
import math
import os

interval_size = 82500  # ms
confidence_cutoff = .5

name = "Anon"
var = 1  #1 dark 0 light

rel_path_input = "dark" if var else "blue"
rel_path_output = "dark" if var else "blue"

script_dir = os.path.dirname(__file__)

abs_path_input = os.path.join(script_dir, name + "_Data/pupil_positions_" + rel_path_input + ".csv")
abs_path_output = os.path.join(script_dir,  name + "_Data/pupil_output_" + rel_path_output+ ".csv")

csv_filepath = abs_path_input
csv_output_filepath = abs_path_output


def read_in_csv(filepath):
    """
    Reads in the .csv file created by Pupil Core and outputs a numpy array with data to be used in PUI Calculation
    """
    pupil_positions_data = pd.read_csv(filepath)
    pupil_positions_df = pd.DataFrame(pupil_positions_data,
                                      columns=['pupil_timestamp', 'eye_id', 'diameter', 'confidence'])
    numpy_data = pupil_positions_df.to_numpy()
    return numpy_data


def remove_eye_data(data, wanted_eye):
    """
    Removes data from the eye which we do not want the data from and data with lower than wanted confidence
    """
    data = data[np.logical_not(data[:, 1] == wanted_eye)]
    data = data[np.logical_not(data[:, 3] < confidence_cutoff)]
    data = data[np.logical_not(data[:, 2] > 100)]
    # data = data[np.logical_not()]
    return data


def find_nearest(array, value):
    """
    Locate index closes to the value we are looking for
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def find_intervals(data):
    """
    Determine the intervals within which we will find the mean segmentation for PUI calculation
    """
    search = prev = 0
    interval = [prev]
    time_interval = [data[prev][0]]
    while search < len(data) - 1:
        interval_growth = 0
        while search == prev:
            search = find_nearest(data[:, 0], data[prev][0] + ((interval_size + interval_growth) / 1000))
            interval_growth += 1
        prev = search
        interval.append(int(prev))
        time_interval.append(data[prev][0])
    return interval, time_interval


def create_means(data, intervals):
    """
    Find the means of the previously found intervals. Serves as a high-pass filter
    """
    means = []
    for i in range(len(intervals) - 1):
        means.append(data[intervals[i]:intervals[i + 1]].mean(axis=0)[2])
    return means


def sum_absolute_dif(means_list, record_length):
    """
    Takes the list of means and returns the sum of the absolute differences between them
    """
    absolute_dif = []
    total = 0
    for i in range(len(means_list) - 1):
        absolute_dif.append(abs(means_list[i] - means_list[i + 1]))
        total += abs(means_list[i] - means_list[i + 1])
    return absolute_dif, total / record_length


def main():
    """
    Determines the PUI and outputs the mean, and list of means to a .csv file named pupil_output_data
    """

    csv_data = read_in_csv(csv_filepath)

    df = pd.DataFrame()
    csv_headers = ["Eye Data 0", "Eye PUI 0", "Eye Mean 0", "Eye Data 1", "Eye PUI 1", "Eye Mean 1", "Better Eye"]
    to_csv_data_list = []

    num_data_points = []
    better_means = []
    better_pui = []

    for i in range(2):
        data = np.copy(csv_data)

        data = remove_eye_data(data, i)

        num_data_points.append(data.shape[0])

        recording_length = (data[len(data) - 1][0] - data[0][0]) / 60  # min

        intervals, time_intervals = find_intervals(data)

        means_list = create_means(data, intervals)
        absolute_list, pui = sum_absolute_dif(means_list, recording_length)

        to_csv_data_list.append(means_list)
        to_csv_data_list.append([pui])
        to_csv_data_list.append([str(np.mean(np.array(means_list)))])

        better_pui.append(pui)
        better_means.append(means_list)

    best_eye = False  # False -> Eye 0, True -> Eye 1
    if num_data_points[1] > num_data_points[0]:
        best_eye = not best_eye
    to_csv_data_list.append([int(best_eye)])

    max_list_length = max([len(i) for i in to_csv_data_list])
    for idx, i in enumerate(to_csv_data_list):
        i += ["" for _ in range(max_list_length - len(i))]
        df[csv_headers[idx]] = i
    df.to_csv(csv_output_filepath)
    print(df.iloc[0].drop(labels=['Eye Data 0', 'Eye Data 1']))

    print(better_pui[best_eye])
    print(better_means[best_eye])


if __name__ == "__main__":
    main()
