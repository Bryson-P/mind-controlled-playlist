# Bryson Pelechaty Computer Science Capstone Research Project
# Purpose of program: To analyze and visualize EEG (Electroencephalography) data of indivduals listening to music.
# Functions included: Cleans, pre-processes, visualizes and analyzes EEG Data

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
import uuid
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix
import sklearn
# from os import listdir
import os
from sklearn.ensemble import RandomForestClassifier


# % matplotlib inline

# Note about data:
# Each CSV file used in this project is an 8 by N CSV file, with the data measuring a value of voltage. Each datapoint corresponds with a sensor on the headset.


def clean1(filename):  # Method one of reading in the EEG CSV File. Used when visualizing
    data = pd.read_csv(filename, skiprows=4)
    data.columns = data.columns.str.strip()
    return data


# Second method of reading in an EEG CSV file. Used when processing the data.
def clean2(filename):
    data = pd.read_csv(filename, sep=",", header=None, skiprows=5005).values

    return data


def clean3(filename):
    data = pd.read_csv(filename)

    data = data.drop("Epoch Time", axis=1)
    data = data.drop("Timestamp", axis=1)

    data = data.values
    return data


def get_data():
    arr = os.listdir(
        "/Users/brysonpelechaty/PycharmProjects/pythonProject1/Data3")
    arr1 = []

    for file in os.listdir('/Users/brysonpelechaty/PycharmProjects/pythonProject1/Data3'):
        if file.endswith(".txt"):
            # print(os.path.join('/Users/brysonpelechaty/PycharmProjects/pythonProject1/Data', file))
            arr1.append(os.path.join(
                '/Users/brysonpelechaty/PycharmProjects/pythonProject1/Data3', file))

    for i in range(len(arr1)):
        data = clean1(arr1[i])

    data.head()

    return arr1
    # data=clean2(temp)


def search_string(filename):
    s1 = "ex01"
    s2 = "ex02"
    s3 = "ex03"
    s4 = "ex04"
    s5 = "ex05"
    s6 = "ex06"
    s7 = "ex07"
    s8 = "ex08"
    s9 = "ex09"

    check = [s1, s2, s3, s4, s5, s6, s7, s8, s9]

    fullstring = filename

    for i in range(len(check)):
        substring = check[i]
        if fullstring.find(substring) != -1:
            # print("Found")
            return substring

        # else:
        # print("Not found")


fs = 1000
band = (15, 50)


def notch(val, data, fs=200):
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        fin = data = signal.lfilter(b, a, data)
    return fin


def bandpass(start, stop, data, fs=200):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data, axis=0)


def fft(data, fs):
    L = len(data)
    freq = np.linspace(0.0, 1.0 / (2.0 * fs ** -1), L // 2)
    yi = np.fft.fft(data)[1:]
    y = yi[range(int(L / 2))]
    return freq, abs(y)


def fft_filter(data):
    channels = []

    bandpass_channels = []
    bandpass_notch_channels = []
    notch_channels = []
    fs = 1000
    band = (15, 50)

    for i in range(8):
        channels.append(data[:, 1 + i].astype(np.float))
    t = len(channels[0]) / fs
    time = np.linspace(0, t, len(channels[0]))

    for i in range(len(channels)):
        notch_channels.append(notch(60, channels[i], fs=fs))

    # for i in range(len(notch_channels)):
    # plt.plot(time, notch_channels[i])

    for i in range(len(channels)):
        bandpass_channels.append(
            bandpass(band[0], band[1], channels[i], fs=fs))

    for i in range(len(notch_channels)):
        bandpass_notch_channels.append(
            bandpass(band[0], band[1], notch_channels[i], fs=fs))

        # for i in range(len(bandpass_notch_channels)):
    #   freq, y = fft(bandpass_notch_channels[i], fs)
    #  plt.plot(freq, y)
    # plt.ylabel("Magnintude of frequency")
    # plt.xlabel("Frequency in Hertz")
    # plt.title(title)
    # plt.ylim(0, 1e7)
    # plt.xlim(10,60)

    return bandpass_notch_channels


def get_dataframes(data):
    bandpass_notch_channels = data
    df1 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    # df2 = pd.DataFrame(columns=["Frequency", "y","sensor])
    df3 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    df4 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    df5 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    df6 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    df7 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    df8 = pd.DataFrame(columns=["Frequency", "y", "sensor"])
    for i in range(len(bandpass_notch_channels)):
        freq, y = fft(bandpass_notch_channels[i], fs)
        if (i == 0):
            df1["Frequency"] = freq
            df1["y"] = y
            df1["sensor"] = 1
        # elif (i == 1):
        #   df2["Frequency"] = freq
        #  df2["y"] = y
        elif (i == 2):
            df3["Frequency"] = freq
            df3["y"] = y
            df3["sensor"] = 3
        elif (i == 3):
            df4["Frequency"] = freq
            df4["y"] = y
            df4["sensor"] = 4
        elif (i == 4):
            df5["Frequency"] = freq
            df5["y"] = y
            df5["sensor"] = 5
        elif (i == 5):
            df6["Frequency"] = freq
            df6["y"] = y
            df6["sensor"] = 6
        elif (i == 6):
            df7["Frequency"] = freq
            df7["y"] = y
            df7["sensor"] = 7
        elif (i == 7):
            df8["Frequency"] = freq
            df8["y"] = y
            df8["sensor"] = 8

    return df1, df3, df4, df5, df6, df7, df8


def make_vector(df1, df3, df4, df5, df6, df7, df8):
    vector = (get_freq_band(df1))
    sensor1 = get_freq_band(df1)
    vector = vector.append(get_freq_band(df3))
    vector = vector.append(get_freq_band(df4))
    vector = vector.append(get_freq_band(df5))
    vector = vector.append(get_freq_band(df6))
    vector = vector.append(get_freq_band(df7))
    vector = vector.append(get_freq_band(df8))

    # print(vector)
    # print("vector length is ",len(vector))

    sensorno = []
    sensor = "sensor"
    band = 1
    counter = 0

    for i in range(35):
        sensorno.append(sensor + str(band))
        counter += 1
        if (counter == 5):
            band += 1
            counter = 0
    vector["Sensor"] = sensorno

    values = []

    for i in range(len(vector)):
        item = vector.iloc[i]["val"]
        values.append(item)

    # bar1=sensor1.plot.bar(x='band', y='val', legend=False)
    # bar2=sensor2.plot.bar(x="band", y="val",legend=False)

    vector.head()

    return vector


def make_vector2(df1, df3, df4, df5, df6, df7, df8):
    vector = df1
    vector = vector.append(df3)
    vector = vector.append(df4)
    vector = vector.append(df5)
    vector = vector.append(df6)
    vector = vector.append(df7)
    vector = vector.append(df8)

    return vector


def get_freq_band(data):
    fft_vals = data["y"]

    # Get frequencies for amplitudes in Hz
    fft_freq = data["Frequency"]

    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    # Plot the data (using pandas here cause it's easy)
    import pandas as pd
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    # ax = df.plot.bar(x='band', y='val', legend=False)
    # ax.set_xlabel("EEG band")
    # ax.set_ylabel("Mean band Amplitude")
    # print(df)

    return df


def do_experiment(data, expno):
    data = fft_filter(data)

    df1, df3, df4, df5, df6, df7, df8 = get_dataframes(data)

    data = make_vector2(df1, df3, df4, df5, df6, df7, df8)

    explist = []
    for i in range(len(data)):
        explist.append(expno)

    data["Label"] = explist

    return data


def build_ML_Model(data):
    print("less go", data)
    brain_wave_dict = {"Delta": 0, "Theta": 1,
                       "Alpha": 2, "Beta": 3, "Gamma": 4}

    label_dict = {"ex01": 0, "ex02": 1, "ex03": 2, "ex04": 3,
                  "ex05": 4, "ex06": 5, "ex07": 6, "ex08": 7, "ex09": 8}
    sensor_dict = {"sensor1": 0, "sensor3": 2, "sensor4": 3, "sensor5": 4, "sensor6": 5, "sensor7": 6,
                   "sensor8": 7}

    data["Label"] = data["Label"].map(label_dict)
    # data["Sensor"] = data["Sensor"].map(sensor_dict)
    y = data["Label"].values

    X = data

    X = X.drop("Label", axis=1)
    X_encoded = pd.get_dummies(X, columns=["sensor"])

    yarr = []
    for i in range(len(data["Label"])):
        yarr.append(data.iloc[i]["Label"])

    actual_y = pd.DataFrame(columns=["Label"])
    actual_y["Label"] = yarr

    print("here")
    print(X)
    print("here")
    print(actual_y)
    X_train, X_test, y_train, y_test, = train_test_split(
        X_encoded, yarr, test_size=0.1)
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)

    boobs = lin_clf.score(X_test, y_test)
    print("the socre is", boobs)
    plot_confusion_matrix(lin_clf, X_test, y_test)
    plt.show()


def build_ML_Model_RF(data):
    label_dict = {"ex01": 0, "ex02": 1, "ex03": 2, "ex04": 3,
                  "ex05": 4, "ex06": 5, "ex07": 6, "ex08": 7, "ex09": 8}
    data["Label"] = data["Label"].map(label_dict)
    X = data[["Frequency", "y", "sensor"]]
    y = data["Label"]
    model = RandomForestClassifier(n_estimators=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model.fit(X_train, y_train)
    from sklearn import metrics
    from sklearn.metrics import plot_confusion_matrix
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))


def do_all():
    plt.rcParams["figure.figsize"] = (15, 15)
    data_list = get_data()

    cleaned_data_list = []

    main_data = pd.DataFrame
    temp_data = pd.DataFrame

    for i in range(len(data_list)):
        temp_data = do_experiment(
            clean2(data_list[i]), search_string(data_list[i]))

        cleaned_data_list.append(temp_data)

    print(len(cleaned_data_list))

    main_data = pd.concat(cleaned_data_list)

    build_ML_Model_RF(main_data)

    return main_data


def pre_process(data):
    brain_wave_dict = {"Delta": 0, "Theta": 1,
                       "Alpha": 2, "Beta": 3, "Gamma": 4}

    label_dict = {"ex01": 0, "ex02": 1, "ex03": 2, "ex04": 3,
                  "ex05": 4, "ex06": 5, "ex07": 6, "ex08": 7, "ex09": 8}

    data["band"] = data["band"].map(brain_wave_dict)
    data["Label"] = data["Label"].map(label_dict)
    y = data["Label"].values

    X = data
    X = X.drop("Sensor", axis=1)
    X = X.drop("Label", axis=1)
    X_encoded = pd.get_dummies(X, columns=["band"])

    return X_encoded, y


def do_all2():
    data_list = get_data()
    classify = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    cleaned_data_list = []
    clf = sklearn.linear_model.SGDClassifier()
    main_data = pd.DataFrame
    temp_data = pd.DataFrame
    other_data = pd.DataFrame
    # print(len(data_list))
    # print(clf)
    temp_data = do_experiment(
        clean2(data_list[0]), search_string(data_list[0]))
    other_data = do_experiment(
        clean2(data_list[1]), search_string(data_list[1]))
    # print("other data is ",other_data)
    x_new, y_new = pre_process(other_data)

    brain_wave_dict = {"Delta": 0, "Theta": 1,
                       "Alpha": 2, "Beta": 3, "Gamma": 4}

    label_dict = {"ex01": 0, "ex02": 1, "ex03": 2, "ex04": 3,
                  "ex05": 4, "ex06": 5, "ex07": 6, "ex08": 7, "ex09": 8}

    temp_data["band"] = temp_data["band"].map(brain_wave_dict)
    temp_data["Label"] = temp_data["Label"].map(label_dict)
    y = temp_data["Label"].values

    X = temp_data
    X = X.drop("Sensor", axis=1)
    X = X.drop("Label", axis=1)
    X_encoded = pd.get_dummies(X, columns=["band"])
    print(X_encoded)
    print(y)

    clf.partial_fit(X_encoded, y, classes=classify)
    print("model is ", clf.score(x_new, y_new))
    clf.partial_fit(x_new, y_new)
    print("model is ", clf.score(X_encoded, y))

    print(main_data.head())


def do_all3():
    data_list = get_data()
    classify = np.array
    classify = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    clf = sklearn.linear_model.SGDClassifier()
    clf.classes_ = classify.shape
    main_data = pd.DataFrame
    temp_data = pd.DataFrame

    temp_data = do_experiment(
        clean2(data_list[0]), search_string(data_list[0]))
    x_1, y_1 = pre_process(temp_data)
    clf.partial_fit(x_1, y_1)

    for i in range(1, (len(data_list) - 2)):
        main_data = do_experiment(
            clean2(data_list[i]), search_string(data_list[i]))
        X, y = pre_process(main_data)
        clf.partial_fit(X, y)

    temp_data = do_experiment(
        clean2(data_list[-1]), search_string(data_list[-1]))
    x_1, y_1 = pre_process(temp_data)
    print("the score is ", clf.score(x_1, y_1))
