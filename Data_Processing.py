import scipy.io as sc
import numpy as np
i = 0
j = 1
n = [i]
m = []
while (i != j) :
    # print(i, j)
    if  j > 3: #6????
        j = 0
    elif ((i,j) not in m) & ((j, i) not in m):
        m.append((n[-1], j))
        n.append(j)
        i=j
        j = i + 1
    else:
        j += 1
print(12)
## the algorithm 1 in paper Human activity recognition
# using wearable sensors by deep convolutional neural networks

def PAMAP_Image(file_data):
    i = 0
    j = 1
    n = [i]
    m = []
    while (i != j) :
        # print(i, j)
        if  j > 12:
            j = 0
        elif ((i,j) not in m) & ((j, i) not in m):
            m.append((n[-1], j))
            n.append(j)
            i=j
            j = i + 1
        else:
            j += 1
    # print(n,len(n))
    new_data = []
    for line in file_data:
        line_split = []
        line_split.append([line[0], line[17], line[34]])
        line_split.append(line[1:4])
        line_split.append(line[4:7])
        line_split.append(line[7:10])
        line_split.append(line[10:13])
        line_split.append(line[18:21])
        line_split.append(line[21:24])
        line_split.append(line[24:27])
        line_split.append(line[27:30])
        line_split.append(line[35:38])
        line_split.append(line[38:41])
        line_split.append(line[41:44])
        line_split.append(line[44:47])
        line_split.append(line[51:54])
        new_line = []
        for i in range(len(n) - 1):
            sensor = line_split[n[i]]  # x, y, z
            x, y, z = sensor[0], sensor[1], sensor[2]
            xyz = []
            xyz.append(x), xyz.append(y), xyz.append(z)
            yxz = []
            yxz.append(y), yxz.append(x), yxz.append(z)
            zxy = []
            zxy.append(z), zxy.append(x), zxy.append(y)
            sensor_long = []
            if i % 2 == 0:
                sensor_long.extend(xyz), sensor_long.extend(xyz), sensor_long.extend(xyz)
            else:
                sensor_long.extend(xyz), sensor_long.extend(yxz), sensor_long.extend(zxy)

            new_line.extend(sensor_long)
        new_line.extend(line_split[-1])
        new_data.append(new_line)
    return np.array(new_data)

def MHEALTH_Image(file_data):
    i = 0
    j = 1
    n = [i]
    m = []
    while (i != j) :
        # print(i, j)
        if  j > 7: #6????
            j = 0
        elif ((i,j) not in m) & ((j, i) not in m):
            m.append((n[-1], j))
            n.append(j)
            i=j
            j = i + 1
        else:
            j += 1
    # print(n,len(n))
    new_data = []
    for line in file_data:
        line_split = []
        line_split.append([line[3], line[4], line[3]])
        line_split.append(line[0:3])
        line_split.append(line[5:8])
        line_split.append(line[8:11])
        line_split.append(line[11:14])
        line_split.append(line[14:17])
        line_split.append(line[17:20])
        line_split.append(line[20:23])

        line_split.append(line[-2:])
        new_line = []
        for i in range(len(n)):
            sensor = line_split[n[i]]  # x, y, z
            x, y, z = sensor[0], sensor[1], sensor[2]
            xyz = []
            xyz.append(x), xyz.append(y), xyz.append(z)
            yxz = []
            yxz.append(y), yxz.append(x), yxz.append(z)
            zxy = []
            zxy.append(z), zxy.append(x), zxy.append(y)
            sensor_long = []
            if i % 2 == 0:
                sensor_long.extend(xyz), sensor_long.extend(xyz), sensor_long.extend(xyz)
            else:
                sensor_long.extend(xyz), sensor_long.extend(yxz), sensor_long.extend(zxy)

            new_line.extend(sensor_long)
        new_line.extend(line_split[-1])
        new_data.append(new_line)
    return np.array(new_data)

def REAL_Image(file_data):
    i = 0
    j = 1
    n = [i]
    m = []
    while (i != j) :
        # print(i, j)
        if  j > 5:
            j = 0
        elif ((i,j) not in m) & ((j, i) not in m):
            m.append((n[-1], j))
            n.append(j)
            i=j
            j = i + 1
        else:
            j += 1
    # print(n,len(n))
    new_data = []
    for line in file_data:
        line_split = []
        line_split.append(line[0:3])
        line_split.append(line[3:6])
        line_split.append(line[6:9])
        line_split.append(line[9:12])
        line_split.append(line[12:15])
        line_split.append(line[15:18])

        line_split.append(line[18:])
        new_line = []
        for i in range(len(n) - 1):
            sensor = line_split[n[i]]  # x, y, z
            x, y, z = sensor[0], sensor[1], sensor[2]
            xyz = []
            xyz.append(x), xyz.append(y), xyz.append(z)
            yxz = []
            yxz.append(y), yxz.append(x), yxz.append(z)
            zxy = []
            zxy.append(z), zxy.append(x), zxy.append(y)
            sensor_long = []
            if i % 2 == 0:
                sensor_long.extend(xyz), sensor_long.extend(xyz), sensor_long.extend(xyz)
            else:
                sensor_long.extend(xyz), sensor_long.extend(yxz), sensor_long.extend(zxy)

            new_line.extend(sensor_long)
        new_line.extend(line_split[-1])
        new_data.append(new_line)
    return np.array(new_data)

def REAL_Image_2sensors(file_data):
    i = 0
    j = 1
    n = [i]
    m = []
    while (i != j):
        # print(i, j)
        if j > 3:
            j = 0
        elif ((i, j) not in m) & ((j, i) not in m):
            m.append((n[-1], j))
            n.append(j)
            i = j
            j = i + 1
        else:
            j += 1
    # print(n,len(n))
    new_data = []
    for line in file_data:
        line_split = []
        line_split.append(line[0:3])
        line_split.append(line[3:6])
        line_split.append(line[6:9])
        line_split.append(line[9:12])
        line_split.append(line[12:15])
        line_split.append(line[15:18])

        line_split.append(line[18:])
        new_line = []
        for i in range(len(n) - 1):
            sensor = line_split[n[i]]  # x, y, z
            x, y, z = sensor[0], sensor[1], sensor[2]
            xyz = []
            xyz.append(x), xyz.append(y), xyz.append(z)
            yxz = []
            yxz.append(y), yxz.append(x), yxz.append(z)
            zxy = []
            zxy.append(z), zxy.append(x), zxy.append(y)
            sensor_long = []
            if i % 2 == 0:
                sensor_long.extend(xyz), sensor_long.extend(xyz), sensor_long.extend(xyz)
            else:
                sensor_long.extend(xyz), sensor_long.extend(yxz), sensor_long.extend(zxy)

            new_line.extend(sensor_long)
        new_line.extend(line_split[-1])
        new_data.append(new_line)
    return np.array(new_data)
# def PAMAP_Cube(file_data, time_window):
#     new_data = []
#     for i in range(file_data.shape[0]):
#         if time_window // 2 * i + time_window - 1 >= file_data.shape[0]:
#             break
#         if not all(file_data[time_window // 2 * i][-2:] == \
#                 file_data[time_window // 2 * i + time_window - 1][-2:]):
#             continue
#         cube = []
#         for j in range(time_window // 2 * i, time_window // 2 * i + time_window - 1 + 1):
#             cube.extend(file_data[j][:-3])
#         cube.extend(file_data[j][-3:])
#         new_data.append(cube)
#     return np.array(new_data)

def extract_time_sequences(file_data, n_time_window, nb_sequence):
    new_data = []
    n = 0
    while 1:
        if (n + 1) * n_time_window - 1 >= file_data.shape[0]:
            break

        if any(file_data[n * n_time_window][-2:] != file_data[(n + 1) * n_time_window - 1][-2:]):
            n += 1
            continue

        time_sequence = []
        for i in range(n * n_time_window, (n + 1) * n_time_window):
            time_sequence.append(file_data[i])
        new_data.append(time_sequence)
        n += 1
        # if n%10000==0:
        #     print(n)

    new_data = np.array(new_data)
    np.random.shuffle(new_data)
    new_data = new_data[:nb_sequence]
    new_data = np.reshape(new_data, [-1, 54])  ## 54-d data

    return new_data

def extract_time_sequences_MH(file_data, n_time_window, nb_sequence):
    new_data = []
    n = 0
    while 1:
        if (n + 1) * n_time_window - 1 >= file_data.shape[0]:
            break

        if any(file_data[n * n_time_window][-2:] != file_data[(n + 1) * n_time_window - 1][-2:]):
            n += 1
            continue

        time_sequence = []
        for i in range(n * n_time_window, (n + 1) * n_time_window):
            time_sequence.append(file_data[i])
        new_data.append(time_sequence)
        n += 1
        # if n%10000==0:
        #     print(n)

    new_data = np.array(new_data)
    np.random.shuffle(new_data)
    new_data = new_data[:nb_sequence]
    new_data = np.reshape(new_data, [-1, 25])  ## 25-d data

    return new_data

def extract_time_sequences_REAL(file_data, n_time_window, nb_sequence):
    new_data = []
    n = 0
    while 1:
        if (n + 1) * n_time_window - 1 >= file_data.shape[0]:
            break

        if any(file_data[n * n_time_window][-2:] != file_data[(n + 1) * n_time_window - 1][-2:]):
            n += 1
            continue

        time_sequence = []
        for i in range(n * n_time_window, (n + 1) * n_time_window):
            time_sequence.append(file_data[i])
        new_data.append(time_sequence)
        n += 1
        # if n%10000==0:
        #     print(n)

    new_data = np.array(new_data)
    np.random.shuffle(new_data)
    new_data = new_data[:nb_sequence]
    new_data = np.reshape(new_data, [-1, 21])  ## 25-d data

    return new_data

if __name__ == '__main__':
    file_name = 'MHEALTH_balance'
    file_data = sc.loadmat(file_name + '.mat')
    file_data = file_data[file_name]
    file_data = file_data[:45]
    file_data = MHEALTH_Image(file_data)
    print(file_data)

    # feature_num = 702
    # test_subject = 5
    # n_time_window = 7
    # training_data_size = 100
    # file_name = 'PAMAP2_Protocol_6_classes_balance'
    # file_data = sc.loadmat(file_name + '.mat')
    # file_data = file_data[file_name]
    # training_data_dic = {1: file_data[156028:],
    #                      2: np.append(file_data[: 156028], file_data[312445:], axis=0),
    #                      3: np.append(file_data[: 312445], file_data[424232:], axis=0),
    #                      4: np.append(file_data[: 424232], file_data[567124:], axis=0),
    #                      5: np.append(file_data[: 567124], file_data[738401:], axis=0),
    #                      6: np.append(file_data[: 738401], file_data[894893:], axis=0),
    #                      7: np.append(file_data[: 894893], file_data[1034840:], axis=0),
    #                      8: np.append(file_data[: 1034840], file_data[1197950:], axis=0),
    #                      9: file_data[: 1197950]}
    #
    # test_data_dic = {1: file_data[: 156028],
    #                  2: file_data[156028: 312445],
    #                  3: file_data[312445: 424232],
    #                  4: file_data[424232: 567124],
    #                  5: file_data[567124: 738401],
    #                  6: file_data[738401: 894893],
    #                  7: file_data[894893: 1034840],
    #                  8: file_data[1034840: 1197950],
    #                  9: file_data[1197950:]}
    # training_data = training_data_dic[test_subject]
    # # test_data = test_data_dic[test_subject]
    # training_data = extract_time_sequences(training_data, n_time_window, training_data_size // n_time_window)
    # print(training_data)