import numpy as np

def smooth(data, sm=2):
    smooth_data = []
    if sm > 1:
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return np.array(smooth_data)

"""draw min model error and first model error with train policy: line chart"""
def plot_info(title, y1: np.array, y2: np.array=None):
    # delete Abnormal point
    def delete_no_data(data_array)->np.array:
        mean = np.mean(data_array)
        std = np.std(data_array)
        preprocessed_data_array = [x for x in data_array if (x > mean - std)]
        preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + std)]
        return preprocessed_data_array

    import matplotlib.pyplot as plt

    y1 = y1.ravel()
    y1 = np.log10(y1)
    y1mean = np.mean(delete_no_data(y1))
    steps = y1.size
    length = range(steps)

    filename = "dad error"
    title = title
    plt.title(title, fontweight="bold")
    plt.ylabel("dad error (log)")
    plt.xlabel("steps")
    plt.plot(length, y1, marker="^", linestyle="-", color="r", label="minerror -mean: "+str(y1mean))
    if y2 is not None:
        y2 = y2.ravel()
        y2mean = round(np.mean(y2),2)
        y2 = np.log10(y2)
        plt.plot(length, y2, marker="s", linestyle="-", color="b", label="initerror -mean: "+str(y2mean))

    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98))
    plt.grid()
    plt.savefig(f"./{filename}.jpg")
    plt.show()

"""bar chart"""
def plot_info_1(filename, title, y1: np.array, y2: np.array=None):

    import matplotlib.pyplot as plt
    if not isinstance(y1, np.ndarray):
        y1 = np.asarray(y1)
    y1 = y1.ravel()
    y1 = np.log10(y1)
    steps = y1.size
    length = range(steps)

    filename = filename
    title = title
    plt.title(title, fontweight="bold")
    plt.ylabel("dad error (log)")
    plt.xlabel("steps N")
    plt.bar(length, y1, align='center',width=0.4, alpha=0.5, linestyle="-", label="train dad")
    if y2 is not None:
        if not isinstance(y2, np.ndarray):
            y2 = np.asarray(y2)
        y2 = y2.ravel()
        y2mean = np.mean(y2)
        y2 = np.log10(y2)
        plt.plot(length, y2, marker="s", linestyle="-", color="b", label="initerror -mean: "+str(y2mean))

    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.98))
    plt.grid()
    plt.savefig(f"./{filename}.jpg")
    plt.show()


"""draw reward: line chart"""
def plot_info_3(title, y1: np.array, idx: np.array, y2: np.array=None):
    # delete Abnormal point

    import matplotlib.pyplot as plt

    y1 = np.log10(y1)
    length = idx
    # length = idx
    y1 = y1.ravel()
    filename = title
    title = title
    plt.title(title, fontweight="bold")
    plt.ylabel("rewards")
    plt.xlabel("steps")
    # sns.set(style="darkgrid", font_scale=1.5)
    plt.plot(length, y1, color="r", linestyle="-", label="ppo ",alpha=0.7)

    if y2 is not None:
        y2 = y2.ravel()
        plt.plot(length, y2, linestyle="-", color="b", label="ppo+dad", alpha=0.7)

    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98))
    plt.grid()
    plt.savefig(f"./{filename}.jpg")
    plt.show()

def read_npy_data(cwd, plot=True):
    data = np.load(cwd, allow_pickle=True)
    if plot:
        a = data.item()[list(data.item().keys())[0]]
        b = data.item()[list(data.item().keys())[1]]

        plot_info("train model error with train policy", a, b)

    else:
        print(list(data.item().keys()))

def get_max_re(data, interval):
    def delete_no_data(data_array)->np.array:
        mean = np.mean(data_array)
        std = np.std(data_array)
        preprocessed_data_array = [x for x in data_array if (x > mean - std)]
        preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + std)]
        return preprocessed_data_array
    l = len(data)
    data_b = []
    b = []
    for i in range(l):
        cur = min(i+interval, l)
        a = data[i:cur]
        # b = delete_no_data(a)
        if len(b) == 0:
            data_b.append(np.max(a))
        else:
            data_b.append(np.max(b))
    return np.array(data_b)


def uniform_dataandidx(data1, data2):
    """default index1 > index2"""
    l1 = len(data1)
    l2 = len(data2)
    ratio = 1.0 * l1 / l2
    new_data1 = []
    for i in range(l2):
        idx = round(ratio * i)
        new_data1.append(data1[idx])
    return np.array(new_data1)



def read_npy_data_1(cwd, plot=1, cwd2 = None):
    data1 = np.load(cwd, allow_pickle=True)
    data2 = None
    if cwd:
        data2 = np.load(cwd2, allow_pickle=True)
    if plot==1:
        a = data1.item()[list(data1.item().keys())[0]]

        plot_info_1(cwd, "model error with train dad", a)

    elif plot==2:
        print(list(data1.item().keys()))
    else :
        # idx = 0
        # for (index, item) in enumerate(data):
        #     if item[0] >= 800000:
        #         idx = index
        #         break
        # # idx =  - 1
        reward1 = data1[:, 1]
        interval = 40
        idx = data1[:, 0]
        if cwd2:
            reward2 = data2[:, 1]
            if len(reward1) > len(reward2):
                idx = data2[:, 0]
                reward1 = uniform_dataandidx(reward1, reward2)
                # reward1 = get_max_re(reward1, interval)
                # reward2 = get_max_re(reward2, interval)
                plot_info_3(cwd, reward1, idx, reward2)
            elif len(reward1) < len(reward2):

                reward2 = uniform_dataandidx(reward2, reward1)
                # reward1 = get_max_re(reward1, interval)
                # reward2 = get_max_re(reward2, interval)
                plot_info_3(cwd, reward1, idx, reward2)
            else:
                plot_info_3(cwd, reward1, idx, reward2)

        else:
            plot_info_3(cwd, reward1, idx)


# read_npy_data_1("./dad_train_error.npy", plot=True)
# read_npy_data_1("./recorder_ppo.npy", plot=3, cwd2="./recorder_dad.npy")
# data = np.load("./p_recorder.npy", allow_pickle=True)

# print(len(data))
a = np.load("./recorder_ppo.npy")
b = np.load("./recorder_dad.npy")
la = len(a)
lb = len(b)
lamax = 0
lbmax = 0
for i in range(la):
    if a[i][1] > 4000.0:
        lamax = i+1
        break
for i in range(la):
    if b[i][0] > 5e6:
        lbmax = i+1
        break

stepsa = a[:lamax, 0]
rewa = a[:lamax, 1]
stepsb = b[:, 0]
rewb = b[:, 1]



plot_info_3("ppo", y1=rewa, idx=stepsa)
