import imageio as iio
import numpy as np
import cv2
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
from csv import DictWriter

def write_to_csv(rows: list[dict]):
    with open('./thickness.csv', 'w') as output:
        writer = DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]

img_stk = iio.mimread('imgs.tiff')
pred_stk = iio.mimread('predictions.tiff')

n = len(img_stk)

thickness_list = []

for idx in tqdm(range(n)):
    widths = [0 for i in range(6)]
    count = 0

    img, pred = cv2.cvtColor(img_stk[idx], cv2.COLOR_BGR2GRAY), pred_stk[idx]
    for i in range(512):
        real_col = np.reshape(img[:, i], (512))
        mid_col = np.reshape(pred[:, i], (512))

        peaks, _ = find_peaks(mid_col, prominence=0.3)
        if(len(peaks) != 6):
            continue

        # averaging to get distances
        avg_col = []
        window = 10

        for i in range(len(real_col)-window):
            avg_col.append(int(sum(real_col[i:i+window])/window))

        avg_peaks, _ = find_peaks(avg_col, prominence=1)
        if(len(avg_peaks) < 6):
            continue
        
        count += 1
        new_peaks = []

        for peak in peaks:
            new_peaks.append(find_closest(avg_peaks, peak))
        
        results_half = peak_widths(avg_col, new_peaks, rel_height=0.5)
        widths = [widths[j]+results_half[0][j] for j in range(6)]

    widths = np.array(widths)/count

    values = {
        'L1': widths[5],
        'L3': widths[4],
        'L5': widths[3],
        'L7': widths[2],
        'L9': widths[1]
    }

    sum_diff = [0 for i in range(5)]
    count = 0

    names = ['L2', 'L4', 'L6', 'L8', 'L10']

    for i in range(512):
        peaks, _ = find_peaks(np.reshape(pred[:, i], (512)), prominence=0.5)
        if(len(peaks) == 6):
            count += 1
            for j in range(5):
                sum_diff[j] += peaks[j+1]-peaks[j]

    for i in range(5):
        values[names[i]] = sum_diff[i]/count

    thickness_list.append(values)

write_to_csv(thickness_list)