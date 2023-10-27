from thesis import loading as loader

l = loader.Loading("../DTAI-Thesis-main\data")
l.load_all()
subjects = l.time_series

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][1][1]

from thesis.oit import simulate_rotation as rotate
rotated_time_series = rotate(time_series, [64,23,22])


from thesis.oit import simulate_sudden_changes as rotate_sequences
rotated_time_series = rotate_sequences(time_series, [[600,1600],[1600,2400],[2400,5000]],[[64,23,22],[15,123,18],[105,80,150]])

"""
from thesis.oit import norm
baseline = norm(time_series)


from thesis.oit import svd
literature = svd(time_series)

from thesis.oit import usvd_skew
unique_skew = usvd_skew(time_series)

from thesis.oit import usvd_mean
unique_mean = usvd_mean(time_series)


from thesis.oit import usvd_abs
unique_absolute = usvd_abs(time_series)


from thesis.oit import wsvd
windowed = wsvd(time_series,5)

from thesis.oit import wusvd_mean
windowed_unique_mean = wusvd_mean(time_series,5)

print(windowed_unique_mean.shape)

from pathlib import Path
from thesis.detection import mtmmdtw
Path('mtmmdtw_data').mkdir(parents=True, exist_ok=True)
data = mtmmdtw('mtmmdtw_data/mtmmdtw_data_regular.npy', time_series, templates) 
print(data)
"""

from pathlib import Path
from thesis.detection import novel_mtmmdtw_mean
Path('mtmmdtw_data').mkdir(parents=True, exist_ok=True)
data = novel_mtmmdtw_mean('mtmmdtw_data/mtmmdtw_data_novel.npy', [[templates, time_series]]) 
print(data.shape)
