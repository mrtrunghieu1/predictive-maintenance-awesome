# define filepath to read data
dir_path = './datasets/CMAPSS_JetEngine'

# define column names for easy indexing
index_names = ['unit', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['sensor_{}'.format(i) for i in range(1, 22)]

# Parameters
epochs = 5
window_length = 50