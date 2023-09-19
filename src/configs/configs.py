# define filepath to read data
dir_path = './datasets/CMAPSS_JetEngine'

# define column names for easy indexing
index_names = ['unit', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['sensor_{}'.format(i) for i in range(1, 22)]

# Parameters of window extraction
WINDOW_LENGTH = 30

# Parameters of LSTM model
EPOCHS = 150
N_HIDDEN = 96
N_LAYER = 4
BATCH_SIZE = 256
LEARNING_RATE = 0.01