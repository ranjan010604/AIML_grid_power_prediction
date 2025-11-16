# Data constants
TARGET_COLUMN = 'Global_active_power'
DATE_COLUMN = 'DateTime'

# Model constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Feature constants
LAG_FEATURES = [24, 48]
ROLLING_WINDOWS = [24, 72]

# Evaluation metrics
METRICS = ['MAE', 'RMSE', 'R2']