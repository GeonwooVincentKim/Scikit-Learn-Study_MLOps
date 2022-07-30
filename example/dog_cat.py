import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout)
from tensorflow.keras.datasets.mnist import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()
