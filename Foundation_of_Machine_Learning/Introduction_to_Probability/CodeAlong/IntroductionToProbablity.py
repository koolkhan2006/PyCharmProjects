import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lst = [1,2,3,4,5,1,3,2,1]
numpy_array = np.array(lst)
panda_series = pd.Series(lst)
print(panda_series.sort_values().mean())
print(panda_series.sort_values().median())
