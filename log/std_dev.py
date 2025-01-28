# This script should calculate the mean and standard deviation of results.

import numpy as np

smc_data = [[28.7, 29.1, 28.67, 28.56, 29.13, 28.23],
            [28.47, 28.77, 28.18, 28.31, 28.43, 28.53],
            [28.74, 29.01, 28.81, 28.43, 29.04, 28.81],
            [28.48, 28.77, 28.37, 29.27, 27.91, 28.54],
            [28.83, 28.26, 28.71, 28.96, 28.73, 29.1]]

smc_conti_data = [[33.33, 33.3, 30.02, 33.65, 34.22, 31.97],
                  [33.12, 33.17, 30.07, 33.39, 33.95, 31.63],
                  [33.24, 33.56, 30.05, 33.53, 34.25, 31.88],
                  [32.95, 33.69, 30.08, 33.3, 34.4, 31.45],
                  [33.35, 33.29, 30.07, 33.44, 34.22, 31.88]]

lula_data = [[31.49, 32.4, 31.36, 0., 0., 0.],
[33.36, 29.12, 27.56, 0., 0., 0.],
[30.6, 32.02, 29.24, 0., 0., 0.],
[33.16, 31.59, 30.55, 0., 0., 0.],
[31.95, 31.82, 30.75, 0., 0., 0.]]

if __name__ == "__main__":
    full_data = lula_data
    data_per_wVal = np.array([sublist[5] for sublist in full_data])
    mean = np.mean(data_per_wVal)
    std_dev = np.std(data_per_wVal)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
