from wann_src.nsga_sort import nsga_sort
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load objective values
    objVals_path = "log/lula/lula_256_480_objVals.out"
    if isinstance(objVals_path, str):
        objVals = np.loadtxt(objVals_path, delimiter=',')

    # Number of rows and columns
    print("Rows: ", objVals.shape[0])
    print("Cols: ", objVals.shape[1])
    objVals = "log/lula/lula_256_480_objVals.out"
    if isinstance(objVals, str):
        objVals = np.loadtxt(objVals, delimiter=',')
    # Number of rows and cols
    print("Rows: ", objVals.shape[0])
    print("Cols: ", objVals.shape[1])
    rank, fronts = nsga_sort(objVals[:, [249, 251]], returnFronts=True)
    print("Pareto Fronts: ", fronts)
