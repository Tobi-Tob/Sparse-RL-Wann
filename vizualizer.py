from vis.viewInd import viewInd

if __name__ == "__main__":
    # fig, ax = viewInd("champions/swing.out", "swingup")
    fig, ax = viewInd("log/smc_conti_1024_best.out", "sparse_mountain_car_conti")
    fig.show()
