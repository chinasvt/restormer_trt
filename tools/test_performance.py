import numpy as np

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:",res,diff0,diff1)



if __name__ == "__main__":
    trt_output = np.loadtxt("../data/result.txt").reshape(1, 3, 368,552)
    pt_output = np.load("../data/output.npy")
    check(trt_output,pt_output)


