import random
import time
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

X = np.array([
    [random.randint(-200,200)],
    [random.randint(-200,200)],
    [random.randint(-200,200)],
    [random.randint(-90,90)],
    [random.randint(-90,90)],
    [random.randint(-90,90)],
    [random.randint(-200,200)],
    [random.randint(-200,200)],
    [random.randint(-200,200)],
    [random.randint(-50,50)],
    [random.randint(-50,50)],
    [random.randint(-50,50)]
])

# Top 3 are translational forces, bottom 3 are rotational forces.
U = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0]
])

# A is basically the derivative of values of X, so displacement becomes velocity and velocity becomes 0 (to be added on by matmul of B and U)
A = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

])

B = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],

    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
])

# We aren't bothered with tuning Q and R right now for balancing speed and efficiency, so we define Q and R as identity matrices.
Q = np.eye(A.shape[0])
R = np.eye(B.shape[1]) * 10 # I have tuned R here so that it doesn't update the state variables too quickly

# This solves the equation that we derived in the paper to find P.
P = solve_continuous_are(A, B, Q, R)

# This uses the calculated P to find K, which is used to update U. Details are in the paper.
R_inv = np.linalg.inv(R)
K = np.matmul(np.matmul(R_inv, B.T), P)
U = np.matmul(-K, X)

run = True
# Initialise the X values for plotting later.
avg_ts = [np.mean(np.array([X[0], X[1], X[2]]))]
avg_rs = [np.mean(np.array([X[3], X[4], X[5]]))]
avg_tv = [np.mean(np.array([X[6], X[7], X[8]]))]
avg_rv = [np.mean(np.array([X[9], X[10], X[11]]))]

loop_counter = 0
y = [0]
while run:
    loop_counter += 1

    time.sleep(0.001)
    dt = 0.001

    U = np.matmul(-K, X)
    X = X + dt * (np.matmul(A,X) + np.matmul(B,U))
    avg_ts.append(np.mean(np.array([X[0], X[1], X[2]])))
    avg_rs.append(np.mean(np.array([X[3], X[4], X[5]])))
    avg_tv.append(np.mean(np.array([X[6], X[7], X[8]])))
    avg_rv.append(np.mean(np.array([X[9], X[10], X[11]])))
    y.append(loop_counter)
    print(f"Whole plant average error: {np.mean(X)}")
    if abs(np.mean(X)) <= 0.000001:
        run = False


# Create 4 graphs to show results:
# First, create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Top-left
axs[0, 0].plot(y, avg_ts)
axs[0, 0].set_title("Avergate translational displacement")
axs[0, 0].set_xlabel("Iterations")

# Top-right
axs[0, 1].plot(y, avg_rs)
axs[0, 1].set_title("Average rotational displacement")
axs[0, 1].set_xlabel("Iterations")

# Bottom-left
axs[1, 0].plot(y, avg_tv)
axs[1, 0].set_title("Average translational velocity")
axs[1, 0].set_xlabel("Iterations")

# Bottom-right
axs[1, 1].plot(y, avg_rv)
axs[1, 1].set_title("Average rotational velocity")
axs[1, 1].set_xlabel("Iterations")

plt.tight_layout()
plt.show()
