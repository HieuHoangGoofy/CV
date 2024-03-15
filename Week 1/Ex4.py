import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define points
A = np.array([1, 1, 1])
B = np.array([-1, 1, 1])
C = np.array([1, -1, 1])
D = np.array([-1, -1, 1])
E = np.array([1, 1, -1])
F = np.array([-1, 1, -1])
G = np.array([1, -1, -1])
H = np.array([-1, -1, -1])

camera = np.array([2, 3, 5])

Points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]
points = {k: v - camera for k, v in Points.items()}


def pinhole(v):
    x, y, z = v
    if z == 0:
        return np.array([float('inf'), float('inf')])
    return np.array([x / z, y / z])


def rotate(R, v):
    return np.dot(R, v)


def getRotX(angle):
    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)

    return Rx


def getRotY(angle):
    Ry = np.zeros((3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1

    return Ry


def getRotZ(angle):
    Rz = np.zeros((3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1

    return Rz


angles = [30, 40, 50]

# Plot for Z rotations
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_z in enumerate(angles):
    Rz = getRotZ(np.degrees(angle_z))

    # Apply rotation to points
    ps = {key: rotate(Rz, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    # Plot edges
    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"Z{angle_z}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()

# Plot for Y rotations
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_y in enumerate(angles):
    Ry = getRotY(np.degrees(angle_y))

    # Apply rotation to points
    ps = {key: rotate(Ry, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    # Plot edges
    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"Y{angle_y}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()

# Plot for X rotations
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_x in enumerate(angles):
    Rx = getRotX(np.degrees(angle_x))

    # Apply rotation to points
    ps = {key: rotate(Rx, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    # Plot edges
    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"X{angle_x}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()