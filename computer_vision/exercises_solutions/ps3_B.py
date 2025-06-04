import os
import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask

# set numpy prints to 4 decimal points
np.set_printoptions(precision=4, suppress=True)

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps_3'


def homogenize(p):
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)


def dehomogenize(p):
    return (p / p[:, -1][:, np.newaxis])[:, :-1]


def construct_A_epipolar(p1, p2):
    assert p1.shape[0] == p2.shape[0], "Number of points in p1 and p2 must be the same"
    n_pts = p1.shape[0]
    A = np.zeros((n_pts, 9))
    A[:, 0] = p2[:, 0] * p1[:, 0]
    A[:, 1] = p2[:, 0] * p1[:, 1]
    A[:, 2] = p2[:, 0]
    A[:, 3] = p2[:, 1] * p1[:, 0]
    A[:, 4] = p2[:, 1] * p1[:, 1]
    A[:, 5] = p2[:, 1]
    A[:, 6] = p1[:, 0]
    A[:, 7] = p1[:, 1]
    A[:, 8] = 1
    return A


def normalize_transform(p):
    M = np.array([[1, 0, -np.mean(p[:, 0])],
                  [0, 1, -np.mean(p[:, 1])],
                  [0, 0, 1]])
    S = np.diag(np.concatenate([1 / np.std(p, axis=0), [1]]))
    return S @ M


def compute_fundamental_matrix(p1, p2, normalize=False):
    if normalize:
        T1 = normalize_transform(p1)
        T2 = normalize_transform(p2)
        p1 = dehomogenize((T1 @ homogenize(p1).T).T)
        p2 = dehomogenize((T2 @ homogenize(p2).T).T)
    A = construct_A_epipolar(p1, p2)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    F_hat = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_hat)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    if normalize:
        F = T2.T @ F @ T1
    return F


def boundary_lines(img):  # , normalize_according_to=None):
    p_upper_left = homogenize(np.array([[0, 0]]))
    p_lower_left = homogenize(np.array([[0, img.shape[0]]]))
    p_upper_right = homogenize(np.array([[img.shape[1], 0]]))
    p_lower_right = homogenize(np.array([[img.shape[1], img.shape[0]]]))
    # if normalize_according_to is not None:
    #     T = normalize_transform(normalize_according_to)
    #     p_upper_left = (T @ p_upper_left.T).T
    #     p_lower_left = (T @ p_lower_left.T).T
    #     p_upper_right = (T @ p_upper_right.T).T
    #     p_lower_right = (T @ p_lower_right.T).T
    l_L = np.cross(p_upper_left, p_lower_left)
    l_R = np.cross(p_upper_right, p_lower_right)
    return l_L, l_R


def draw_epipolar_lines(pts_a_2d, pts_b_2d, img_a, img_b, normalize=False):
    F = compute_fundamental_matrix(pts_a_2d, pts_b_2d, normalize=normalize)
    l_b = F @ homogenize(pts_a_2d).T
    l_b = l_b.T
    l_a = F.T @ homogenize(pts_b_2d).T
    l_a = l_a.T

    l_L_a, l_R_a = boundary_lines(img_a)  # , normalize_according_to=pts_a_2d if normalize else None)
    l_L_b, l_R_b = boundary_lines(img_b)  # , normalize_according_to=pts_b_2d if normalize else None)

    p_a_L = np.cross(l_a, l_L_a)
    p_a_R = np.cross(l_a, l_R_a)
    p_b_L = np.cross(l_b, l_L_b)
    p_b_R = np.cross(l_b, l_R_b)

    # if normalize:
    #     T_a_inv = np.linalg.inv(normalize_transform(pts_a_2d))
    #     T_b_inv = np.linalg.inv(normalize_transform(pts_b_2d))
    #     p_a_L = (T_a_inv @ p_a_L.T).T
    #     p_a_R = (T_a_inv @ p_a_R.T).T
    #     p_b_L = (T_b_inv @ p_b_L.T).T
    #     p_b_R = (T_b_inv @ p_b_R.T).T

    p_a_L = (p_a_L / p_a_L[:, -1][:, np.newaxis])[:, :-1]
    p_a_R = (p_a_R / p_a_R[:, -1][:, np.newaxis])[:, :-1]
    p_b_L = (p_b_L / p_b_L[:, -1][:, np.newaxis])[:, :-1]
    p_b_R = (p_b_R / p_b_R[:, -1][:, np.newaxis])[:, :-1]

    # img_a_copy = img_a.copy()
    # img_b_copy = img_b.copy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for p_L, p_R in zip(p_a_L, p_a_R):
        print(f"Epipolar line in image A: {p_L}, {p_R}")
        cv2.line(img_a, tuple(p_L.astype(int)), tuple(p_R.astype(int)), (255, 0, 0), 1)
    axes[0].set_title('Epipolar Lines in Image A')
    axes[0].imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    for p_L, p_R in zip(p_b_L, p_b_R):
        cv2.line(img_b, tuple(p_L.astype(int)), tuple(p_R.astype(int)), (255, 0, 0), 1)
    axes[1].set_title('Epipolar Lines in Image B')
    axes[1].imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    axes[1].axis('off')
    plt.show()


if __name__ == '__main__':
    img_a = cv2.imread(os.path.join(DATA_PATH, 'pic_a.jpg'))
    img_b = cv2.imread(os.path.join(DATA_PATH, 'pic_b.jpg'))
    pts_a_2d = np.loadtxt(os.path.join(DATA_PATH, 'pts2d-pic_a.txt'))
    pts_b_2d = np.loadtxt(os.path.join(DATA_PATH, 'pts2d-pic_b.txt'))
    pts_a_2d_norm = np.loadtxt(os.path.join(DATA_PATH, 'pts2d-norm-pic_a.txt'))
    pts_3d = np.loadtxt(os.path.join(DATA_PATH, 'pts3d.txt'))
    pts_3d_norm = np.loadtxt(os.path.join(DATA_PATH, 'pts3d-norm.txt'))
    draw_epipolar_lines(pts_a_2d, pts_b_2d, img_a, img_b, normalize=False)
