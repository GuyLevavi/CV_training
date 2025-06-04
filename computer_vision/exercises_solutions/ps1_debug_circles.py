import os
import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps_1'


def load_gray_smooth(path, sigma=1):
    return gaussian_filter(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), sigma=sigma)


checkerboard_img = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input0.png'), sigma=0)
noisy_checkerboard_img = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input0-noise.png'))
coins_and_pens_image_ideal = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input1.png'))
coins_and_pens_image_real1 = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input2.png'))[50:, 50:]
coins_and_pens_image_real2 = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input3.png'))
coins_and_pens_image_real2_warped = load_gray_smooth(os.path.join(DATA_PATH, 'ps1-input3-warped.png'), sigma=0.5)
# resize to the shape of coins_and_pens_image_real2
coins_and_pens_image_real2_warped = cv2.resize(coins_and_pens_image_real2_warped,
                                               coins_and_pens_image_real2.shape[::-1])
# plt.imshow(coins_and_pens_image_real1)
import numpy as np
from scipy.ndimage import sobel


def hough_circles_acc(edges, radius_range=(10, 50), angle_sweep_res=10, angle_sweep_lim=7.5, radius_res=50, bins=200):
    edges_coords = np.argwhere(edges > 0)
    grad_y = sobel(edges.astype(float), axis=0)
    grad_x = sobel(edges.astype(float), axis=1)
    grad_angles = np.arctan2(grad_y[edges_coords[:, 0], edges_coords[:, 1]],
                             grad_x[edges_coords[:, 0], edges_coords[:, 1]])  # shape: (n_edges,)
    # enrich angles with nearby angles
    enrich = np.linspace(np.radians(-angle_sweep_lim), np.radians(angle_sweep_lim), angle_sweep_res)
    grad_angles = np.repeat(grad_angles, angle_sweep_res) + np.tile(enrich, edges_coords.shape[
        0])  # shape: (n_edges * n_enrich,)
    # Radii setup
    r_values = np.linspace(radius_range[0], radius_range[1], radius_res)  # (n_r,)
    r_values_tiled = np.tile(r_values, edges_coords.shape[0] * angle_sweep_res)  # shape: (n_edges * n_enrich * n_r,)

    # Repeat data
    grad_angles_repeated = np.repeat(grad_angles, radius_res)  # shape: (n_edges * n_enrich * n_r,)
    edge_y_repeated = np.repeat(edges_coords[:, 0], radius_res * angle_sweep_res)  # shape: (n_edges * n_enrich * n_r,)
    edge_x_repeated = np.repeat(edges_coords[:, 1], radius_res * angle_sweep_res)  # shape: (n_edges * n_enrich * n_r,)

    # Compute center estimates
    center_x = edge_x_repeated - r_values_tiled * np.cos(grad_angles_repeated)
    center_y = edge_y_repeated - r_values_tiled * np.sin(grad_angles_repeated)

    coords = np.stack([center_x, center_y, r_values_tiled], axis=1)  # shape: (n_edges * n_enrich * n_r, 3)
    # normalize each coordinate seperately
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    normed_coords = (coords - min_coords) / (max_coords - min_coords)
    # Accumulate into 3D histogram
    _, edges_out = np.histogramdd(coords, bins=bins)
    H, _ = np.histogramdd(normed_coords, bins=bins)
    H = H / edges_out[-1][:1][:,np.newaxis][:,np.newaxis]
    return H, edges_out


def hough_peaks_circles(H, edges_out, max_peaks=20, min_distance=10):
    peaks_indices = peak_local_max(H, num_peaks=max_peaks, exclude_border=False, min_distance=min_distance)
    aedges, bedges, redges = edges_out
    avalues = 0.5 * (aedges[:-1] + aedges[1:])
    bvalues = 0.5 * (bedges[:-1] + bedges[1:])
    rvalues = 0.5 * (redges[:-1] + redges[1:])
    peaks_values = np.array(
        [avalues[peaks_indices[:, 0]], bvalues[peaks_indices[:, 1]], rvalues[peaks_indices[:, 2]]]).T
    return peaks_values, peaks_indices


def draw_circles_on_image(image, a_b_r_array, thickness=2, color=(255, 0, 0)):
    img_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    for (a, b, r) in a_b_r_array:
        cv2.circle(img_copy, (int(a), int(b)), int(r), color=color, thickness=thickness)
    return img_copy


def draw_lines_on_image(image, d_theta_array, thickness=2):
    h, w = image.shape[:2]
    img_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128),
              (255, 128, 0)]
    for (d, theta), color in zip(d_theta_array, colors):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        y1 = (d - cos_t * 0) / sin_t
        y2 = (d - cos_t * (w - 1)) / sin_t
        p1 = (0, int(round(y1)))
        p2 = (w - 1, int(round(y2)))
        try:
            cv2.line(img_copy, p1, p2, color, thickness)
        except:
            print('line go oops')
    return img_copy


def annotate_image(image, xedges, yedges, points, marker='o', color='red', size=40):
    x = points[:, 0]
    y = points[:, 1]

    plt.imshow(image,
               cmap='gray' if image.ndim == 2 else None)

    plt.scatter(x, y, marker=marker, c=color, s=size)

    plt.xlabel("d")
    plt.ylabel("theta")
    plt.xticks([])
    plt.yticks([])
    plt.title("Hough Space")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def find_edges(img, t1=100, t2=100, show=False):
    edges = cv2.Canny(img, t1, t2)
    if show:
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")
        plt.figure()
    return edges


def find_lines_given_edges(img, edges, max_peaks=8, min_distance=50, bins=1000):
    H, dedges, tedges = hough_lines_acc(edges, bins=bins)
    peaks_values, peaks_indices = hough_peaks(H, dedges, tedges, max_peaks=max_peaks, min_distance=min_distance)
    annotate_image(H, dedges, tedges, peaks_indices[:, ::-1])
    img_with_lines = draw_lines_on_image(img, peaks_values)
    plt.imshow(img_with_lines, cmap='gray')
    plt.title('Image with Lines')


def find_lines(img, t1=100, t2=200, max_peaks=8, min_distance=50, bins=1000):
    edges = find_edges(img, t1=t1, t2=t2, show=True)
    find_lines_given_edges(img, edges, max_peaks, min_distance, bins)


def nms_circles(circles, overlap_thresh=0.5):
    if len(circles) == 0:
        return np.array([])

    indices = np.argsort(-circles[:, 3])
    circles = circles[indices]

    keep = []
    while len(circles) > 0:
        current = circles[0]
        keep.append(current)

        if len(circles) == 1:
            break

        # Compute Euclidean distance between current and remaining centers
        dx = circles[1:, 0] - current[0]
        dy = circles[1:, 1] - current[1]
        center_dists = np.sqrt(dx ** 2 + dy ** 2)

        # Compute combined radius threshold (e.g., average or max of radii)
        r_avg = (circles[1:, 2] + current[2]) / 2.0

        # Suppress if center distance < overlap threshold * average radius
        mask = center_dists > overlap_thresh * r_avg
        circles = circles[1:][mask]

    return np.array(keep)


def find_circles(img, t1=100, t2=200, radius_range=(10, 40), radius_res=100, bins=300, show_edges=True, max_peaks=20,
                 min_distance=10, angle_sweep_res=10, angle_sweep_lim=7.5):
    edges = find_edges(img, t1=t1, t2=t2, show=show_edges)
    H, edges_out = hough_circles_acc(edges, radius_range=radius_range, radius_res=radius_res, bins=bins,
                                     angle_sweep_res=angle_sweep_res, angle_sweep_lim=angle_sweep_lim)
    peaks_values, peaks_indices = hough_peaks_circles(H, edges_out, max_peaks=max_peaks, min_distance=min_distance)
    keep = nms_circles(np.concatenate((peaks_values,H[tuple(peaks_indices.T)][:, np.newaxis]),axis=1),
                       overlap_thresh=0.9)
    peaks_values = keep[:, :-1]
    img_with_lines = draw_circles_on_image(img, peaks_values)
    plt.imshow(img_with_lines, cmap='gray')
    plt.title('Image with Circles')
    plt.show()

find_circles(coins_and_pens_image_real1, radius_range=(20, 40), t1=25, t2=150, show_edges=True, max_peaks=20,
             min_distance=10, radius_res=100, angle_sweep_res=5, angle_sweep_lim=5, bins=[200, 200, 50])
