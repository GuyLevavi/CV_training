import os
import cv2
import numpy as np
from scipy.stats import wasserstein_distance

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps_6\input'
SAVE_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps_6\output'


def predict(particles, std, w_std, h_std, changing_window_size=False):
    # Add Gaussian noise to each particle
    noise = np.random.normal(0, std, size=(particles.shape[0], 2)).astype(np.float32)
    if changing_window_size:
        noise = np.hstack(
            (noise,
             np.random.uniform((-w_std, -h_std), (w_std, h_std), size=(particles.shape[0], 2)).astype(np.float32)))
    particles += noise
    return particles


def get_bgr_histogram(bgr_image, num_bins=64):
    # Split the image into its B, G, R channels
    b_channel = bgr_image[:, :, 0]
    g_channel = bgr_image[:, :, 1]
    r_channel = bgr_image[:, :, 2]

    # Calculate histograms for each channel
    # We use 256 bins for 8-bit images (0-255 pixel values)
    # The range (0, 256) ensures all 256 values are covered.
    # .flatten() is used to convert the 2D channel array to a 1D array of pixel values.
    hist_b, _ = np.histogram(b_channel.flatten(), bins=num_bins, range=(0, 256))
    hist_g, _ = np.histogram(g_channel.flatten(), bins=num_bins, range=(0, 256))
    hist_r, _ = np.histogram(r_channel.flatten(), bins=num_bins, range=(0, 256))

    # Normalize the histograms so they sum to 1. This is crucial for
    # probability distributions, which `wasserstein_distance` expects.
    hist_b = hist_b.astype(np.float32) / (hist_b.sum() + 1e-6)  # Add a small epsilon to prevent division by zero
    hist_g = hist_g.astype(np.float32) / (hist_g.sum() + 1e-6)
    hist_r = hist_r.astype(np.float32) / (hist_r.sum() + 1e-6)

    # (n_bins,3)
    hist = np.stack((hist_b, hist_g, hist_r), axis=-1)

    return hist


def chi_square_distance(hist1, hist2, eps=1e-10):
    if len(hist1.shape) == 2 and len(hist2.shape) == 2:
        return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    elif len(hist1.shape) == 3 and len(hist2.shape) == 3:
        # Assuming hist1 and hist2 are 3D histograms (e.g., BGR)
        dist = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps), axis=-1)
        return dist.mean()
    else:
        raise ValueError('Invalid histogram shape')


def compute_weights(particles, frame, template, sigma_mse, changing_window_size=False, dist_type='mse'):
    num_particles = particles.shape[0]
    weights = np.zeros(num_particles, dtype=np.float32)
    dists = np.zeros(num_particles, dtype=np.float32)
    for i, particle in enumerate(particles):
        if changing_window_size:
            x, y, w, h = particle
            half_w, half_h = int(w // 2), int(h // 2)
        else:
            x, y = particle
            h, w = (template.shape[0], template.shape[1])
            half_w, half_h = w // 2, h // 2
        x, y = int(x), int(y)
        x1, y1 = x - half_w, y - half_h
        x2, y2 = x + half_w, y + half_h
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            weights[i] = 1e-16
            continue
        patch = frame[y1:y2, x1:x2].astype(np.float32)
        if changing_window_size:
            try:
                patch = cv2.resize(patch, (template.shape[1], template.shape[0]))
                # template = cv2.resize(template, (patch.shape[1], patch.shape[0]))
            except:
                patch = None
        if patch is None:
            print('El problemo es con el patch')
            weights[i] = 1e-16
            dists[i] = np.inf
            continue
        elif dist_type == 'mse':
            mse = np.mean((patch - template) ** 2)
            dists[i] = mse
        elif dist_type == 'hist':
            # d(p,q) = (sum over k bins) (p(k) - q(k))^2 / p(k) + q(k)
            hist_patch = get_bgr_histogram(patch)
            hist_template = get_bgr_histogram(template)
            dists[i] = chi_square_distance(hist_patch, hist_template)
        elif dist_type == 'wasserstein':
            # hist_patch = np.histogram(patch.ravel(), bins=50, range=(0, 256), density=True)[0]
            # hist_template = np.histogram(template.ravel(), bins=50, range=(0, 256), density=True)[0]
            hist_patch = get_bgr_histogram(patch)
            hist_template = get_bgr_histogram(template)
            dists[i] = wasserstein_distance(hist_patch, hist_template)
            # dists[i] = wasserstein_distance(hist_template, hist_patch)
        else:
            raise ValueError(f"Unknown distance type: {dist_type}")
        weights[i] = np.exp(-dists[i] / (2 * sigma_mse ** 2))

    # Normalize
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        weights.fill(1.0 / num_particles)
    return weights, dists


def resample(particles, weights):
    num_particles = particles.shape[0]
    # Cumulative distribution
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # ensure sum
    # Stratified resampling
    positions = (np.arange(num_particles) + np.random.rand(num_particles)) / num_particles
    indexes = np.searchsorted(cdf, positions)
    particles[:] = particles[indexes]
    return particles


def estimate(particles, weights):
    # Weighted mean
    mean = np.average(particles, weights=weights, axis=0)
    # Spread: weighted sum of distances
    dists = np.linalg.norm(particles[:, :2] - mean[:2], axis=1)
    spread = np.dot(weights, dists)
    return mean, spread


def calculate_best_template(particles, mses, gray, half_w, half_h):
    # the highest weighted particle
    best_index = np.argmin(mses)
    best_particle = particles[best_index]
    x, y = int(best_particle[0]), int(best_particle[1])
    # Extract the template around the best particle
    best_template = gray[y - half_h:y + half_h, x - half_w:x + half_w].astype(np.float32)
    return best_template


def pad_template_to_shape(source_template, target_shape, mode='constant', constant_values=0):
    if source_template.shape == target_shape:
        return source_template

    pad_h = target_shape[0] - source_template.shape[0]
    pad_w = target_shape[1] - source_template.shape[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    target_shape = ((pad_top, pad_bottom), (pad_left, pad_right))
    if len(source_template.shape) == 3:  # Color image
        target_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    padded_template = np.pad(source_template,
                             target_shape,
                             mode=mode,
                             constant_values=constant_values)
    return padded_template


def run_particle_filter(video_path,
                        init_center,
                        window_size=(50, 50),
                        window_size_multiplier=1.0,
                        num_particles=1000,
                        dyn_std=15,
                        sigma_mse=10,
                        display=True,
                        save_path=None,
                        changing_template=False,
                        alpha=0.5,
                        changing_window_size=False,
                        fix_aspect_ratio=True,
                        w_std=3,
                        h_std=5,
                        dist_type='mse',
                        color=False,
                        resample_freq=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Failed to read first frame")
    if not color:
        img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    else:
        img = first_frame.copy()

    # Extract template around initial center
    w, h = window_size
    w *= window_size_multiplier
    h *= window_size_multiplier
    half_w, half_h = int(w // 2), int(h // 2)
    cx, cy = init_center
    cx, cy = int(cx), int(cy)
    template = img[cy - half_h:cy + half_h, cx - half_w:cx + half_w].astype(np.float32)
    # Initialize particles around init_center
    particles = np.random.normal(loc=[cx, cy], scale=dyn_std, size=(num_particles, 2)).astype(np.float32)
    aspect_ratio = w / h
    if changing_window_size:
        particles = np.hstack((particles,
                               np.random.uniform((w - w_std, h - h_std),
                                                 (w + w_std, h + h_std),
                                                 size=(num_particles, 2)).astype(np.float32)))
    weights = np.ones(num_particles, dtype=np.float32) / num_particles

    # Video writer
    writer = None
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

    i = 0
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if not color:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame.copy()

        # Predict
        particles = predict(particles, dyn_std, w_std, h_std, changing_window_size)
        # Keep within frame
        particles[:, 0] = np.clip(particles[:, 0], 0, frame.shape[1] - 1)
        particles[:, 1] = np.clip(particles[:, 1], 0, frame.shape[0] - 1)
        if changing_window_size:
            particles[:, 2] = np.clip(particles[:, 2], 5, frame.shape[1] - 1)
            particles[:, 3] = np.clip(particles[:, 3], 7, frame.shape[0] - 1)

        # Update
        weights, dists = compute_weights(particles, img, template, sigma_mse, changing_window_size, dist_type)
        # all_dists.append(dists)
        # dist_thresh = np.quantile(dists, 0.95)
        # # Remove particles with high distance
        # weights[dists > dist_thresh] = 0
        if changing_template:
            best_template = calculate_best_template(particles, dists, img, half_w, half_h)
            best_template = pad_template_to_shape(best_template, template.shape)
            if changing_window_size:
                template = cv2.resize(template, (best_template.shape[1], best_template.shape[0]))
            template = alpha * best_template + (1 - alpha) * template

        # Estimate
        mean, spread = estimate(particles, weights)
        mean_x, mean_y = int(mean[0]), int(mean[1])
        if changing_window_size:
            mean_w, mean_h = mean[2], mean[3]
            if fix_aspect_ratio:
                # mean_h = mean_w / aspect_ratio
                mean_w = mean_h * aspect_ratio
            half_w, half_h = int(mean_w // 2), int(mean_h // 2)

        # Resample
        if i % resample_freq == 0:
            particles = resample(particles, weights)

        # Visualization
        vis = frame.copy()
        # Draw particles
        for x, y in particles[:, :2]:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Draw tracking window (rectangle)
        x1, y1 = mean_x - half_w, mean_y - half_h
        x2, y2 = mean_x + half_w, mean_y + half_h
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw spread circle
        radius = int(spread)
        cv2.circle(vis, (mean_x, mean_y), radius, (0, 0, 255), 2)

        if display:
            cv2.imshow('Particle Tracker', vis)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_name = 'pedestrians'
    bb_name = video_name  # 'hand'
    video_path = os.path.join(DATA_PATH, video_name + '.avi')
    bb_path = os.path.join(DATA_PATH, bb_name + '.txt')
    (x, y), (w, h) = np.loadtxt(bb_path, delimiter=' ', unpack=True).T
    init_center = (x + w / 2, y + h / 2)
    hyperparams = dict(
        num_particles=500,
        dyn_std=(3, 15),
        sigma_mse=12,
        dist_type='mse',
        color=True,
        resample_freq=4,
        window_size_multiplier=1.0,
        changing_template=False,
        alpha=0.1,
        changing_window_size=True,
        fix_aspect_ratio=True,
        w_std=5,
        h_std=5
    )
    # save name is based on hyperparameters
    hyperparams_str = '_'.join(f'{k}{v}' for k, v in hyperparams.items())
    save_path = os.path.join(SAVE_PATH, f'{video_name}_{bb_name}_{hyperparams_str}.avi')
    run_particle_filter(video_path,
                        init_center=init_center,
                        window_size=(w, h),
                        save_path=save_path,
                        display=True,
                        **hyperparams
                        )
