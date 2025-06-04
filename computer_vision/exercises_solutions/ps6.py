import os
import cv2
import numpy as np

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps6\input'
SAVE_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps6\output'

def initialize_particles(num_particles, frame_shape):
    # Uniform distribution over image
    h, w = frame_shape
    particles = np.empty((num_particles, 2), dtype=np.float32)
    particles[:, 0] = np.random.uniform(0, w, size=num_particles)  # x coordinates
    particles[:, 1] = np.random.uniform(0, h, size=num_particles)  # y coordinates
    return particles


def predict(particles, std):
    # Add Gaussian noise to each particle
    noise = np.random.normal(0, std, particles.shape)
    particles += noise
    return particles


def compute_weights(particles, frame, template, window_size, sigma_mse):
    num_particles = particles.shape[0]
    weights = np.zeros(num_particles, dtype=np.float32)
    mses = np.zeros(num_particles, dtype=np.float32)
    h, w = template.shape
    half_w, half_h = w // 2, h // 2

    for i, (x, y) in enumerate(particles):
        x, y = int(x), int(y)
        x1, y1 = x - half_w, y - half_h
        x2, y2 = x + half_w, y + half_h
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            weights[i] = 1e-16
            continue
        patch = frame[y1:y2, x1:x2].astype(np.float32)
        mse = np.mean((patch - template) ** 2)
        mses[i] = mse
        weights[i] = np.exp(-mse / (2 * sigma_mse**2))

    # Normalize
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        weights.fill(1.0 / num_particles)
    return weights, mses


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
    dists = np.linalg.norm(particles - mean, axis=1)
    spread = np.dot(weights, dists)
    return mean, spread

def calculate_best_template(particles, mses, gray, half_w, half_h):
    # the highest weighted particle
    best_index = np.argmin(mses)
    best_particle = particles[best_index]
    x, y = int(best_particle[0]), int(best_particle[1])
    # Extract the template around the best particle
    best_template = gray[y-half_h:y+half_h, x-half_w:x+half_w].astype(np.float32)
    return best_template


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
                        changing_window_size=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Failed to read first frame")
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Extract template around initial center
    w, h = window_size
    w *= window_size_multiplier
    h *= window_size_multiplier
    half_w, half_h = int(w // 2), int(h // 2)
    cx, cy = init_center
    cx, cy = int(cx), int(cy)
    template = gray[cy-half_h:cy+half_h, cx-half_w:cx+half_w].astype(np.float32)
    # Initialize particles around init_center
    particles = np.random.normal(loc=[cx, cy], scale=[dyn_std, dyn_std], size=(num_particles, 2)).astype(np.float32)
    weights = np.ones(num_particles, dtype=np.float32) / num_particles

    # Video writer
    writer = None
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Predict
        particles = predict(particles, dyn_std)
        # Keep within frame
        particles[:, 0] = np.clip(particles[:, 0], 0, frame.shape[1]-1)
        particles[:, 1] = np.clip(particles[:, 1], 0, frame.shape[0]-1)

        # Update
        weights, mses = compute_weights(particles, gray, template, window_size, sigma_mse)
        if changing_template:
            best_template = calculate_best_template(particles, mses, gray, half_w, half_h)
            template = alpha * best_template + (1 - alpha) * template

        # Estimate
        mean, spread = estimate(particles, weights)
        mean_x, mean_y = int(mean[0]), int(mean[1])

        # Resample
        particles = resample(particles, weights)

        # Visualization
        vis = frame.copy()
        # Draw particles
        for x, y in particles:
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
    video_name = 'pres_debate'
    bb_name = 'hand'
    video_path = os.path.join(DATA_PATH, video_name + '.avi')
    bb_path = os.path.join(DATA_PATH, bb_name + '.txt')
    (x,y), (w,h) = np.loadtxt(bb_path, delimiter=' ', unpack=True).T
    init_center = (x + w / 2, y + h / 2)
    hyperparams = dict(
        num_particles=500,
        dyn_std=12,
        sigma_mse=12,
        window_size_multiplier=0.8,
        changing_template=True,
        alpha=0.05,
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
