import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps_1'


def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    img = load(os.path.join(DATA_PATH, 'ps1-input3.png'))

    # Pad the image
    img = np.pad(
        img,
        pad_width=((100, 100), (100, 100)),
        mode='constant',
        constant_values=0
    )

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # To draw color circles

    src_pts = []
    max_points = 4

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < max_points:
            src_pts.append([x, y])
            print(f'Point {len(src_pts)}: ({x}, {y})')
            cv2.circle(img_color, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    print("Click on 4 source points in the image. Press ESC when done.")

    while True:
        cv2.imshow('Image', img_color)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        if len(src_pts) == max_points:
            print("4 points selected.")
            break

    cv2.destroyAllWindows()

    src_pts = np.array(src_pts, dtype=np.float32)

    x_size = img.shape[1]
    y_size = img.shape[0]
    dst_pts = np.array([
        [0, 0],
        [x_size, 0],
        [x_size, y_size],
        [0, y_size]
    ], dtype=np.float32)

    H, status = cv2.findHomography(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(img, H, (x_size, y_size))

    cv2.imshow("Warped", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(DATA_PATH, 'ps1-input3-warped.png'), warped_image)
