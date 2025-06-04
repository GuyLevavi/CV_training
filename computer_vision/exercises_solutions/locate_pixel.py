import cv2

# Global variables to store mouse click coordinates
mouse_x, mouse_y = -1, -1
drawing = False

def draw_circle(event, x, y, flags, param):
    """
    Mouse callback function to draw a circle at the clicked pixel
    and store its coordinates.
    """
    global mouse_x, mouse_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mouse_x, mouse_y = x, y
        print(f"Clicked pixel coordinates: (x={x}, y={y})")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def main():
    """
    Reads the first frame of an AVI video, displays it,
    and allows interactive pixel highlighting.
    """
    video_path = r'C:\Users\user\Desktop\CV Training Guy\CV_training\computer_vision\course_resources\ps6\input\pres_debate.avi'

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Create a window to display the frame
    window_name = 'First Frame - Click a Pixel'
    cv2.namedWindow(window_name)

    # Set the mouse callback function for the window
    cv2.setMouseCallback(window_name, draw_circle)

    while True:
        display_frame = frame.copy() # Create a copy to draw on

        # If a pixel has been clicked, draw a circle on it
        if mouse_x != -1 and mouse_y != -1:
            cv2.circle(display_frame, (mouse_x, mouse_y), 5, (0, 255, 0), -1) # Green circle

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to exit
        if key == ord('q'):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()