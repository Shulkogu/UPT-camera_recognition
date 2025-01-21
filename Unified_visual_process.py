import cv2
import numpy as np
import os

def calibrate_camera(calib_file="camera_calib.npz"):
    # Define the chessboard size
    chessboard_size = (9, 6)  # Adjust based on your printed chessboard

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    cap = cv2.VideoCapture(0)
    print("Point the camera at the chessboard and press 'c' to capture frames for calibration.")
    print("Press 'q' to finish calibration.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret:
            obj_points.append(objp)
            img_points.append(corners)
            print("Frame captured.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) > 0:
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        if ret:
            np.savez(calib_file, mtx=mtx, dist=dist)
            print(f"Calibration successful. Parameters saved to {calib_file}")
        else:
            print("Calibration failed.")
    else:
        print("No frames captured for calibration.")

def undistort_video(calib_file="camera_calib.npz"):
    if not os.path.exists(calib_file):
        print(f"Calibration file '{calib_file}' not found. Please run the calibration first.")
        return

    # Load calibration data
    data = np.load(calib_file)
    mtx, dist = data['mtx'], data['dist']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Undistort the frame
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Crop the image
        x, y, w, h = roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        cv2.imshow('Undistorted Video', undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_test_tubes(frame):
    """
    Detect test tubes in a given frame using contour detection.

    Parameters:
        frame (numpy.ndarray): Input frame from the camera.

    Returns:
        processed_frame (numpy.ndarray): Frame with detected test tubes highlighted.
        tube_positions (list): List of bounding boxes [(x, y, w, h)] for detected test tubes.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store tube positions
    tube_positions = []

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Filter contours based on aspect ratio and size (specific to test tubes)
        aspect_ratio = h / w
        if 2.0 < aspect_ratio < 6.0 and w > 20 and h > 50:  # Adjust thresholds as needed
            tube_positions.append((x, y, w, h))
            # Draw the bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, tube_positions

def main():
    """
    Main function to calibrate the camera, undistort the video, and detect test tubes.
    """
    mode = input("Enter 'calibrate', 'undistort', or 'detect': ").strip().lower()
    if mode == "calibrate":
        calibrate_camera()
    elif mode == "undistort":
        undistort_video()
    elif mode == "detect":
        cap = cv2.VideoCapture(0)  # Open the default camera
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Detect test tubes in the frame
            processed_frame, tube_positions = detect_test_tubes(frame)

            # Display the frame with detected test tubes
            cv2.imshow("Test Tube Detection", processed_frame)

            # Print positions for debugging
            if tube_positions:
                print("Detected test tube positions:", tube_positions)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid mode. Please enter 'calibrate', 'undistort', or 'detect'.")

if __name__ == "__main__":
    main()
