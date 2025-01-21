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
