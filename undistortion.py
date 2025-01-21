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
