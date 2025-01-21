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
    Main function to capture video, detect test tubes, and display results in real-time.
    """
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

if __name__ == "__main__":
    main()
