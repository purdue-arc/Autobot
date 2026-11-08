import cv2 as cv
import numpy as np

def main():
    # 1. Initialize the video capture (0 is usually the default/USB camera)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return

    # 2. Define source points from your video and corresponding destination points
    # Format: np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # Example: Selecting a distorted quad in a 640x480 frame

    src_pts = np.float32([
        [70,   50],   # Top-Left
        [470,  50],   # Top-Right
        [470,  450],  # Bottom-Right
        [70,   450]   # Bottom-Left (provided)
    ])    
    # Define the output dimensions for the rectified frame
    out_w, out_h = 600, 500
    # Top-Left
    dst_pts = np.float32([[0,0],          
        [out_w, 0],      # Top-Right
        [out_w, out_h],  # Bottom-Right
        [0, out_h]       # Bottom-Left
    ])

    # 3. Pre-compute the 3x3 homography matrix outside the loop for performance
    H = cv.getPerspectiveTransform(src_pts, dst_pts)

    print("Pipeline running. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 4. Warp the perspective of the frame using the pre-computed matrix
        # cv.INTER_LINEAR balances speed and image quality perfectly
        rectified_frame = cv.warpPerspective(
            frame, 
            H, 
            (out_w, out_h), 
            flags=cv.INTER_LINEAR
        )

        # 5. Display both the original feed and the rectified output
        cv.imshow("Original Feed", frame)
        cv.imshow("Homography Pipeline Output", rectified_frame)

        # Break loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Clean up resources
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
