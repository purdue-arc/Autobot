# STILL MAKES YOU PICK 4 POINTS, NOT AUTOMATIC
# IMPROVE THIS LATER
import cv2 as cv
import numpy as np

def pick_points(frame, n=4, win="click corners"):
    scale = 1000 / max(frame.shape[:2])            # scale to make image fit on screen
    view = cv.resize(frame, None, fx=scale, fy=scale)
    pts = []

    # Place dots on where mouse is clicked
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv.circle(view, (x, y), 6, (0, 255, 0), -1)
            cv.putText(view, str(len(pts)), (x + 8, y - 8),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.imshow(win, view)

            return frame

    cv.imshow(win, view)
    cv.setMouseCallback(win, on_mouse)
    while len(pts) < n:
        if cv.waitKey(20) == 27:                # esc aborts
            break
    cv.waitKey(300)
    cv.destroyWindow(win)
    return np.array(pts, dtype=np.float32) / scale   # back to full-res pixels

# Adjust accordingly
out_h = 800
out_w = int(out_h * 1)

# img = cv.imread('Photos/mishima1.jpg')
cap = cv.VideoCapture('Videos/test_video.mp4')

# Capture the first frame of the video to pick points from(only really works with a perfectly still camera)
isTrue, first_frame = cap.read()
first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Cover corners of the video
# ORDER IS IMPORTANT, Go top left -> top right -> bottom right -> bottom left
# Clockwise order
src = pick_points(first_frame)
dst = np.float32([[0, 0],                     # Top left
                [out_w - 1, 0],             # Top right
                [out_w - 1, out_h - 1],     # Bottom right
                [0, out_h - 1]])


# Makes sure the first frame is included so we can use it to select points
cap.set(cv.CAP_PROP_POS_FRAMES, 0)      

# Add warpPerspective with out_h and out_w
# Make image/video speed and quality good
while True:
    isTrue, frame = cap.read()
    if not isTrue:
        print("Error: Can't read")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Finds where new points are in the video using a original input image and second input image/video
    # Calculates using optical flow
    # Maybe improve to using canny or something else later, but this works for now
    # Optical flow also drifts sometimes, so it isn't completely consistent
    new_points, status, _ = cv.calcOpticalFlowPyrLK(first_gray, gray, src, None,
    winSize=(31, 31) #larger values give more chances for fast motion detection, but may increase motion blurriness
    , maxLevel=3)

    # if status.all():            # makes sure it only updated if the 4 points are found in the video 
    #     src = new_points
    # first_gray = gray

    
    # Compute H (homography matrix)
    H = cv.getPerspectiveTransform(src.reshape(4, 2), dst)
    flat = cv.warpPerspective(frame, H, (out_w, out_h))
    
    cv.imshow('Warped Image', flat)

    key = cv.waitKey(1)
    if key & 0xFF == ord('d') or cv.getWindowProperty('Warped Image', cv.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv.destroyAllWindows()
