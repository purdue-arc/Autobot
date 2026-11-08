import cv2 as cv
import numpy as np

def pick_points(img, n=4, win="click corners"):
    scale = 1000 / max(img.shape[:2])            # scale to make image fit on screen
    view = cv.resize(img, None, fx=scale, fy=scale)
    pts = []

    # Place dots on where mouse is clicked
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv.circle(view, (x, y), 6, (0, 255, 0), -1)
            cv.putText(view, str(len(pts)), (x + 8, y - 8),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.imshow(win, view)

    cv.imshow(win, view)
    cv.setMouseCallback(win, on_mouse)
    while len(pts) < n:
        if cv.waitKey(20) == 27:                # Esc aborts
            break
    cv.waitKey(300)
    cv.destroyWindow(win)
    return np.array(pts, dtype=np.float32) / scale   # back to full-res pixels

img = cv.imread('Photos/mishima1.jpg')

# Cover corners of the photo
# ORDER IS IMPORTANT, Go top left -> top right -> bottom right -> bottom left
# Clockwise order
src = pick_points(img)

# Destination points: a flat rectangle with the cover's real aspect ratio.
# Guesstimate of book height
out_h = 1200
out_w = int(out_h * 0.65)
dst = np.array([[0, 0],                     #Top left
                [out_w - 1, 0],             #Top right
                [out_w - 1, out_h - 1],     #Bottom right
                [0, out_h - 1]], dtype=np.float32)

# Compute H (homography matrix)
H, _ = cv.findHomography(src, dst)

flat = cv.warpPerspective(img, H, (out_w, out_h), flags=cv.INTER_CUBIC)
cv.namedWindow("flat image", cv.WINDOW_NORMAL)
cv.resizeWindow("flat image", 442, 680)
cv.imshow("flat image", flat)
cv.waitKey(0)
cv.destroyAllWindows()