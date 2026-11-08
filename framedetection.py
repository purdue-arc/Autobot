import cv2
import numpy as np

VIDEO_PATH = r"C:\code\arc\AutoBots\autobot\HueyvOpenHELL_OVERHEAD-ORIGINAL.mp4"
FRAME_NUMBER = 10    # which frame to grab (1 = first frame of the video)
SCALE = 0.5          # display size: 0.5 = half of the original video size
GRAB_RADIUS = 20     # how close (in pixels) you must click to grab a corner

# Start from the full frame, or paste a saved shape here (scaled pixels), e.g.
# [(518, 41), (1422, 57), (1919, 1079), (0, 1042)]
ARENA_POLY = None

# Read the video up to the chosen frame and keep that one
cap = cv2.VideoCapture(VIDEO_PATH)
for _ in range(FRAME_NUMBER):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read frame %d from the video" % FRAME_NUMBER)
cap.release()

frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
H, W = frame.shape[:2]

# The four corners, as [x, y] lists so they can be changed while dragging
if ARENA_POLY:
    poly = [list(p) for p in ARENA_POLY]
else:
    poly = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]

# drag = index of the corner being dragged (None if none)
# new_pts = corners being clicked in after pressing 'r' (None if not active)
state = {"drag": None, "new_pts": None}


def get_coords(original=False):
    """Return the corners as (x, y) tuples in the order TL, TR, BR, BL.

    original=False -> scaled pixels (what you see on screen)
    original=True  -> pixels in the original, full-size video
    """
    pts = [(int(x), int(y)) for x, y in poly]

    # Label the corners by position instead of by the order you placed them
    tl = min(pts, key=lambda p: p[0] + p[1])   # smallest x + y
    br = max(pts, key=lambda p: p[0] + p[1])   # largest x + y
    tr = min(pts, key=lambda p: p[1] - p[0])   # smallest y - x
    bl = max(pts, key=lambda p: p[1] - p[0])   # largest y - x
    pts = [tl, tr, br, bl]

    if original:
        pts = [(int(round(x / SCALE)), int(round(y / SCALE))) for x, y in pts]
    return pts


def on_mouse(event, x, y, flags, param):
    # Keep the mouse position inside the image
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))

    # Otherwise: press on a corner to grab it, move to drag it, release to drop it
    if event == cv2.EVENT_LBUTTONDOWN:
        best = None
        best_dist = GRAB_RADIUS
        for i, (px, py) in enumerate(poly):
            d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if d <= best_dist:
                best = i
                best_dist = d
        state["drag"] = best
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
        poly[state["drag"]] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        state["drag"] = None


cv2.namedWindow("Arena", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Arena", on_mouse)

while True:
    out = frame.copy()
    pts = np.array(poly, dtype=np.int32)

    # Tint the selected region: paint yellow on a copy, then blend at 20% strength
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)

    # Outline the region (black edge underneath so it shows on any background)
    cv2.polylines(out, [pts], True, (0, 0, 0), 7)
    cv2.polylines(out, [pts], True, (0, 255, 255), 4)

    # Numbered corner handles you can drag
    for i, (px, py) in enumerate(poly):
        cv2.circle(out, (px, py), 9, (0, 0, 0), -1)
        cv2.circle(out, (px, py), 7, (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # While placing new corners with 'r', show a prompt
    if state["new_pts"] is not None:
        cv2.putText(out, "Click 4 corners in order (%d/4)" % len(state["new_pts"]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Arena", out)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        # Start over: click four new corners
        state["new_pts"] = []
        state["drag"] = None
    elif key == ord("c"):
        # Reset to the full frame
        poly[:] = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
        state["new_pts"] = None
    elif key == ord("p"):
        # Print the coordinates to the console
        print("-" * 50, flush=True)
        print("scaled   (TL, TR, BR, BL):", get_coords(), flush=True)
        print("original (TL, TR, BR, BL):", get_coords(original=True), flush=True)
        print("-" * 50, flush=True)

cv2.destroyAllWindows()