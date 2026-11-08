import cv2
import numpy as np

VIDEO_PATH = r"C:\code\arc\AutoBots\autobot\Huey-vs-OpenHELL-3lb-OVERHEAD-360p.mp4"
SCALE = 0.5          # shrink the video to half size so it's faster and fits your screen
MIN_AREA = 400       # ignore moving blobs smaller than this (in pixels)
WARMUP_FRAMES = 30   # frames the detector needs to learn the background first


SCALE = 1.5

# The region to watch, in pixels of the HALF-SIZE frame (SCALE = 0.5): TopLeft, TopRight, BottomRight, BottomLeft
ARENA = np.array([
    (258, 42),
    (690, 33),
    (959, 539),
    (0, 519)
], dtype=np.int32)

# Math behind how you get the arena size:
### Current optimal frame size:
#  Frame:
# scaled   (TL, TR, BR, BL): [(258, 42), (690, 33), (959, 539), (0, 519)]
# original (TL, TR, BR, BL): [(172, 28), (460, 22), (639, 359), (0, 346)]
# ARENA_POLY = [(258, 42), (690, 33), (959, 539), (0, 519)]
#
# ARENA_POLY is based on scaled, not original frame size
#
# Math here:
#
# 1. Scaled -> original: original = scaled / SCALE (with SCALE = 1.5)
#      TL: (258, 42)  -> (258 / 1.5, 42 / 1.5) = (172, 28)
#      TR: (690, 33)  -> (690 / 1.5, 33 / 1.5) = (460, 22)
#      BR: (959, 539) -> (959 / 1.5, 539 / 1.5) = (639, 359)
#      BL: (0, 519)   -> (0 / 1.5, 519 / 1.5) = (0, 346)
#    Original -> scaled is the reverse: scaled = original * SCALE
#
# 2. Frame size check: the biggest scaled values are x = 959 and y = 539.
#    Pixels are counted from 0, so the last pixel is (size - 1), which means
#    the scaled frame is 960 x 540. Dividing by SCALE:
#      width  = 960 / 1.5 = 640
#      height = 540 / 1.5 = 360
#    So the original video is 640 x 360. The BR corner (959, 539) sits on
#    the bottom-right pixel of the scaled frame, and BL's x = 0 sits on the left edge.
#
# 3. Corner labels (how TL, TR, BR, BL were chosen from the four dragged points):
#      TL = smallest x + y     BR = largest x + y
#      TR = smallest y - x     BL = largest y - x
#    Check with the scaled corners:
#                 x + y    y - x
#      TL (258, 42)   300      -216
#      TR (690, 33)   723      -657  <- smallest y - x
#      BR (959, 539)  1498     -420
#      BL (0, 519)    519       519  <- largest y - x
#    TL has the smallest x + y (300) and BR the largest (1498), so the labels hold.
#
# 4. Changing SCALE: ARENA must be recomputed as original * new SCALE.
#      Example, SCALE = 0.5:
#      TL (172, 28) * 0.5 = (86, 14)
#      TR (460, 22) * 0.5 = (230, 11)
#      BR (639, 359) * 0.5 = (319.5, 179.5) -> (319, 179)
#      BL (0, 346)   * 0.5 = (0, 173)
#    (cast to int, since pixel coordinates must be whole numbers).
#
# 5. Arena area (shoelace formula) on the scaled coordinates, for reference:
#      area = 1/2 * |sum(x_i * y_(i+1) - x_(i+1) * y_i)|, going around the corners in order
#      TL->TR: 258*33  - 690*42  = 8514 - 28980 = -20466
#      TR->BR: 690*539 - 959*33 = 371910 - 31647 = 340263
#      BR->BL: 959*519 - 0*539  = 497721 - 0 = 497721
#      BL->TL: 0*42    - 258*519 = 0 - 133902 = -133902
#      sum = 693616    area = 693616 / 2 = 346,808 px^2
#    The full scaled frame is 960 * 540 = 518,400 px^2, so the arena covers
#    346,808 / 518,400 = about 66.9% of the frame.
#    MIN_AREA = 200 px^2 is about 0.058% of the arena, so only tiny specks get filtered out.


cap = cv2.VideoCapture(VIDEO_PATH)

# The background subtractor learns what the scene normally looks like.
# Anything that differs from that (a moving bot) shows up as white in its mask.
bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)

# WINDOW_NORMAL makes the window resizable: drag its edges or hit maximize and the
# whole picture (video, tint, and boxes) stretches to fit, like a browser tab.
# WINDOW_KEEPRATIO stops the video from getting squished out of shape.
# This only changes how the finished picture is DISPLAYED. Detection still runs on
# the fixed SCALE frame below, so resizing the window never breaks the arena
# coordinates or restarts the background learning.
cv2.namedWindow("Moving objects", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Moving objects", 960, 540)   # starting window size in screen pixels

frame_number = 0

while True:
    ok, frame = cap.read()          # ok is False when the video ends
    if not ok:
        break

    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    frame_number += 1

    # Make a black image the same size as the frame, then paint the arena white.
    # Later we use it to throw away any motion outside the arena.
    region = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(region, [ARENA], 255)

    # Find motion (white = moving), then keep only the part inside the arena
    mask = bg.apply(frame)
    mask = cv2.bitwise_and(mask, region)

    # Remove tiny specks of noise, then grow the blobs a little so nearby
    # pieces of one bot merge into a single blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.bitwise_and(mask, region)   # growing may have leaked outside, so clip again

    # Tint the arena gray: paint gray on a copy, then blend the copy at 25% strength
    gray = frame.copy()
    cv2.fillPoly(gray, [ARENA], (128, 128, 128))
    frame = cv2.addWeighted(gray, 0.25, frame, 0.75, 0)

    # Turn each white blob into a box (skipped during warmup to avoid false boxes)
    if frame_number > WARMUP_FRAMES:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Moving objects", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):   # wait 30 ms, quit if q is pressed
        break

cap.release()
cv2.destroyAllWindows()