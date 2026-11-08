import cv2
import numpy as np

VIDEO_PATH = r"C:\code\arc\AutoBots\autobot\HueyvOpenHELL_OVERHEAD-360p.mp4"
SCALE = 0.5          # shrink the video to half size so it's faster and fits your screen
MIN_AREA = 200       # ignore moving blobs smaller than this (in pixels)
WARMUP_FRAMES = 30   # frames the detector needs to learn the background first

# The region to watch, in pixels of the HALF-SIZE frame (SCALE = 0.5): TopLeft, TopRight, BottomRight, BottomLeft
ARENA = np.array([(518, 41), (1422, 57), (1919, 1079), (0, 1042)], dtype=np.int32)
# Math behind how you get the arena size:
### Current optimal frame size:
#  Frame:
# scaled   (TL, TR, BR, BL): [(518, 41), (1422, 57), (1919, 1079), (0, 1042)]
# original (TL, TR, BR, BL): [(1036, 82), (2844, 114), (3838, 2158), (0, 2084)]
# ARENA_POLY = [(518, 41), (1422, 57), (1919, 1079), (0, 1042)]
#
# ARENA_POLY is based on scaled, not original frame size
#
# Math here:
#
# 1. Scaled -> original:  original = scaled / SCALE  (with SCALE = 0.5, multiply by 2)
#      TL: (518, 41)    -> (518 / 0.5,  41 / 0.5)   = (1036, 82)
#      TR: (1422, 57)   -> (1422 / 0.5, 57 / 0.5)   = (2844, 114)
#      BR: (1919, 1079) -> (1919 / 0.5, 1079 / 0.5) = (3838, 2158)
#      BL: (0, 1042)    -> (0 / 0.5,    1042 / 0.5) = (0, 2084)
#    Original -> scaled is the reverse:  scaled = original * SCALE
#
# 2. Frame size check: the biggest scaled values are x = 1919 and y = 1079. Pixels are
#    counted from 0, so the last pixel is (size - 1), which means the scaled frame is
#    1920 x 1080. Dividing by SCALE:
#      width  = 1920 / 0.5 = 3840
#      height = 1080 / 0.5 = 2160
#    So the original video is 3840 x 2160 (4K). The BR corner (1919, 1079) sits on the
#    bottom-right pixel of the frame, and BL's x = 0 sits on the left edge.
#
# 3. Corner labels (how TL, TR, BR, BL were chosen from the four dragged points):
#      TL = smallest x + y     BR = largest x + y
#      TR = smallest y - x     BL = largest y - x
#    Check with the scaled corners:
#                 x + y    y - x
#      TL (518, 41)    559     -477
#      TR (1422, 57)  1479    -1365   <- smallest y - x
#      BR (1919, 1079) 2998   -840
#      BL (0, 1042)   1042     1042   <- largest y - x
#    TL has the smallest x + y (559) and BR the largest (2998), so the labels hold.
#
# 4. Changing SCALE: ARENA must be recomputed as  original * new SCALE.
#      Example, SCALE = 0.25:  TL (1036, 82) * 0.25 = (259, 20.5) -> (259, 20)
#                              BR (3838, 2158) * 0.25 = (959.5, 539.5) -> (959, 539)
#    (cast to int, since pixel coordinates must be whole numbers).
#
# 5. Arena area (shoelace formula) on the scaled coordinates, for reference:
#      area = 1/2 * |sum(x_i * y_(i+1) - x_(i+1) * y_i)|, going around the corners in order
#      TL->TR: 518*57   - 1422*41   =  29526 -  58302 =  -28776
#      TR->BR: 1422*1079 - 1919*57  = 1534338 - 109383 = 1424955
#      BR->BL: 1919*1042 - 0*1079   = 1999598 -      0 = 1999598
#      BL->TL: 0*41      - 518*1042 =       0 - 539756 = -539756
#      sum = 2855021    area = 2855021 / 2 = 1,427,510.5 px^2
#    The full scaled frame is 1920 * 1080 = 2,073,600 px^2, so the arena covers
#    1,427,510.5 / 2,073,600 = about 68.8% of the frame.
#    MIN_AREA = 200 px^2 is about 0.014% of the arena, so only tiny specks get filtered out.



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