"""Standalone OAK-D live camera viewer.

Run this file directly to initialize the OAK-D and display its RGB camera feed.
Press q in the video window to stop.
"""

import cv2
import depthai as dai


def main():
    # Initialize the OAK-D device and camera pipeline.
    device = dai.Device()

    try:
        with dai.Pipeline(device) as pipeline:
            camera = pipeline.create(dai.node.Camera).build()
            rgb_output = camera.requestOutput(
                (600, 500),
                type=dai.ImgFrame.Type.RGB888p,
            )
            output_queue = rgb_output.createOutputQueue()

            pipeline.start()
            print("OAK-D started. Press 'q' in the camera window to quit.")

            while pipeline.isRunning():
                frame_packet = output_queue.get()
                frame = frame_packet.getCvFrame()

                cv2.imshow("OAK-D Live Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping OAK-D viewer...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
