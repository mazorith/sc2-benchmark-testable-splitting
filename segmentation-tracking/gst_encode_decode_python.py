import time
import cv2

# Cam properties
fps = 30.
frame_width = 640
frame_height = 480
# Create capture

# cap = cv2.VideoCapture(0)




# # Set camera properties
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# cap.set(cv2.CAP_PROP_FPS, fps)
gst_in = 'autovideosrc ! videoconvert ! appsink' # encode decode here, then encode to expected format

cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Define the gstreamer sink
# gst_out = "appsrc ! filesink location=/Users/AkhilG/Desktop/hevctest/gstreamerTest/video_1.avi "
# gst_str_rtp = "appsrc  ! autovideosink"
# gst_str_rtp = "appsrc ! autovideosink"  #"videoconvert ! x264 ! matroskamux ! filesink location=the_gstreamer_enjoyer.mkv"

# gst_out = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000"
gst_out = "appsrc ! videoconvert ! x265enc ! h265parse !filesink location=/Users/AkhilG/Desktop/hevctest/gstreamerTest/gst_test_3.hevc"
# Check if cap is open

if cap.isOpened() is not True:
    print("Cannot open camera. Exiting.")
    quit()

# Create videowriter as a SHM sink
out = cv2.VideoWriter(gst_out, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height), True)
# Loop it
frame_count = fps * 10
while frame_count >= 0:

    # Get the frame
    ret, frame = cap.read()
    # Check
    if ret is True:
        # Flip frame
        frame = cv2.flip(frame, 1)
        # cv2.imshow('Webcam', frame); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
        # Write to SHM
        out.write(frame)
        print(frame_count)
        frame_count -= 1
    else:
        print("Camera error.")
        time.sleep(10)

cap.release()