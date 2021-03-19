# first, import all necessary modules
import time
from pathlib import Path
import cv2
import depthai
import numpy as np
from centroidTracker import PersonTrackerDebug, PersonTracker
import track_stats

pt = PersonTrackerDebug() # if debug else PersonTracker()

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)    # NN input mobilenet-ssd
#cam_rgb.setPreviewSize(544, 320)    # NN input person-detection-retail
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setPreviewKeepAspectRatio(False)

# manip = pipeline.createImageManip()
# manip.setResize(300,300)
# manip.setKeepAspectRatio(True)
# manip.setFrameType(depthai.RawImgFrame.Type.BGR888p)
# cam_rgb.video.link(manip.inputImage)

# Next, we want a neural network that will produce the detections
# detection_nn = pipeline.createNeuralNetwork()
detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models\mobilenet-ssd\FP16\mobilenet-ssd.blob.sh7cmx7NCE1')).resolve().absolute()))
# detection_nn.setBlobPath(str((Path(__file__).parent / Path('models\person-detection-retail-0013\model.blob')).resolve().absolute()))
detection_nn.setConfidenceThreshold(0.5)


# How many threads should the node use to run the network.
detection_nn.setNumInferenceThreads(2)

# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
# manip.out.link(detection_nn.input)
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
#mainp.video.link(xout_rgb.input)
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
device = depthai.Device(pipeline)
# And start. From this point, the Device will be in "running" mode and will start sending data via XLink
device.startPipeline()

# To consume the device results, we get two output queues from the device, with stream names we assigned earlier
q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")

frame = None
bboxes = []

# Since the bboxes returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
# receive the actual position of the bounding box on the image
def bbox_frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

# Main host-side application loop
while True:
    # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    if in_rgb is not None:
        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        
        bboxes = [
            [det.xmin, det.ymin, det.xmax, det.ymax] for 
            det in in_nn.detections if 
            det.label == 15 and det.confidence > 0.4]      # label = 15 is person

        # # when data from nn is received, it is also represented as a 1D array initially, just like rgb frame
        # bboxes = np.array(in_nn.getFirstLayerFp16())
        # # the nn detections array is a fixed-size (and very long) array. The actual data from nn is available from the
        # # beginning of an array, and is finished with -1 value, after which the array is filled with 0
        # # We need to crop the array so that only the data from nn are left
        # bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # # next, the single NN results consists of 7 values: id, label, confidence, x_min, y_min, x_max, y_max
        # # that's why we reshape the array from 1D into 2D array - where each row is a nn result with 7 columns
        # bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # # Finally, we want only these results, which confidence (ranged <0..1>) is greater than 0.8, and we are only
        # # interested in bounding boxes (so last 4 columns)
        # bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]

    if frame is not None:
        for raw_bbox in bboxes:
            # for each bounding box, we first normalize it to match the frame size
            bbox = bbox_frame_norm(frame, raw_bbox)
            # and then draw a rectangle on the frame to show the actual result
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # people tracker parse
        persons, deregistered_persons = pt.parse(frame, bboxes)

        # After all the drawing is finished, we show the frame on the screen
        cv2.imshow("preview", frame)

    # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
    if cv2.waitKey(1) == ord('q'):
        break