import cv2
import depthai

device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["previewout", "object_tracker"],
    "ai": {
        #blob compiled for maximum 12 shaves
        #blob can be generated using: python3 depthai_demo.py -cnn mobilenet-ssd -sh 12
        #it will be written to <path_to_depthai>/resources/nn/mobilenet-ssd/mobilenet-ssd.blob.sh12cmx12NCE1
        "blob_file": "C:\code\github\oak-tracker\downloads\mobilenet-ssd\FP16\mobilenet-ssd.blob.sh7cmx7NCE1",
        "blob_file_config": "C:\code\github\oak-tracker\downloads\mobilenet-ssd\FP16\mobilenet-ssd.json",
        #"blob_file": "C:\code\github\oak-tracker\downloads\person-detection-retail-0013\model.blob",
        #"blob_file_config": "C:\code\github\oak-tracker\downloads\person-detection-retail-0013\config.json",
        "shaves" : 12,
        "cmx_slices" : 12,
        "NN_engines" : 1,
    },
    'ot': {
        'max_tracklets': 20,
        'confidence_threshold': 0.5,
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
tracklets = None

while True:
    for packet in p.get_available_data_packets():
        if packet.stream_name == 'object_tracker':
            tracklets = packet.getObjectTracker()
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            traklets_nr = tracklets.getNrTracklets() if tracklets is not None else 0

            for i in range(traklets_nr):
                tracklet = tracklets.getTracklet(i)
                if tracklet.getLabel() == 15:
                    left = tracklet.getLeftCoord()
                    top = tracklet.getTopCoord()
                    right = tracklet.getRightCoord()
                    bottom = tracklet.getBottomCoord()
                    tracklet_label = labels[tracklet.getLabel()]

                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))

                    middle_pt = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                    cv2.circle(frame, middle_pt, 0, (255, 0, 0), -1)
                    cv2.putText(frame, f"ID {tracklet.getId()}", middle_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    cv2.putText(frame, tracklet_label, (left, bottom - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, tracklet.getStatus(), (left, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del p
del device