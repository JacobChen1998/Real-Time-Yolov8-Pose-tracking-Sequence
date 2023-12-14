tracker = 'bytetrack.yaml'#'botsort.yaml

import cv2, time, argparse,torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--kptSeqNum", default=None, required=False)
parser.add_argument("--max_miss", type=int, default=4, required=False)
parser.add_argument("--GPU_no", type=int, default=0, required=False)
parser.add_argument("--video_path", required=True)
args = parser.parse_args()

video_path = args.video_path
GPU_no = args.GPU_no
max_miss = args.max_miss
kptSeqNum = args.kptSeqNum
if kptSeqNum is not None:
    kptSeqNum = int(kptSeqNum)
# Load the YOLOv8 model
model = YOLO('weights/yolov8x-pose-p6.pt')
cap = cv2.VideoCapture(video_path)
track_history=defaultdict(lambda: [])
drop_counting=defaultdict(lambda: 0)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    kptThres = 0.1
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < kptThres:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            # print((int(x_coord), int(y_coord)))
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<kptThres or conf2<kptThres:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return im

count = 0
while cap.isOpened():
    success, frame = cap.read()
    t1 = time.time()
    if not success:
        break
    result = model.track(frame,
                        persist=True,
                        tracker=tracker,
                        verbose = False
                        #   show=True,
                        #   conf=0.7,
                        #    iou=0.5,
                        )[0]
    t2 = time.time()
    # print(f"Time cost : {t2-t1} sec")
    boxes = result.boxes.xywh.cpu()
    keypoints = result.keypoints.data#.cpu().numpy()

    track_ids = result.boxes.id#.int().cpu().tolist()
    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()

    # print("track_ids : ",track_ids)
    diff = list(set(list(set(track_history.keys()))).difference(track_ids))
    for d in diff:
        if drop_counting[d] > max_miss:
            del drop_counting[d]
            del track_history[d]
        else:
            drop_counting[d]+=1

    track_ids_conform_frame_num = [] ; poseTrackResult = []
    boxess = []
    for box, track_id,keypoint in zip(boxes, track_ids,keypoints):
        track = track_history[track_id]
        track.append(keypoint.unsqueeze(0))

        if kptSeqNum is not None:
            if len(track) > kptSeqNum:  
                track.pop(0)
            if len(track) == kptSeqNum:
                poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
                track_ids_conform_frame_num.append(track_id)  
        else:
            poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
            track_ids_conform_frame_num.append(track_id)  
        boxess.append(box)
    # print(track_ids_conform_frame_num)
    for resultIdx,track_id in enumerate(track_ids_conform_frame_num):
        number_of_seq = poseTrackResult[resultIdx].numpy().shape
        current_kpt = poseTrackResult[resultIdx][0,-1,:,:].numpy().flatten()
        x,y,w,h = boxess[resultIdx]
        x1,y1,x2,y2 = int(x-(w/2)),int(y-(h/2)),int(x+(w/2)),int(y+(h/2))
        # print(x1,y1,x2,y2)
        text = f"tid:{track_id} with seq : {number_of_seq}"
        # print("number_of_seq :",number_of_seq)
        # print("current kpt : ",current_kpt)
        frame = plot_skeleton_kpts(frame,current_kpt,3)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(frame, text, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Result', frame)
    # cv2.imwrite(f"results/{count}.png",frame)
    if cv2.waitKey(1) == ord('q'):
        break
    count+=1
cap.release()
cv2.destroyAllWindows()

