# Yolov8-Pose-tracking-Sequence

Yolov8-pose is a one of the SOTA pose estimation methods, which supported by [Ultralytics](https://github.com/ultralytics).
In their [document]([https://docs.ultralytics.com/tasks/pose/#models](https://docs.ultralytics.com/modes/track/)), the usage of Pose tracking is mentioned.
Based on Pose estimation with tracking algorithm, we can know the current poses with a unique identity. 

The primary purpose of attitude tracking is to acquire successive attitudes of each identity and assemble them into sequential information for purposes such as:
1. Animation and game production: Motion capture using real actors can create realistic animated character movements. These captured gesture sequences can be used to manipulate virtual characters to make their movements more realistic.
2. Sports Science: Researchers can use posture sequences to analyze athletes' movements to help improve their performance or reduce the risk of injury.
3. Medicine and Rehabilitation: By analyzing a patient's posture sequences, doctors and therapists can assess joint range of motion, gait abnormalities, or other sports-related problems.
4. Robotics: Gesture sequences can help robots learn and mimic the movements of humans or other creatures.
5. Virtual and Augmented Reality: Tracking a user's posture in a virtual reality or augmented reality environment can provide a more immersive experience.

However, the original repository has not yet provided an effective mechanism for each identity to assemble its data sequence.
Therefore, I was inspired to propose an algorithm for posture sequence development.
The concept of this method:
![Concept of this method](https://github.com/JacobChen1998/Real-Time-Yolov8-Pose-tracking-Sequence/blob/main/pose_track_illustruction.png)

Demo:
![Concept of this method](https://github.com/JacobChen1998/Real-Time-Yolov8-Pose-tracking-Sequence/blob/main/demo.gif)

## Quick start with anaconda 

#### 0. Clone and into project 
```
git clone https://github.com/JacobChen1998/Real-Time-Yolov8-Pose-tracking-Sequence
cd Real-Time-Yolov8-Pose-tracking-Sequence/
```

#### 1. Environment create
```
conda create --name posetrack python=3.8
```

#### 2. Environment activate
```
conda activate posetrack
```

#### 3. Packages install
```
pip install ultralytics
```

#### 4. Run the code
```
python main.py --video_path <your_video_path>
```
(argparse option):
```
--max_miss : number of miss track than empty tracked seqeunce
--kptSeqNum : The max keypoint sequence number. The default value is None which means it will append until max_miss trigger.
--GPU_no : GPU number of using. The default value is 0 which means first GPU.

ex: python main.py --video_path <your_video_path> --kptSeqNum 40 --max_miss 5 --GPU_no 0
```

Without setting kptSeqNum:
![Org_frames](https://github.com/JacobChen1998/Real-Time-Yolov8-Pose-tracking-Sequence/blob/main/demo/no_set_max.gif)

Setting kptSeqNum=40:
![Org_frames](https://github.com/JacobChen1998/Real-Time-Yolov8-Pose-tracking-Sequence/blob/main/demo/set_max_40.gif)
