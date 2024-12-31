import cv2
import glob
import torch
import numpy as np
import pandas as pd
from moviepy.editor import *
from ball_tracker import BallTracker
from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from hawkeye_checker import in_out_checker
from utils import scene_detect, read_video, write

input_video_folder_path = r"input_videos"
output_video_folder_path = r"output_videos"

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
         hawk_eye, draw_trace=False, trace=7):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        hawk_eye: list of image numbers where shot was 'out'
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images
    """
    imgs_res = []
    width_minimap = (166*1.5)
    height_minimap = (350*1.5)
    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]
                
                # Draw Hawkeye stats
                width_hwk = 70
                height_hwk = 35
                start_x_hwk = img_res.shape[1]-193
                start_y_hwk = img_res.shape[0]-400
                end_x_hwk = start_x_hwk + width_hwk
                end_y_hwk = start_y_hwk + height_hwk
                if hawk_eye[i] == 1:
                    img_res = cv2.rectangle(img_res, (start_x_hwk, start_y_hwk), (end_x_hwk, end_y_hwk), (0, 0, 0), -1)
                    text = "Hawkeye"
                    img_res = cv2.putText(img_res, text, (start_x_hwk+3, start_y_hwk+12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                    text = "OUT"
                    img_res = cv2.putText(img_res, text, (start_x_hwk+20, start_y_hwk+30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)
                    

                minimap = court_img.copy()

                # draw persons
                persons = persons_top[i] + persons_bottom[i]                    
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # transmit person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                           radius=0, color=(255, 0, 0), thickness=80)

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)

        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
    return imgs_res        


if __name__ == '__main__':
    # Getting all videos from the input directory
    input_videos = glob.glob(f"{input_video_folder_path}/*")
    if len(input_videos) == 0:
        print("No video is present in the input folder")
    else:    
        for video_path in input_videos:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            frames, fps, audioclip, input_name = read_video(f"{video_path}") 
            scenes = scene_detect(f"{video_path}")    

            print('ball and bounce detection')
            ball_tracker = BallTracker(model_path='models/tracknet_model_best.pt', device=device)
            ball_detections = ball_tracker.detect_frames(frames)
            ball_track = ball_tracker.interpolate_ball_positions(ball_detections)

            bounce_detector = BounceDetector(path_model='models/bounce_model/')
            x_ball = [x[0] for x in ball_track]
            y_ball = [x[1] for x in ball_track]
            bounces, preds = bounce_detector.predict(x_ball, y_ball, fps)

            print('court detection')
            court_detector = CourtDetectorNet(path_model='models/model_tennis_court_det.pt', device=device)
            homography_matrices, kps_court = court_detector.infer_model(frames)

            print('person detection')
            person_detector = PersonDetector(device)
            persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

            # For Hawk eye 0:IN, 1:OUT
            hawk_eye = np.full(shape=len(frames), fill_value=np.nan)
            for bounce in bounces:
                hawk_eye[bounce] = in_out_checker(kps_court[bounce], ball_track[bounce])

            hawk_eye = pd.DataFrame(hawk_eye)
            hawk_eye = hawk_eye.ffill(axis=0)
            hawk_eye = hawk_eye.fillna(0)
            hawk_eye = list(hawk_eye[0])

            imgs_res = main(frames, scenes, bounces,
                            ball_track, homography_matrices, kps_court,
                            persons_top, persons_bottom, hawk_eye, draw_trace=True)

            write(imgs_res, fps, f"{output_video_folder_path}", audioclip, input_name)
            torch.cuda.empty_cache()
