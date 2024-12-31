from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import os 
import cv2
import glob
from moviepy.editor import *

def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    """
    video_manager = VideoManager([path_video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    # video_manager.set_downscale_factor() # Enable for faster pre-processing
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    scenes = [[x[0].frame_num, x[1].frame_num]for x in scene_list]    
    return scenes

def read_video(path_video):
    # Make a tmp directory
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Extracting audio
    audioclip = AudioFileClip(path_video)
    new_audioclip = CompositeAudioClip([audioclip])
    input_name = os.path.basename(f"{path_video}")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize frame to 1920x1080
            if frame.shape[1] != 1920 or frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps, new_audioclip, input_name

 
def write(imgs_res, fps, path_output_video, audioclip, input_name):
    height, width = imgs_res[0].shape[:2]
    tmp_dir_path = f"tmp/output_old_{input_name}.avi"
    out = cv2.VideoWriter(tmp_dir_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    
    
    # avi to mp4
    video_clip_final = VideoFileClip(f"tmp/output_old_{input_name}.avi")
    path, file_name = os.path.split(f"{path_output_video}/output_{input_name}.avi")
    output_name = os.path.join(path_output_video, os.path.splitext(file_name)[0])

    # adding audio
    video_clip_final.audio = audioclip
    video_clip_final.write_videofile(output_name, logger=None, audio_codec='aac')

    ## Deleting redundant files
    files = glob.glob('tmp/*')
    for f in files:
       os.remove(f)
    if os.path.exists('tmp'):
       os.rmdir('tmp')    
