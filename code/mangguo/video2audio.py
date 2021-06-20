from ffmpy import FFmpeg
import os
from tqdm import tqdm

paths = ['data_B/', 'data_A/train/', 'data_A/test/', 'data_C/']


for path in paths:

    for file in tqdm(os.listdir(path)):

        ff = FFmpeg(
            # ffmpeg.exe的路径
            executable=r"D:\搜狗高速下载\ffmpeg-N-101953-g4e64c8fa29-win64-gpl\ffmpeg-N-101953-g4e64c8fa29-win64-gpl\bin\ffmpeg",
            inputs={path+file: None},  # 视频路径
            outputs={'wavs/'+file.replace('.mp4', '.wav'): '-ar 32000 -ac 1 -y'}  # 音频路径 -ar 采样率 -ac 声道 -y 覆盖同名文件
        )
        # print(ff.cmd)
        ff.run()
