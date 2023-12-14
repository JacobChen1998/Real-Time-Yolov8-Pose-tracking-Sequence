print(1233333)
from moviepy.editor import VideoFileClip
import os

# def convert_avi_to_gif(input_avi, output_gif, fps=10):
#     video_clip = VideoFileClip(input_avi)
#     video_clip.write_gif(output_gif, fps=fps, opt="nq")

#     print("GIF created and saved to", output_gif)


def convert_avi_to_gif(input_avi, output_gif, fps=10):
    # 讀取AVI視頻
    video_clip = VideoFileClip(input_avi)

    # 將AVI轉換為GIF，使用非常規的量化方法
    video_clip.write_gif(output_gif, fps=fps, opt="nq")

    # 使用gifsicle進行進一步的壓縮
    os.system(f"gifsicle -O3 {output_gif} -o {output_gif}")

    print("GIF created and saved to", output_gif)

input_avi_path = "../1.avi"
output_gif_path = "../1.gif"

convert_avi_to_gif(input_avi_path, output_gif_path, fps=10)
