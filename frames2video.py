from PIL import Image
import imageio
import os

def resize_images(input_folder, output_folder, size=(640, 360)):
    os.makedirs(output_folder, exist_ok=True)
    filenames = os.listdir(input_folder)
    filenames.sort(key=lambda x: int(x[:-4]))

    for filename in filenames:
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.ANTIALIAS)
            output_path = os.path.join(output_folder, f"resized_{filename}")
            img_resized.save(output_path)

    print("Images resized and saved to", output_folder)

def create_gif(input_folder, output_gif, duration=0.6):
    images = []
    filenames = os.listdir(input_folder)
    # filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        if filename.startswith("resized_"):
            img_path = os.path.join(input_folder, filename)
            images.append(imageio.imread(img_path))

    imageio.mimsave(output_gif, images, duration=duration)
    print("GIF created and saved to", output_gif)

input_folder_path = "./results"
output_folder_path = "./results_resize"
output_gif_path = "demo/no_set_max.gif"

# 將圖片resize
# resize_images(input_folder_path, output_folder_path, size=(640, 360))

# 創建GIF
create_gif(output_folder_path, output_gif_path, duration=0.001)
