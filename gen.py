import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import shutil

def tensor_to_image(tensor):
    """Convert a tensor to a numpy array for visualization."""
    image = tensor[0].cpu().float().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0
    return np.clip(image, 0, 1)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk."""
    image_pil = Image.fromarray((image_numpy * 255).astype(np.uint8))
    image_pil.save(image_path)

def delete_files_with_keyword(directory, keyword):
    # 获取目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否包含关键字
        if keyword in filename:
            file_path = os.path.join(directory, filename)
            # 确保路径是文件而不是目录
            if os.path.isfile(file_path):
                os.remove(file_path)
    print(f"Delete done")

def rename_files_sequentially(directory):
    # 获取目录中的所有文件，并按名称排序
    files = sorted(os.listdir(directory))
    
    # 初始化计数器
    counter = 0
    
    for filename in files:
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)
        
        # 确保路径是文件而不是目录
        if os.path.isfile(file_path):
            # 获取文件的扩展名
            file_extension = os.path.splitext(filename)[1]
            # 构造新的文件名
            new_filename = f"{counter}{file_extension}"
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(file_path, new_file_path)
            # 增加计数器
            counter += 1
    print(f"Rename done")

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    image_num = len(os.listdir(opt.dataroot))
    opt.num_test = image_num
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    dataset = create_dataset(opt)  # create a dataset
    model = create_model(opt)      # create a model
    model.setup(opt)               # regular setup

    if opt.eval:
        model.eval()

    output_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        for label, image in visuals.items():
            image_numpy = tensor_to_image(image)
            image_save_path = os.path.join(output_dir, f'{os.path.basename(img_path[0])}_{label}.png')
            save_image(image_numpy, image_save_path)
        if i % 100 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        
    directory_path = f'results/style_monet/test_{opt.epoch}'
    keyword = 'real'
    delete_files_with_keyword(directory_path, keyword)
    rename_files_sequentially(directory_path)
