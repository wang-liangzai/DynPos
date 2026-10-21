#
# import pickle
# import os
# from PIL import Image
# from torchvision import transforms
# import numpy as np
# from tqdm import tqdm
#
# def load_file(filename):
#     with open(filename, 'rb') as filehandle:
#         ret = pickle.load(filehandle)
#         return ret
# if __name__ == "__main__":
#
#     opt = {
#         'data_path': 'input/prepared_clean/',
#         'img_path': 'C:/Users/Admin/Desktop/工作文件/第六篇论文/第六篇论文/第六篇论文/DynRT-main/dataset_image/'
#     }
#     id ={
#         "train": load_file(opt["data_path"] + "train_id"),
#         "test": load_file(opt["data_path"] + "test_id"),
#         "valid": load_file(opt["data_path"] + "valid_id")
#     }
#     img_dir = opt["img_path"]
#
#     transform_train = transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     transform_valid = transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     transform = {
#         "train": transform_train,
#         "valid": transform_valid,
#         "test": transform_test
#         }
#     image_tensor = {
#         "train": [],
#         "valid": [],
#         "test": []
#     }
#     save_path = 'image_tensor/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     for mode in id.keys():
#         for idx in id[mode]:
#             img_path = os.path.join(
#                     img_dir,
#                     "{}.jpg".format(idx)
#                 )
#             img = Image.open(img_path)
#             img = img.convert('RGB')  # convert grey picture
#             trainsform_img = transform[mode](img)
#             image_tensor[mode].append(trainsform_img.unsqueeze(0))
#             np.save(save_path + str(idx) + '.npy', trainsform_img.numpy())
#
#
#
#
import pickle
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret


if __name__ == "__main__":

    opt = {
        'data_path': 'input/prepared_clean/',
        'img_path': r'C:/Users/Admin/Desktop/工作文件/第六篇论文/第六篇论文/第六篇论文/DynRT-main/dataset_image/'
    }

    id = {
        "train": load_file(opt["data_path"] + "train_id"),
        "test": load_file(opt["data_path"] + "test_id"),
        "valid": load_file(opt["data_path"] + "valid_id")
    }

    img_dir = opt["img_path"]

    transform_base = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    transform = {
        "train": transform_base,
        "valid": transform_base,
        "test": transform_base
    }

    save_path = 'image_tensor/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始处理
    for mode in id.keys():

        print(f"\n开始处理 {mode} 数据...")

        for idx in tqdm(
                id[mode],
                desc=f"{mode}",
                ncols=100):

            try:
                img_path = os.path.join(
                    img_dir,
                    f"{idx}.jpg"
                )

                img = Image.open(img_path)
                img = img.convert('RGB')

                transform_img = transform[mode](img)

                np.save(
                    os.path.join(save_path, f"{idx}.npy"),
                    transform_img.numpy()
                )

            except Exception as e:
                print(f"\n跳过异常图片: {idx}")
                print("错误:", e)
                continue

    print("\n全部处理完成！")