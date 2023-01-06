
import cv2
import numpy as np
import random as r
import os
import torch
import torch.utils.data as data

class Crack_Datasets(data.Dataset):
    def __init__(self, data_root, img_list, img_size, mode='train') -> None:
        super().__init__()

        self.data_root = data_root
        self.img_list = img_list
        self.img_size = img_size
        self.mode = mode
        with open(img_list) as f:
            self.imgID = f.readlines()
        self.num = len(self.imgID)

    def __getitem__(self, index) :
        image,label = read_files(self.data_root, self.imgID[index].strip(), self.img_size, mode=self.mode)


        # augmentation
        if self.mode == 'train':
            image,label = random_scale_and_creat_patch(image,label,self.img_size[0],self.img_size[1])
            image,label = random_flip(image,label)
        else:
            label = np.expand_dims(label,axis=-1)

        # normalization
        image = (image.astype(np.float32) - (114., 121., 134.)) / 255.0
        label = label.astype(np.float32)

        label[label == 0] = 0
        label[label > 0] = 1

        # To tensor
        image = np2Tensor(image)
        label = np2Tensor(label)

        label = label[0,:,:].unsqueeze_(0)

        sample = {'image':image,'label':label}
        sample['name'] = self.imgID[index].strip()

        return sample
    
    def __len__(self):
        return self.num


def read_files(data_root,file_name,img_size,mode):
    image_name = os.path.join(data_root,'images', file_name + '.jpg')
    label_name = os.path.join(data_root,'masks', file_name + '.png')
    if mode == 'train':
        image = cv_imread(image_name)
        label = cv_imread(label_name)
    else:
        image = cv_imread(image_name)
        image = cv2.resize(image, (img_size[0],img_size[1]), interpolation=cv2.INTER_CUBIC)
        label = cv_imread(label_name)
        label = cv2.resize(label, (img_size[0],img_size[1]),interpolation=cv2.INTER_NEAREST)

    return image, label


def cv_imread(file_path):
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    return img


def np2Tensor(array):
    tensor = torch.FloatTensor(array.transpose(2,0,1).astype(float))
    return tensor


def random_scale_and_creat_patch(image, label, img_size_w, img_size_h):
    # random scale
    if r.random() < 0.5:
        h, w, c = image.shape
        scale = 0.75 + 0.5 * r.random()
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    # creat patch
    if r.random() < 0.5:
        h, w, c = image.shape
        if h > img_size_h and w > img_size_w:
            x = r.randrange(0, w - img_size_w)
            y = r.randrange(0, h - img_size_h)
            image = image[y:y + img_size_h, x:x + img_size_w, :]
            label = np.expand_dims(label, axis=-1)
            label = label[y:y + img_size_h, x:x + img_size_w, :]
        else:
            image = cv2.resize(image, (img_size_w, img_size_h), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (img_size_w, img_size_h), interpolation=cv2.INTER_NEAREST)
            label = np.expand_dims(label, axis=-1)
    else:
        image = cv2.resize(image, (img_size_w, img_size_h), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (img_size_w, img_size_h), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, axis=-1)

    return image, label


def random_flip(image, label):
    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
        label = np.expand_dims(label, axis=-1)

    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        label = np.expand_dims(label, axis=-1)
    return image, label