import os
from PIL import Image
from datasets.seafood_det import SeafoodDET
import datasets.transforms.functional as F


class SeafoodDETTest(SeafoodDET):
    def __init__(self, root_dir, split, mean, std):
        super(SeafoodDETTest, self).__init__(root_dir, split, None)
        self.mean = mean
        self.std = std
        print("=> Reading {} Test Image...".format(len(self.mdf)))
        self.images = self.read_images()
        print("=> Done!")

    def read_images(self):
        images = []
        for name in self.mdf:
            img_name = os.path.join(self.images_dir, '{}.jpg'.format(name))
            image = Image.open(img_name).convert("RGB")
            w, h = image.size
            resize_height, resize_width = 512, 512
            h_ratio, w_ratio = h/resize_height, w/resize_width
            image = F.resize_img(image, (resize_height, resize_width))
            image = F.img_to_tensor(image)
            image = F.normalize(image, mean=self.mean, std=self.std)
            images.append((image, name, (h_ratio, w_ratio)))
        return images

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        return self.images[item]
