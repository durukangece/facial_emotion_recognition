from PreprocessingAndAugmentation import Augmentation
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2

class FerDataset(Dataset):            # 'FerDataset' sınıfı, PyTorch'un Dataset sınıfından türetilmiştir. Bu sınıf, bir veri kümesini temsil eder ve veri yükleme işlemlerini gerçekleştirir.
    def __init__(self, data_dir, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.data_list = []
        self.label_mapping = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6
        }
        
        # Bu kod bloğu, veri setinin oluşturulması için belirtilen dizin yapısını kullanarak, 
        # her bir görüntü dosyasının yolunu ve etiketini data_list listesine ekler. 
        # Bu listede her bir örnek, bir görüntü dosyasının yoluyla ona karşılık gelen bir etiketle temsil edilir.
        for root, dirs, files in os.walk(self.data_dir):
            for directory in dirs:
                for file in os.listdir(os.path.join(root, directory)):
                    if file.endswith(".jpg"):
                        img_path = os.path.join(root, directory, file)
                        label = self.label_mapping[directory]
                        self.data_list.append((img_path, label))
        
        if self.augment:
            self.preprocess_augment = Augmentation()

    def __len__(self):                    # Bu metot, veri kümesinin uzunluğunu döndürür, yani içerdiği örnek sayısını temsil eder.
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.augment:
            img_np = np.array(img)
            img_np = self.preprocess_augment.apply(img_np) # Augmentasyon
            img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        if self.transform:
            img = self.transform(img)

        return img, label