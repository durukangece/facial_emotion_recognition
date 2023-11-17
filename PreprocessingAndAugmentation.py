import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2
import numpy as np

class Augmentation:
    def __init__(self):
        self.augmentation_transforms = A.Compose([ # Compose sınıfı, bir dizi görüntü dönüşümünü birleştirmek ve bunları belirli bir sırayla veya rastgele uygulamak için kullanılır. 
                                                   # Bu sınıf, Albumentations kütüphanesinin bir parçasıdır ve bu kütüphane, görüntü artırma işlemleri için bir dizi önceden tanımlanmış dönüşüm ve işlev içerir.
            A.RandomRotate90(), # Bu dönüşüm, görüntüyü rastgele 90, 180 veya 270 derece çevirir. 
            A.HorizontalFlip(p=0.5), #görüntüyü simetriğine çevirir.p=0.5 olduğunda, her bir görüntünün yatay çevirilme olasılığı %50'dir. 
            A.GaussianBlur(p=0.3), #Modelin gerçek dünya koşullarına daha iyi uyarlanmasını sağlayamak için average ve medyan yerine gaussian blurring kullanılmış.
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2), # Renk tonu, doygunluk ve parlaklık değişiklikleri uygulayarak görüntü artırma sağlar. 
        ])                                                                                           # limit=10 olması ise, alacağı değeri -10 ile 10 arasında sınırlar.

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Giriş olarak alınan görüntüyü (img) RGB renk uzayından BGR'ye dönüştürür. Albumentations kütüphanesi, GBR renk uzayını bekler, bu nedenle dönüşüm yapılmış olmalıdır.
        augmented = self.augmentation_transforms(image=img) # görüntü üzerine belirtilen dönüşümleri uygular ve sonucu augmented değişkenine atar. Bu işlem sonunda, augmented değişkeni, 
                                                            # belirtilen görüntü üzerinde uygulanan artırma dönüşümlerinin sonucu olan bir sözlüğü temsil eder. Bu sözlük içinde, özellikle 'image' anahtarı, artırılmış görüntüyü içerir.
        return augmented['image']  #Artırılmış görüntüyü döndürür. Artırılmış görüntü, augmented sözlüğünün içindeki 'image' anahtarından alınır.