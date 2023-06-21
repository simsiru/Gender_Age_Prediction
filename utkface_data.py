import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class UTKFaceDataset(Dataset):
    def __init__(self, utkface_df, img_dir='', transform=None, 
                 target_transform=None):
        self.utkface_df = utkface_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.utkface_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 
                                self.utkface_df.iloc[idx, 0])
        image = read_image(img_path)
        
        age_label = self.utkface_df.iloc[idx, 1]
        gender_label = self.utkface_df.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            age_label = self.target_transform(age_label)
            gender_label = self.target_transform(gender_label)
            
        return {'image':image, 'age': age_label, 'gender': gender_label}