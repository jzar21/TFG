import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchio as tio
import pydicom
import numpy as np
import os
import SimpleITK as sitk


class DataSetMRIs(Dataset):
    def __init__(self, mri_dir, transform=None, num_central_images=10):
        """
        Args:
            mri_dit (str): Path to the directory with the MRIs.
            transform (callable, optional): Optional transform to be applied to the sample.
        """

        self.transform = transform
        self.num_central_images = num_central_images

        self.mris_paths = []

        for patient in os.listdir(mri_dir):
            patient_path = os.path.join(mri_dir, patient)
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                for mri in os.listdir(study_path):
                    mri_path = os.path.join(study_path, mri)
                    self.mris_paths.append(mri_path)

    def __len__(self):
        """Return the number of samples"""
        return len(self.mris_paths)

    def __getitem__(self, idx):
        """Fetch a sample:

            Returns a 3D volume and the age asociated with it.
        """

        tensor, dicom_img = self.get_image_tensor(idx)
        age = self.get_age(dicom_img)
        tensor = self.get_k_central_images(tensor, self.num_central_images)

        if self.transform != None:
            tensor = self.transform(tensor)

        return tensor.unsqueeze(0), age

    def get_image_tensor(self, idx):
        mri_path = self.mris_paths[idx]

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(mri_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)

        tensor = torch.tensor(image_array, dtype=torch.float32)
        return tensor, dicom_names

    def get_age(self, dicom_imgs):
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_imgs[0])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        k = '0010|1010'
        age = reader.GetMetaData(k)
        age = float(''.join(filter(str.isdigit, age)))  # 019Y for example
        return age

    def get_k_central_images(self, tensor, num_central_images):
        num_slices = tensor.shape[0]
        if num_slices <= num_central_images:
            padding = num_central_images - num_slices

            padding_left = padding // 2
            padding_right = padding - padding_left

            tensor = torch.cat([
                torch.zeros(padding_left, *tensor.shape[1:]),
                tensor,
                torch.zeros(padding_right, *tensor.shape[1:])
            ])
            return tensor

        center = num_slices // 2
        k = num_central_images
        start = center - k // 2
        end = center + k // 2 + 1
        tensor = tensor[start:end, :, :]

        return tensor


def __test():
    data_folder = '../../datos/'

    transform = transforms.Compose([
        tio.Resize((20, 350, 350)),
    ])

    data_set = DataSetMRIs(data_folder, transform=transform)
    data_loader = DataLoader(data_set, batch_size=2)

    for im, label in data_loader:
        print(im.shape, label)


if __name__ == '__main__':
    __test()
