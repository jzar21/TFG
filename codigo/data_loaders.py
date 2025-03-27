import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchio as tio
import pydicom
import numpy as np
import os
import SimpleITK as sitk
import cv2


class DataSetMRIs(Dataset):
    def __init__(self, mri_dir, transform=None, num_central_images=10, it=[5]*5):
        """
        Args:
            mri_dit (str): Path to the directory with the MRIs.
            transform (callable, optional): Optional transform to be applied to the sample.
        """

        self.transform = transform
        self.num_central_images = num_central_images

        self.mris_paths = []
        self.it = it

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
        # tensor = self.apply_bias_field_corrector(tensor)

        if self.transform != None:
            tensor = self.transform(tensor)

        tensor = self.apply_otsu_thresholding(tensor)
        tensor = tensor.unsqueeze(0)

        return tensor, torch.tensor(age)

    def apply_bias_field_corrector(self, tensor):
        corrector_bias = sitk.N4BiasFieldCorrectionImageFilter()
        corrector_bias.SetMaximumNumberOfIterations(self.it)
        corrector_bias.SetNumberOfThreads(30)

        for i in range(tensor.shape[0]):
            slice_2d = tensor[i, :, :].numpy()

            img_sitk = sitk.GetImageFromArray(slice_2d)
            img_bfc = corrector_bias.Execute(img_sitk)
            img_bfc = sitk.GetArrayFromImage(img_bfc)

            tensor[i, :, :] = torch.tensor(img_bfc, dtype=torch.float32)

        return tensor

    def apply_otsu_thresholding(self, tensor):
        """
        Apply Otsu's thresholding to each 2D slice of the 3D tensor.
        """
        for i in range(tensor.shape[0]):
            slice_2d = tensor[i, :, :].numpy()

            min_val = slice_2d.min()
            max_val = slice_2d.max()

            denominator = max_val - min_val

            if np.isclose(denominator, 0.0):
                denominator = 1

            rescaled = 255 * (slice_2d - min_val) / denominator
            rescaled = np.uint8(rescaled)

            ret, otsu_thresholded = cv2.threshold(
                rescaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            tensor[i, :, :] = torch.tensor(
                otsu_thresholded, dtype=torch.float32)

        return tensor

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


class DataSetMRIClassification(DataSetMRIs):
    def __init__(self, mri_dir, transform=None, num_central_images=10, it=[5] * 5):
        super().__init__(mri_dir, transform, num_central_images, it)

    def __getitem__(self, idx):
        tensor, age = super().__getitem__(idx)
        older_than_18 = torch.tensor(
            0, dtype=torch.float32) if age >= 18 else torch.tensor(1, dtype=torch.float32)

        return tensor, older_than_18


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
