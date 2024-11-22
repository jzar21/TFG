import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchio as tio
import pydicom
import numpy as np
import os


class DataSetMRIs(Dataset):
    def __init__(self, mri_dir, transform=None):
        """
        Args:
            mri_dit (str): Path to the directory with the MRIs.
            transform (callable, optional): Optional transform to be applied to the sample.
        """

        self.transform = transform

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
        mri_path = self.mris_paths[idx]

        full_mri_img = []
        age = -1
        for dicom in os.listdir(mri_path):
            dicom_path = os.path.join(mri_path, dicom)
            dcm = pydicom.dcmread(dicom_path)
            full_mri_img.append(np.array(dcm.pixel_array))
            age = dcm.PatientAge

        tensor_mri = np.stack(full_mri_img)
        tensor_mri = np.expand_dims(tensor_mri, axis=0)

        if self.transform != None:
            tensor_mri = self.transform(tensor_mri)

        tensor_mri = torch.tensor(tensor_mri, dtype=torch.float32)

        age = float(''.join(filter(str.isdigit, age)))  # 019Y for example

        return tensor_mri, torch.tensor(age)


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
