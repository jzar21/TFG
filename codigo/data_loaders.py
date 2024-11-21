import torch
from torch.utils.data import Dataset
import os
import pydicom


class DataSetMRIs(Dataset):
    def __init__(self, mri_dir, label_file, transform=None):
        """
        Args:
            mri_dit (str): Path to the directory with the MRIs.
            label_file (str): Path to a file that contains image paths and corresponding labels.
            transform (callable, optional): Optional transform to be applied to the sample.
        """

        self.label_file = label_file
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
        mri = self.mris_paths[idx]

        full_mri_img = []
        age = -1
        for dicom in mri:
            dcm = pydicom.dcmread(dicom)
            full_mri_img.append(torch.tensor(dcm.pixel_array))
            age = dcm.PatientAge

        tensor_mri = torch.stack(full_mri_img)

        if self.transform != None:
            tensor_mri = self.transform(tensor_mri)

        return tensor_mri, age
