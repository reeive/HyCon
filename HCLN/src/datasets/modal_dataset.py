import torch
from torch.utils.data import Dataset
import os
import numpy as np
import logging
from torchvision import transforms

DEFAULT_RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiModalDataset(Dataset):
    def __init__(self, data_dir: str,
                 mode: str = 'train',
                 mode1_folder_suffix: str = 'baseline',
                 mode2_folder_suffix: str = 'followup',
                 mask_folder_name: str = 'mask',
                 sample_list_file: str = 'slice_nidus_all.list',
                 image_load_rate: float = 1.0,
                 transform: transforms.Compose = None,
                 load_mask: bool = True,
                 resample_to_3_channels: bool = True,
                 verbose: bool = True):

        self._data_dir = data_dir
        self.sample_ids = []
        self.mode = mode
        self.mode1_folder_suffix = mode1_folder_suffix
        self.mode2_folder_suffix = mode2_folder_suffix
        self.mask_folder_name = mask_folder_name
        self.transform = transform if transform is not None else DEFAULT_RESNET_TRANSFORM
        self.load_mask_flag = load_mask
        self.resample_to_3_channels = resample_to_3_channels

        list_path = os.path.join(self._data_dir, sample_list_file)

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Sample list file not found: {list_path}")
        if not os.path.isdir(self._data_dir):
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")

        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if line:
                    self.sample_ids.append(line)
        
        if verbose:
            logging.info(f'Raw {self.mode} dataset: Found {len(self.sample_ids)} sample IDs in {list_path}')

        if image_load_rate < 1.0 and "train" in self.mode:
            num_to_load = int(len(self.sample_ids) * image_load_rate)
            self.sample_ids = self.sample_ids[:num_to_load]
            if verbose:
                logging.info(f"Applied image_load_rate ({image_load_rate}): now using {len(self.sample_ids)} samples for {self.mode}.")
        if verbose:
            logging.info(f"Finalizing {self.mode} dataset with {len(self.sample_ids)} samples.")

    def __len__(self):
        return len(self.sample_ids)

    def _prepare_image_data(self, numpy_data: np.ndarray) -> np.ndarray:
        if numpy_data is None:
            return None
        
        if numpy_data.ndim == 2:
            numpy_data = np.expand_dims(numpy_data, axis=0)
        elif numpy_data.ndim == 3 and numpy_data.shape[0] > 3 and numpy_data.shape[2] <= 4:
             numpy_data = np.transpose(numpy_data, (2, 0, 1))
        
        if self.resample_to_3_channels:
            if numpy_data.shape[0] == 1:
                numpy_data = np.repeat(numpy_data, 3, axis=0)
            elif numpy_data.shape[0] != 3:
                logging.warning(f"Image has {numpy_data.shape[0]} channels, cannot resample to 3.")

        if np.issubdtype(numpy_data.dtype, np.floating):
            min_val, max_val = np.min(numpy_data), np.max(numpy_data)
            if max_val > min_val:
                if max_val > 1.0 or min_val < 0.0:
                    numpy_data = (numpy_data - min_val) / (max_val - min_val)
        elif not np.issubdtype(numpy_data.dtype, np.uint8):
             if np.max(numpy_data) > 255 or np.min(numpy_data) < 0:
                 min_val, max_val = np.min(numpy_data), np.max(numpy_data)
                 if max_val > min_val:
                     numpy_data = ((numpy_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                 else:
                     numpy_data = np.zeros_like(numpy_data, dtype=np.uint8)
             else:
                 numpy_data = numpy_data.astype(np.uint8)
        return numpy_data

    def _load_npy_robust(self, folder_path_suffix: str, sample_id: str) -> np.ndarray:
        full_path = os.path.join(self._data_dir, f'imgs_{folder_path_suffix}', f'{sample_id}.npy')
        if not os.path.exists(full_path):
            return None
        try:
            return np.load(full_path)
        except Exception as e:
            logging.error(f"Error loading {full_path}: {e}. Returning None.")
            return None

    def _load_mask_npy_robust(self, sample_id: str) -> np.ndarray:
        full_path = os.path.join(self._data_dir, self.mask_folder_name, f'{sample_id}.npy')
        if not os.path.exists(full_path):
            return None
        try:
            mask_data = np.load(full_path)
            if mask_data.ndim == 2:
                mask_data = np.expand_dims(mask_data, axis=0)
            elif mask_data.ndim == 3 and mask_data.shape[0] != 1:
                mask_data = mask_data[0:1, :, :]
            return mask_data.astype(np.int64)
        except Exception as e:
            logging.error(f"Error loading mask {full_path}: {e}. Returning None.")
            return None

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        is_valid_sample = True

        raw_mode1 = self._load_npy_robust(self.mode1_folder_suffix, sample_id)
        raw_mode2 = self._load_npy_robust(self.mode2_folder_suffix, sample_id)

        if raw_mode1 is None or raw_mode2 is None:
            is_valid_sample = False
            dummy_img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            tensor_mode1 = dummy_img_tensor
            tensor_mode2 = dummy_img_tensor
        else:
            prepared_mode1 = self._prepare_image_data(raw_mode1)
            prepared_mode2 = self._prepare_image_data(raw_mode2)
            
            if prepared_mode1 is None or prepared_mode2 is None:
                is_valid_sample = False
                dummy_img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                tensor_mode1 = dummy_img_tensor
                tensor_mode2 = dummy_img_tensor
            else:
                tensor_mode1 = self.transform(prepared_mode1)
                tensor_mode2 = self.transform(prepared_mode2)

        sample_dict = {'id': sample_id, 'mode1': tensor_mode1, 'mode2': tensor_mode2}

        if self.load_mask_flag:
            raw_mask = self._load_mask_npy_robust(sample_id)
            if raw_mask is None:
                dummy_mask_tensor = torch.zeros((1, tensor_mode1.shape[1], tensor_mode1.shape[2]), dtype=torch.long)
                sample_dict['mask'] = dummy_mask_tensor
                if self.mode != "train_unlabeled":
                    is_valid_sample = False
            else:
                sample_dict['mask'] = torch.from_numpy(raw_mask)

        sample_dict['is_valid'] = is_valid_sample
        return sample_dict