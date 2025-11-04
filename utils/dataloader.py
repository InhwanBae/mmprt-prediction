
import numpy as np
import torch
from torch.utils.data import Dataset


class MMPRTDataset(Dataset):
    def __init__(self, X, y, feature_cols, augment=False):
        """
        Args:
            X (np.array): Feature array.
            y (np.array): Target array.
            feature_cols (list): Names of feature columns.
            augment (bool): Whether to apply augmentation.
        """
        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.augment = augment

        # feature_cols = ['Age', 'Sex', 'Height', 'Weight', 'BMI', 'Sx_duration', 'Chronicity', 'Injury_mechanism', 'Popping', 'Givingway', 'Pre_mHKA', 'Pre_MPTA', 'Pre_JLCA', 'Pre_slope', 'Pre_KL', 'MJW_extension', 'MJW_flexion', 'ICRS', 'Shinycorner', 'MME_absolute', 'MME_relative', 'Lat_PTS_MRI', 'Med_PTS_MRI', 'Effusion', 'BME', 'SIFK']
        # We don't need to augment ['Sex', 'Pre_KL', 'ICRS', 'Shinycorner', 'Effusion', 'BME', SIFK']
        self.perturbations = {
            'Age': (-1, 1),
            'Height': (-1, 1),
            'Weight': (-1, 1),
            'BMI': (-0.125, 0.125),
            'Sx_duration': (-0.25, 0.25),
            'Pre_mHKA': (-0.025, 0.025),
            'Pre_MPTA': (-0.025, 0.025),
            'Pre_LDFA': (-0.025, 0.025),
            'Pre_JLCA': (-0.025, 0.025),
            'Pre_slope': (-0.025, 0.025),
            'MJW_extension': (-0.025, 0.025),
            'MJW_flexion': (-0.025, 0.025),
            'MME_absolute': (-0.025, 0.025),
            'MME_relative': (-0.25, 0.25),
            'Lat_PTS_MRI': (-0.025, 0.025),
            'Med_PTS_MRI': (-0.025, 0.025),
        }
        
        self.mask_out_cols_set = [['Height', 'Weight', 'BMI'],  # 2,3,4
                                  ['Injury_mechanism', 'Popping', 'Givingway'],  # 7,8,9
                                  ['Popping', 'Givingway'],  # 8,9
                                  ['Givingway'],  # 9
                                  ['MJW_flexion'] # 16
                                 ]

        # Map columns to indices
        self.perturb_idx = {self.feature_cols.index(col): rng 
                            for col, rng in self.perturbations.items() if col in self.feature_cols}
        self.mask_out_cols_set_idx = [[self.feature_cols.index(col) for col in mask_out_cols if col in self.feature_cols]
                                      for mask_out_cols in self.mask_out_cols_set]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.augment:
            for idx_col, (low, high) in self.perturb_idx.items():
                # perturb = np.random.uniform(low, high)
                perturb = np.random.normal(loc=0.0, scale=(high - low))
                if x[idx_col] != -1:
                    x[idx_col] += perturb

            # Randomly mask out some features (5% for each)
            for mask_out_cols in self.mask_out_cols_set_idx:
                if np.random.rand() < 0.05:
                    for col_idx in mask_out_cols:
                        x[col_idx] = -1

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
