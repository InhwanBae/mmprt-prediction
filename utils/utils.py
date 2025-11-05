import os
import glob
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, roc_auc_score, confusion_matrix
)

# Turn off sklearn warnings
# warnings.filterwarnings("ignore")



INPUT_OUTPUT_COLS_PATH = 'config/input_output_cols.yaml'
MAP_KNOWN_COLS_PATH = 'config/map_known_cols.yaml'
GROUP_COLS_PATH = 'config/group_cols.yaml'
DATA_DIR = 'data/'

def disable_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    import logging
    logging.getLogger('shap').setLevel(logging.ERROR)


def get_input_output_cols():
    with open(INPUT_OUTPUT_COLS_PATH, 'r') as f:
        cols = yaml.safe_load(f)
    return cols.get('input_columns', []), cols.get('output_columns', [])


def get_map_known_cols():
    with open(MAP_KNOWN_COLS_PATH, 'r') as f:
        mappings = yaml.safe_load(f)
    return mappings.get('mappings', {})


def map_known_cols(df):
    # Deprecated hardcoded mappings
    # mappings = {
    #     'Group':     {'fail':0,'survival':1},
    #     'Subgroup': {'survival':0,'TKA':1,'HTO':2},
    #     'Sex':      {'F':0,'M':1},
    # }
    mappings = get_map_known_cols()

    for col, m in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(m).fillna(-1).astype(int)
    return df


def get_group_cols():
    with open(GROUP_COLS_PATH, 'r') as f:
        group_cols = yaml.safe_load(f)
    return group_cols.get('grouped_columns', {})


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def prepare_data(test_size=0.2, seed=0):
    # 1. Load all csv file from data directory
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    assert len(all_files) > 0, f'No CSV files found in data directory: {DATA_DIR}'

    df_list = []
    for file in tqdm(all_files, desc="Loading CSV files"):
        df = pd.read_csv(file).fillna(-1)
        filename = os.path.basename(file).replace('.csv', '')
        df['Source_File'] = filename
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)

    print(f'Loaded {len(all_files)} files with total {data.shape[0]} records.')
    print(f'Source files ({len(all_files)}): {", ".join([os.path.basename(f) for f in all_files])}')

    # 2. Encode text to labels
    data = map_known_cols(data)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # 3. Define input-output columns
    input_cols, output_cols = get_input_output_cols()
    print(f'Input features ({len(input_cols)}): {", ".join(input_cols)}')
    print(f'Output features ({len(output_cols)}): {", ".join(output_cols)}')

    if 'Group' in output_cols and 'Subgroup' in output_cols:
        output_cols.remove('Group')  # prioritize Subgroup if both exist
    assert all(col in data.columns for col in input_cols), 'Some input columns are missing in the data'
    assert all(col in data.columns for col in output_cols), 'Some output columns are missing in the data'
    X = data[input_cols].values
    y = data[output_cols[0]].values
    num_classes = len(np.unique(y))

    # 4. Train-test split
    if 0 < test_size < 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    elif test_size == 0:
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        raise ValueError('test_size must be between 0 and 1, or exactly 0')

    class_mapping = get_map_known_cols().get(output_cols[0], {})
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    train_set_statistics = [f'{inv_class_mapping[label]}: {count}' for label, count in zip(np.unique(y_train).tolist(), np.bincount(y_train).tolist())]
    print(f'Train set: {y_train.shape[0]} samples ({", ".join(train_set_statistics)})')
    if X_test is not None:
        test_set_statistics = [f'{inv_class_mapping[label]}: {count}' for label, count in zip(np.unique(y_test).tolist(), np.bincount(y_test).tolist())]
        print(f'Test set: {y_test.shape[0]} samples ({", ".join(test_set_statistics)})')

    # 5. Prepare output
    output = {
        'input_cols': input_cols,
        'output_cols': output_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'num_classes': num_classes,
        'class_mapping': class_mapping
    }

    return output


def compute_metrics(y_true, y_pred, y_proba, num_classes=2, mappings={}, require_softmax=False):
    group_survival_label = mappings.get('Group', {}).get('survival', 1)
    subgroup_survival_label = mappings.get('Subgroup', {}).get('survival', 0)

    metrics = {}
    if num_classes == 2:
        if not require_softmax:
            y_proba = y_proba
        else:
            y_proba = softmax(y_proba)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['Group_Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Group_Precision'] = precision_score(y_true, y_pred, pos_label=group_survival_label, zero_division=0)
        metrics['Group_Sensitivity'] = recall_score(y_true, y_pred, pos_label=group_survival_label)
        metrics['Group_Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics['Group_F1'] = f1_score(y_true, y_pred, pos_label=group_survival_label)
        metrics['Group_Brier'] = brier_score_loss(y_true, y_proba[:, group_survival_label])
        metrics['Group_ROC_AUC'] = roc_auc_score(y_true, y_proba[:, group_survival_label])

    elif num_classes > 2:
        # 1. Compute group-level metrics
        # Squeeze the y_true, y_pred and y_proba to single binary predictions and probabilities.
        # fail = 0, survival = 1
        y_true_group = (y_true == subgroup_survival_label).astype(int)
        y_pred_group = (y_pred == subgroup_survival_label).astype(int)

        if not require_softmax:
            y_proba_group = y_proba[:, subgroup_survival_label]
        else:
            temp = np.zeros((y_proba.shape[0], 2), dtype=y_proba.dtype)
            temp[:, 1] = y_proba[:, subgroup_survival_label]
            temp[:, 0] = y_proba[:, [i for i in range(num_classes) if i != subgroup_survival_label]].max(axis=1)
            temp = softmax(temp)
            y_proba_group = temp[:, 1]
            y_proba = softmax(y_proba)
        
        group_tn, group_fp, group_fn, group_tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        metrics['Group_Accuracy'] = accuracy_score(y_true_group, y_pred_group)
        metrics['Group_Precision'] = precision_score(y_true_group, y_pred_group, pos_label=1, zero_division=0)
        metrics['Group_Sensitivity'] = recall_score(y_true_group, y_pred_group, pos_label=1)
        metrics['Group_Specificity'] = group_tn / (group_tn + group_fp) if (group_tn + group_fp) > 0 else np.nan
        metrics['Group_F1'] = f1_score(y_true_group, y_pred_group, pos_label=1)
        metrics['Group_Brier'] = brier_score_loss(y_true_group, y_proba_group)
        metrics['Group_ROC_AUC'] = roc_auc_score(y_true_group, y_proba_group)

        # 2. Compute subgroup-level metrics
        metrics['Subgroup_Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Subgroup_Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['Subgroup_Recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['Subgroup_F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['Subgroup_LogLoss'] = log_loss(y_true, y_proba, labels=np.arange(num_classes))

    else:
        raise ValueError('num_classes must be 2 or greater')

    return metrics
