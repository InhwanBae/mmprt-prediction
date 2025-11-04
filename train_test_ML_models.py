import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm.auto import tqdm

from utils.utils import prepare_data, compute_metrics, disable_warnings


current_time = time.strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = f'results/ml_results_{current_time}.csv'


def main(args):
    # 0. Reproducibility
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    disable_warnings()

    # 1. Prepare data
    data = prepare_data(test_size=args.test_size)
    input_cols = data['input_cols']
    output_cols = data['output_cols']
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    num_classes = data['num_classes']
    class_mapping = data['class_mapping']
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # 2. Define ML models
    models = {
        'ElasticNetLR': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5, random_state=args.seed, n_jobs=-1),
        'MLP': MLPClassifier(max_iter=1000, random_state=args.seed),
        'SVM': SVC(probability=True, random_state=args.seed),
        'RandomForest': RandomForestClassifier(random_state=args.seed, n_estimators=2, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=args.seed, n_estimators=10, max_depth=3, n_jobs=-1),
    }

    # 3. Repeated K-Fold Cross-Validation
    rkf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed)
    pbar = tqdm(total=len(models)*rkf.get_n_splits(X_train, y_train), desc='Training & Validating Models')
    print('\nTraining & Validating Models...')

    summary_val = []
    for name, model in models.items():
        pbar.set_description(f'Training')
        pbar.set_postfix({'Model': name})
        scores = {}
        for tr, val in rkf.split(X_train, y_train):
            model.fit(X_train[tr], y_train[tr])
            yp = model.predict(X_train[val])
            yp_p = model.predict_proba(X_train[val])
            metrics = compute_metrics(y_train[val], yp, yp_p, num_classes)
            for m, v in metrics.items():
                scores.setdefault(m, []).append(v)
            pbar.update()

        pbar.set_description(f'Validating')
        row = {'Model': name}
        for m in metrics:
            arr = np.array(scores[m])
            mu, se = arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr))
            h = 1.96 * se
            row[m] = f"{mu:.4f} ({mu-h:.4f}–{mu+h:.4f})"
        summary_val.append(row)
    pbar.close()

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig') as f:
        f.write('=== ML Results (Train) ===\n')  # create or clear the file
        print("\n=== ML Results (Train) ===")

    df_train = pd.DataFrame(summary_val).set_index('Model')
    df_train_split = [df_train.filter(like=col) for col in ['Group', 'Subgroup']]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None, 'display.width', None):  # print DataFrame without truncation
        for df in df_train_split:
            df.to_csv(OUTPUT_FILE, mode='a', encoding='utf-8-sig')  # Append pandas DataFrame to CSV
            print(df)

    # 4. Evaluate on test set if available
    if X_test is not None:
        print('\nEvaluating Models on Test Set...')
        pbar = tqdm(total=len(models), desc='Evaluating')
        summary_test = []
        for name, model in models.items():
            pbar.set_postfix({'Model': name})
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            yp_p = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, yp, yp_p, num_classes)

            row = {'Model': name}
            for m, v in metrics.items():
                row[m] = f"{v:.4f}"
            summary_test.append(row)
            pbar.update()
        pbar.close()

        with open(OUTPUT_FILE, 'a', encoding='utf-8-sig') as f:
            f.write('\n=== ML Results (Test) ===\n')  # Append to the file
            print("\n=== ML Results (Test) ===")

        df_test = pd.DataFrame(summary_test).set_index('Model')
        df_test_split = [df_test.filter(like=col) for col in ['Group', 'Subgroup']]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None, 'display.width', None):  # print DataFrame without truncation
            for df in df_test_split:
                df.to_csv(OUTPUT_FILE, mode='a', encoding='utf-8-sig')  # Append pandas DataFrame to CSV
                print(df)
    else:
        print("\n=== No test set evaluation (test_size=0) ===")

    print(f"\nResults saved to '{OUTPUT_FILE}'.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate ML models.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test set size')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--n_repeats', type=int, default=3, help='Number of repeats for numerical stability')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args, _ = parser.parse_known_args()
    main(args)
