import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import RepeatedStratifiedKFold

from tqdm.auto import tqdm
import shap
import matplotlib.pyplot as plt
import os
import pickle

from utils.utils import prepare_data, compute_metrics, get_group_cols, disable_warnings
from utils.dataloader import MMPRTDataset
from utils.model import GroupedGraphAttentionNet



current_time = time.strftime('%Y%m%d_%H%M%S')
OUTPUT_FOLDER = f'results/dl_results_{current_time}/'
OUTPUT_FILE = f'results/dl_results_{current_time}.csv'


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def eval_on_loader(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            # temperature = 1.0
            # prob = nn.functional.softmax(logits / temperature, dim=1).cpu().numpy()  # will be done in compute_metrics
            prob = logits.detach().cpu().numpy()
            pred  = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy()); ps.append(pred); probs.append(prob)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_proba= np.vstack(probs)
    return y_true, y_pred, y_proba


def main(args):
    # 0. Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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
    group_cols = get_group_cols()

    # 2. Define DL model
    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = os.path.join(OUTPUT_FOLDER, f'best_model.pt')

    # fresh model & optimizer
    model = GroupedGraphAttentionNet(input_dim=X_train.shape[1],
                    hidden_layer_sizes=tuple(args.hidden),
                    output_dim=num_classes,
                    attention_method='GAT',
                    feature_cols=input_cols,
                    group_cols=group_cols).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # SHAP value storage
    all_shap_values = []
    all_X_explain = []
    all_mean_abs_shap_by_class = [[] for _ in range(num_classes)]

    # 3. Repeated K-Fold Cross-Validation
    rkf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed)
    pbar = tqdm(total=rkf.get_n_splits(X_train, y_train) * args.epochs, desc='Training & Validating Model', unit='epoch')
    print('\nTraining & Validating Models...')

    scores = {}
    for tr_idx, val_idx in rkf.split(X_train, y_train):
        fold_id = len(next(iter(scores.values()), [])) + 1
        # Build loaders
        train_dataset = MMPRTDataset(X_train[tr_idx], y_train[tr_idx], input_cols, augment=True)
        valid_dataset = MMPRTDataset(X_train[val_idx], y_train[val_idx], input_cols, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        # Update normalization parameters
        if hasattr(model, 'normalize'):
            X_train_tensor = torch.tensor(X_train[tr_idx], dtype=torch.float32).to(device)
            model.update_norm_mean_std(X_train_tensor)

        # Training loop
        best_acc = 0.0
        best_metrics = {}
        pbar.set_description(f"Training on fold {fold_id}")
        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            pbar.update()

            # eval on last epoch or every eval_every epochs
            if (epoch == args.epochs - 1) or (args.eval_every > 0 and (epoch + 1) % args.eval_every == 0 and epoch > args.epochs // 2):
                yt, yp, yp_p = eval_on_loader(model, valid_loader, device)
                metrics = compute_metrics(yt, yp, yp_p, num_classes, mappings=class_mapping, require_softmax=True)
                acc = (metrics['Subgroup_F1'] + metrics['Subgroup_Accuracy']) if 'Subgroup_F1' in metrics else (metrics['Group_F1'] + metrics['Group_Accuracy'])
                if acc > best_acc:
                    best_acc = acc
                    best_metrics = metrics
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, best_model_path)
        
        # After fold ends, reload best model
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(best_model_path)['optimizer_state_dict'])

        # Store metrics
        for k, v in best_metrics.items():
            scores.setdefault(k, []).append(v)


        ####################
        #   SHAP analysis  #
        ####################

        model.eval()
        shap_dir = os.path.join(OUTPUT_FOLDER, 'shap_analysis')
        os.makedirs(shap_dir, exist_ok=True)

        # Background & explanation subset (Truncated for speed)
        background = X_train[tr_idx][np.random.choice(len(tr_idx), size=min(1000, len(tr_idx)), replace=False)]
        X_explain = X_train[val_idx][np.random.choice(len(val_idx), size=min(200, len(val_idx)), replace=False)]

        # Calculate SHAP values and plot
        def model_forward_class(i):
            def wrapped(x_numpy):
                x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = nn.functional.softmax(logits, dim=1)[:, i].cpu().numpy()
                return probs  # shape: (n_samples,)
            return wrapped
        
        for class_idx in range(num_classes):
            pbar.set_description(f"Training on fold {fold_id} - SHAP analysis of class {class_idx}")
            pbar.refresh()
            explainer = shap.KernelExplainer(model_forward_class(class_idx), background)
            shap_vals = explainer.shap_values(X_explain, nsamples=100, silent=True)
            shap_vals = np.array(shap_vals)  # shape: (n_samples, n_features)
            
            all_shap_values.append(shap_vals)
            all_X_explain.append(X_explain)

            # Mean abs SHAP bar plot
            mean_abs = np.abs(shap_vals).mean(axis=0)
            all_mean_abs_shap_by_class[class_idx].append(mean_abs)

    pbar.close()

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig') as f:
        f.write('=== DL Results (Train) ===\n')  # create or clear the file
        print("\n=== DL Results (Train) ===")

    summary_val = []
    row = {'Model': model.__class__.__name__}
    for metric, vals in scores.items():
        arr = np.array(vals)
        mu, se = arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr))
        h = 1.96 * se
        row[metric] = f"{mu:.4f} ({mu-h:.4f}–{mu+h:.4f})"
    summary_val.append(row)

    df_train = pd.DataFrame(summary_val).set_index('Model')
    df_train_split = [df_train.filter(like=col) for col in ['Group', 'Subgroup']]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None, 'display.width', None):  # print DataFrame without truncation
        for df in df_train_split:
            df.to_csv(OUTPUT_FILE, mode='a', encoding='utf-8-sig')  # Append pandas DataFrame to CSV
            print(df)

    # 4. Evaluate on test set if available
    if X_test is not None:
        # fresh model & optimizer
        model = GroupedGraphAttentionNet(input_dim=X_train.shape[1],
                        hidden_layer_sizes=tuple(args.hidden),
                        output_dim=num_classes,
                        attention_method='GAT',
                        feature_cols=input_cols,
                        group_cols=group_cols).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print('\nEvaluating Models on Test Set...')
        pbar = tqdm(total=args.epochs * 10, desc='Evaluating Model')
        train_dataset = MMPRTDataset(X_train, y_train, input_cols, augment=True)
        test_dataset = MMPRTDataset(X_test, y_test, input_cols, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Update normalization parameters with full training set
        if hasattr(model, 'normalize'):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            model.set_norm_mean_std(X_train_tensor)

        best_acc = 0.0
        best_metrics = {}
        pbar.set_description(f"Evaluating")
        for epoch in range(args.epochs * 10):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            pbar.update()

            # eval on last epoch or every eval_every epochs
            if (epoch == args.epochs - 1) or (args.eval_every > 0 and (epoch + 1) % args.eval_every == 0 and epoch > args.epochs // 2):
                yt, yp, yp_p = eval_on_loader(model, test_loader, device)
                metrics = compute_metrics(yt, yp, yp_p, num_classes, mappings=class_mapping, require_softmax=True)
                acc = (metrics['Subgroup_F1'] + metrics['Subgroup_Accuracy']) if 'Subgroup_F1' in metrics else (metrics['Group_F1'] + metrics['Group_Accuracy'])
                if acc > best_acc:
                    best_acc = acc
                    best_metrics = metrics
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, best_model_path)
        pbar.close()
        
        # After fold ends, reload best model
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(best_model_path)['optimizer_state_dict'])

        summary_test = []
        row = {'Model': model.__class__.__name__}
        for m, v in best_metrics.items():
            row[m] = f"{v:.4f}"
        summary_test.append(row)
        
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


    # 5. Draw SHAP summary plot
    print("\n=== SHAP Analysis ===")
    # Reshape to (num_classes, n_samples, n_features)
    per_class_shap_values = [np.concatenate([fold_shap_values for idx, fold_shap_values in enumerate(all_shap_values) if idx % num_classes == class_idx], axis=0) for class_idx in range(num_classes)]
    per_class_X_explain = [np.concatenate([fold_X_explain for idx, fold_X_explain in enumerate(all_X_explain) if idx % num_classes == class_idx], axis=0) for class_idx in range(num_classes)]

    for class_idx in range(num_classes):
        class_name = inv_class_mapping.get(class_idx, str(class_idx))
        plt.figure(figsize=(10, 10))
        shap.summary_plot(per_class_shap_values[class_idx], 
                          per_class_X_explain[class_idx], 
                          feature_names=input_cols, 
                          show=False, 
                          rng=np.random.default_rng(0), 
                          max_display=len(input_cols))
        plt.title(f"SHAP Beeswarm (Class {class_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"shap_beeswarm_class{class_name}.png"), dpi=300)
        plt.savefig(os.path.join(shap_dir, f"shap_beeswarm_class{class_name}.svg"))
        plt.clf()

    # Draw mean absolute SHAP values plot
    all_mean_abs_shap = np.concatenate(all_mean_abs_shap_by_class, axis=0).mean(axis=0)  # shape: (num_features,)
    all_mean_abs_shap_by_class = np.array(all_mean_abs_shap_by_class).mean(axis=1)  # shape: (num_classes, n_features)
    all_cumulative_mean_abs_shap_by_class = all_mean_abs_shap_by_class.cumsum(axis=0) / 3

    sorted_idx = np.argsort(all_mean_abs_shap)[::-1]
    plt.figure(figsize=(10, 10))
    plt.barh(np.array(input_cols)[sorted_idx], all_mean_abs_shap[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title("Mean Absolute SHAP Values by Class (All Folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"mean_abs_shap.png"), dpi=300)
    plt.savefig(os.path.join(shap_dir, f"mean_abs_shap.svg"))
    plt.clf()

    plt.figure(figsize=(10, 10))
    for class_idx, class_shap in zip(list(range(num_classes))[::-1], all_cumulative_mean_abs_shap_by_class[:, sorted_idx][::-1]):
        class_name = inv_class_mapping.get(class_idx, str(class_idx))
        plt.barh(np.array(input_cols)[sorted_idx], class_shap, label=f'Class {class_name}')
    plt.gca().invert_yaxis()
    plt.title("Mean Absolute SHAP Values (All Folds & Classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"mean_abs_shap_by_class.png"), dpi=300)
    plt.savefig(os.path.join(shap_dir, f"mean_abs_shap_by_class.svg"))
    plt.clf()

    # Print SHAP analysis results
    print(f"\nNumber of features: {len(input_cols)}")
    print(f"Ordered feature importance (by mean absolute SHAP value):")
    for idx in sorted_idx:
        print(f" - {input_cols[idx]}: {all_mean_abs_shap[idx]:.4f}")
    print(f"Most unuseful feature: {input_cols[sorted_idx[-1]]}")
    print(f"SHAP plots saved to: {shap_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate DL models.')
    parser.add_argument('--hidden', nargs='*', type=int, default=[128,16], help='Hidden layer sizes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test set size')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--n_repeats', type=int, default=3, help='Number of repeats for numerical stability')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(args)
