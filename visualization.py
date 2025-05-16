import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from collections import defaultdict
from umap import UMAP

def plot_all_model_heatmaps(models_dict, loader, fine_label_names, coarse_label_names, save_path="all_models_heatmap.png"):
    import math
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    valid_models = {name: model for name, model in models_dict.items() if hasattr(model, "coarse_classifier") or getattr(model, "use_coarse_supervision", False)}
    num_models = len(valid_models)
    rows, cols = 2, math.ceil(num_models / 2)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(valid_models.items()):
        ax = axes[idx]
        model.eval()
        fine_preds, coarse_preds = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                para_id = batch["para_id"].to(model.device)
                line_id = batch["line_id"].to(model.device)
                coarse_logits, fine_logits, _ = model(input_ids, attention_mask, para_id, line_id)
                if coarse_logits is None:
                    continue
                fine = torch.argmax(fine_logits, dim=1).cpu().numpy()
                coarse = torch.argmax(coarse_logits, dim=1).cpu().numpy()
                fine_preds.extend(fine)
                coarse_preds.extend(coarse)

        matrix = np.zeros((len(coarse_label_names), len(fine_label_names)))
        for c, f in zip(coarse_preds, fine_preds):
            matrix[c, f] += 1
        matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-6)

        sns.heatmap(
            matrix, cmap="coolwarm", annot=False, ax=ax, cbar=True,
            xticklabels=fine_label_names, yticklabels=coarse_label_names,
            linewidths=0.5, linecolor='gray'
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Fine Labels", fontsize=10)
        ax.set_ylabel("Coarse Labels", fontsize=10)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    # Remove empty axes
    for j in range(len(valid_models), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_all_model_norm_trends(models_dict, loader, save_path="all_models_norm_trend.png"):
    plt.figure(figsize=(8, 5))
    for name, model in models_dict.items():
        if not hasattr(model, "para_embed"):
            continue
        model.eval()
        norms = []
        with torch.no_grad():
            for batch in loader:
                para_id = batch["para_id"].to(model.device)
                line_id = batch["line_id"].to(model.device)
                para_vec = model.para_embed(para_id)
                line_vec = model.line_embed(line_id)
                norm = torch.norm(torch.cat([para_vec, line_vec], dim=-1), dim=1).mean().item()
                norms.append(norm)
        plt.plot(norms, marker='o', label=name)
    plt.xlabel("Batch Index", fontsize=10)
    plt.ylabel("Average Structure Norm", fontsize=10)
    plt.title("Norm Trends of Structure Embedding", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_all_model_umaps(models_dict, loader, fine_label_names, save_path="all_models_umap_named.png"):
    import math
    from umap import UMAP
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch

    num_models = len(models_dict)
    rows, cols = 2, math.ceil(num_models / 2)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        model.eval()
        embeddings, labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                para_id = batch["para_id"].to(model.device)
                line_id = batch["line_id"].to(model.device)
                z, _ = model.get_embeddings(input_ids, attention_mask, para_id, line_id)
                embeddings.append(z.cpu().numpy())
                labels.append(batch["fine_label"].cpu().numpy())

        X = np.concatenate(embeddings)
        Y = np.concatenate(labels)
        Y_named = np.array([fine_label_names[y] for y in Y])

        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_2d = reducer.fit_transform(X)

        unique_labels = np.unique(Y_named)
        palette = sns.color_palette("tab20", len(unique_labels))
        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in Y_named]

        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=10, alpha=0.7)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)

    # Add custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=label_to_color[label], markersize=6)
               for label in unique_labels]
    fig.legend(handles, unique_labels, title="Fine Labels", loc="lower center", ncol=6, fontsize=8)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()

