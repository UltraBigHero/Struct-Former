import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import geoopt
from sklearn.metrics import f1_score, silhouette_score, confusion_matrix
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

COARSE_LABEL_NAMES = ["joy", "sadness", "anger", "fear", "surprise", "love", "neutral"]

# ✅ Structure-Aware Classifier
class StructureAwareClassifier(nn.Module):
    def __init__(self,
                 bert_model_name="bert-base-uncased",
                 num_fine_labels=28,
                 num_coarse_labels=7,
                 structure_dim=8,
                 use_structure=True,
                 use_hyperbolic=False,
                 project_structure=True,
                 use_coarse_supervision=False):
        super().__init__()
        self.use_structure = use_structure
        self.use_hyperbolic = use_hyperbolic
        self.project_structure = project_structure
        self.use_coarse_supervision = use_coarse_supervision

        self.bert = AutoModel.from_pretrained(bert_model_name, local_files_only=True)
        self.bert_hidden = self.bert.config.hidden_size

        if use_structure:
            self.para_embed = nn.Embedding(100, structure_dim)
            self.line_embed = nn.Embedding(200, structure_dim)
            if project_structure:
                self.struct_proj = nn.Sequential(
                    nn.Linear(2 * structure_dim, self.bert_hidden),
                    nn.ReLU(),
                    nn.Linear(self.bert_hidden, self.bert_hidden)
                )
                self.gate_layer = nn.Sequential(
                    nn.Linear(self.bert_hidden, self.bert_hidden),
                    nn.Sigmoid()
                )

        if use_hyperbolic:
            self.manifold = geoopt.PoincareBall(c=0.6)
            self.coarse_proj_weight = nn.Parameter(torch.randn(num_coarse_labels, self.bert_hidden))
            self.coarse_proj_bias = nn.Parameter(torch.zeros(num_coarse_labels))
            self.proj_hyper = nn.Linear(self.bert_hidden, self.bert_hidden)
        else:
            self.coarse_classifier = nn.Linear(self.bert_hidden, num_coarse_labels)

        self.fine_classifier = nn.Linear(self.bert_hidden, num_fine_labels)
        self.struct_coarse_predictor = nn.Linear(self.bert_hidden, num_coarse_labels)

    def _compute_embeddings(self, input_ids, attention_mask, para_id=None, line_id=None):
        cls_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        struct_h = None

        if self.use_structure and para_id is not None and line_id is not None:
            para_vec = self.para_embed(para_id)
            line_vec = self.line_embed(line_id)
            struct = torch.cat([para_vec, line_vec], dim=-1)

            if self.project_structure:
                struct_proj = self.struct_proj(struct)
                gate = self.gate_layer(struct_proj)
                struct_h = gate * struct_proj + (1 - gate) * cls_emb
            else:
                struct_h = F.pad(struct, (0, self.bert_hidden - struct.shape[1]))

            # ✅ If using hyperbolic + structure, map struct to hyperbolic space
            if self.use_hyperbolic:
                struct_h = self.manifold.projx(struct_h)

        if self.use_hyperbolic:
            x = self.manifold.projx(cls_emb)
            if struct_h is not None:
                x = self.manifold.mobius_add(x, struct_h)
        else:
            x = cls_emb + struct_h if struct_h is not None else cls_emb

        return x, struct_h

    def forward(self, input_ids, attention_mask, para_id=None, line_id=None):
        x, struct_h = self._compute_embeddings(input_ids, attention_mask, para_id, line_id)

        if self.use_hyperbolic:
            coarse_logits = (torch.einsum("bd,cd->bc", x, self.coarse_proj_weight) + self.coarse_proj_bias
                             if self.use_coarse_supervision else None)
        else:
            coarse_logits = self.coarse_classifier(x) if self.use_coarse_supervision else None

        fine_logits = self.fine_classifier(x)
        struct_logits = self.struct_coarse_predictor(struct_h if struct_h is not None else x)
        return coarse_logits, fine_logits, struct_logits

    def get_embeddings(self, input_ids, attention_mask, para_id=None, line_id=None):
        x, struct_h = self._compute_embeddings(input_ids, attention_mask, para_id, line_id)

        if self.use_hyperbolic:
            # Directly project x and structure embedding into hyperbolic space
            x_hyp = self.manifold.expmap0(x)

            if struct_h is not None:
                struct_h = self.manifold.projx(struct_h)  # already done in forward()

            return x, x_hyp  # return both Euclidean and hyperbolic
        else:
            return x, self.fine_classifier(x)  # use fine logits as structure proxy

# ✅ Structural geometry loss
def compute_structural_geometric_loss(z_euc, z_hyp, para_ids, model, margin=1.0, weight=0.05):
    para_ids = para_ids.detach().cpu()
    loss = 0.0
    count = 0

    for pid in torch.unique(para_ids):
        mask = para_ids == pid
        z_euc_grp = z_euc[mask]
        z_hyp_grp = z_hyp[mask]
        if z_euc_grp.size(0) < 2:
            continue
        r_euc = torch.log1p(torch.cdist(z_euc_grp, z_euc_grp, p=2).mean())
        r_hyp = torch.log1p(torch.cdist(z_hyp_grp, z_hyp_grp, p=2).mean())
        loss += F.relu(r_hyp - r_euc + margin)
        count += 1

    para_centers = []
    for pid in torch.unique(para_ids):
        z_h = z_hyp[para_ids == pid]
        z_e = z_euc[para_ids == pid]
        if len(z_h) > 0:
            para_centers.append((z_h.mean(0), z_e.mean(0)))

    for i in range(len(para_centers)):
        for j in range(i + 1, len(para_centers)):
            r_euc = torch.log1p(F.pairwise_distance(para_centers[i][1], para_centers[j][1], p=2))
            r_hyp = torch.log1p(model.manifold.dist(para_centers[i][0], para_centers[j][0]))
            if abs(i - j) == 1:
                loss += F.relu(r_hyp - r_euc + margin)
            elif abs(i - j) >= 2:
                loss += F.relu(r_euc - r_hyp + margin)
            count += 1

    return loss / (count + 1e-8)

# ✅ Total loss function
def compute_total_loss(fine_logits, fine_labels,
                       coarse_logits=None, coarse_labels=None,
                       struct_logits=None, para_ids=None,
                       z_euc=None, z_hyp=None,
                       lambda_fine=1.0, lambda_coarse=0.5,
                       lambda_align=0.4, lambda_geom=0.05,
                       margin=1.0, model=None):
    losses = {}
    losses['fine'] = lambda_fine * F.cross_entropy(fine_logits, fine_labels)
    total_loss = losses['fine']

    if coarse_logits is not None and coarse_labels is not None:
        losses['coarse'] = lambda_coarse * F.cross_entropy(coarse_logits, coarse_labels)
        total_loss += losses['coarse']

    if struct_logits is not None and z_euc is not None and struct_logits.size() == z_euc.size():
        coarse_soft_label = F.softmax(coarse_logits.detach(), dim=-1)
        logits_log = F.log_softmax(struct_logits, dim=-1)
        losses['align'] = lambda_align * F.kl_div(logits_log, coarse_soft_label, reduction='batchmean')
        total_loss += losses['align']

    if z_euc is not None and z_hyp is not None and para_ids is not None and getattr(model, "use_hyperbolic", False):
        losses['geom'] = lambda_geom * compute_structural_geometric_loss(z_euc, z_hyp, para_ids, model, margin=margin)
        total_loss += losses['geom']

    losses['total'] = total_loss
    return losses

def extended_eval(model, dataloader, fine_to_coarse_map):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets, all_coarse_labels, all_losses, all_embeddings = [], [], [], [], []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            para_id = batch['para_id'].to(device)
            line_id = batch['line_id'].to(device)
            fine_label = batch['fine_label'].to(device)
            coarse_label = batch['coarse_label'].to(device)

            coarse_logits, fine_logits, _ = model(input_ids, attention_mask, para_id, line_id)
            preds = torch.argmax(fine_logits, dim=1)
            z, _ = model.get_embeddings(input_ids, attention_mask, para_id, line_id)

            all_preds.append(preds.cpu())
            all_targets.append(fine_label.cpu())
            all_coarse_labels.append(coarse_label.cpu())
            all_losses.append(criterion(fine_logits, fine_label).cpu())
            all_embeddings.append(z.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_coarse_labels = torch.cat(all_coarse_labels)
    all_losses = torch.cat(all_losses)
    all_embeddings = torch.cat(all_embeddings)

    macro_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average="macro")
    weighted_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average="weighted")

    # === Coarse label-wise metrics
    coarse_loss, coarse_preds, coarse_trues = defaultdict(list), defaultdict(list), defaultdict(list)
    for p, t, l, c in zip(all_preds, all_targets, all_losses, all_coarse_labels):
        c = int(c.item())
        coarse_loss[c].append(l.item())
        coarse_preds[c].append(p.item())
        coarse_trues[c].append(t.item())

    avg_loss_per_coarse = {k: np.mean(v) for k, v in coarse_loss.items()}
    macro_f1_per_coarse = {
        k: f1_score(coarse_trues[k], coarse_preds[k], average="macro")
        for k in coarse_preds if len(set(coarse_trues[k])) > 1
    }

    # === Sentiment-weighted F1
    # COARSE_LABEL_NAMES = [ 0:"joy", 1:"sadness", 2:"anger", 3:"fear", 4:"surprise", 5:"love", 6:"neutral"]
    sentiment_map = {
        0: "positive", 1: "negative", 2: "negative",
        3: "negative", 4: "neutral", 5: "positive", 6: "neutral"
    }
    sentiment_trues, sentiment_preds = defaultdict(list), defaultdict(list)
    for p, t, c in zip(all_preds, all_targets, all_coarse_labels):
        sent = sentiment_map[int(c.item())]
        sentiment_trues[sent].append(t.item())
        sentiment_preds[sent].append(p.item())
    sentiment_weighted_f1 = {
        k: f1_score(sentiment_trues[k], sentiment_preds[k], average="weighted")
        for k in sentiment_trues if len(set(sentiment_trues[k])) > 1
    }

    # === Hierarchical macro-F1 (average across coarse clusters)
    hierarchical_f1_score = np.mean([
        f1_score(coarse_trues[k], coarse_preds[k], average="macro")
        for k in coarse_preds if len(set(coarse_trues[k])) > 1
    ]) if coarse_preds else 0.0

    # === Silhouette Score (only if enough variance)
    try:
        silhouette = silhouette_score(all_embeddings.numpy(), all_targets.numpy())
    except Exception:
        silhouette = -1.0

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hierarchical_macro_f1": hierarchical_f1_score,
        "silhouette_score": silhouette,
        "loss_per_coarse_label": avg_loss_per_coarse,
        "macro_f1_per_coarse_label": macro_f1_per_coarse,
        "sentiment_weighted_f1": sentiment_weighted_f1
    }

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (batch, seq_len)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)              # (batch, hidden_dim)
        return out

class HANClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=128, num_classes=28):
        super(HANClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, para_id=None, line_id=None):
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids)         # (batch, seq_len, embed_dim)
        h, _ = self.gru(embeds)                    # (batch, seq_len, hidden_dim*2)
        pooled = self.attn(h)                      # (batch, hidden_dim*2)
        logits = self.classifier(pooled)           # (batch, num_classes)
        
        # ✅ To be compatible with main training loop
        coarse_logits = None                       # No coarse classifier in HAN
        fine_logits = logits
        struct_logits = None                       # No structural logits

        return coarse_logits, fine_logits, struct_logits

    def get_embeddings(self, input_ids, attention_mask, para_id, line_id):
        # Returns z_euc and z_hyp; HAN supports only Euclidean
        embeds = self.embedding(input_ids)
        h, _ = self.gru(embeds)
        pooled = self.attn(h)
        return pooled, None