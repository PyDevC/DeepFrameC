import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from collections import defaultdict

def video_level_evaluate(model, loader, device):
    """
    Aggregate frame-level predictions to video level.
    Requires dataset to expose video IDs per sample.
    Use mean-pooling of frame probabilities per video.
    """
    model.eval()
    video_probs  = defaultdict(list)
    video_labels = {}

    with torch.no_grad():
        for imgs, labels, video_ids in loader:  # modify dataset __getitem__ to return video_id
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for prob, label, vid_id in zip(probs, labels.numpy(), video_ids):
                video_probs[vid_id].append(prob)
                video_labels[vid_id] = label

    y_score = [np.mean(video_probs[v]) for v in video_probs]
    y_true  = [video_labels[v] for v in video_probs]
    y_pred  = [1 if s > 0.5 else 0 for s in y_score]

    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    print(f"Video-level AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"Video-level ACC: {accuracy_score(y_true, y_pred):.4f}")
