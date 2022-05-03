from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train import ToxicClassifier
import torch


def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, preds, beta=1)
    scores = {'accuracy': accuracy, 'f1': f1_score, 'precision': precision, 'recall': recall}
    return scores

def model_init(config, checkpoint_path):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model
