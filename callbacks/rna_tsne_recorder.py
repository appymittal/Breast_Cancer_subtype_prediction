import pickle
import torch
from pathlib import Path
from sklearn.manifold import TSNE


class TSNERecorderCallbackSingleOmic:
    def __init__(self, val_loader, device='cpu', save_path="tsne_results_single.pkl", perplexity=30):
        self.val_loader = val_loader
        self.device = device
        self.save_path = Path(save_path)
        self.perplexity = perplexity
        self.results = []

    def on_train_begin(self, trainer=None):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_features(self, model, batch_data):
        """Extract features from the CNN encoder."""
        with torch.no_grad():
            rna = batch_data['rna'].to(self.device)
            z_rna = model.norm_rna(model.rna_vae(rna))  # Encode and normalize
            return z_rna  # [B, latent_dim]

    def on_epoch_end(self, epoch, model=None, trainer=None, **kwargs):
        features, labels = [], []
        model.eval()

        for batch_data, batch_labels in self.val_loader:
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)

            # Get features
            feature_batch = self.extract_features(model, batch_data)
            features.append(feature_batch.cpu())
            labels.append(batch_labels.cpu())

        # Convert to numpy
        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        # Check: tsne requires at least 2 samples
        if features.shape[0] < 2:
            print(f"Warning: Only {features.shape[0]} samples found, skipping t-SNE for epoch {epoch}")
            return

        # Compute TSNE
        tsne = TSNE(n_components=2, perplexity=min(self.perplexity, (features.shape[0] - 1)), random_state=42)
        embeddings = tsne.fit_transform(features)

        self.results.append({
            'epoch': epoch,
            'embeddings': embeddings,
            'labels': labels
        })

    def on_train_end(self, trainer=None):
        with self.save_path.open('wb') as f:
            pickle.dump(self.results, f)
