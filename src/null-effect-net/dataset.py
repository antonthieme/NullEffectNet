from torch.utils.data import Dataset
import torch
import numpy as np

class MLPDataset(Dataset):
    def __init__(self, label_df, node_features_df, device='cpu'):
        self.label_df = label_df.reset_index(drop=True)
        self.device = device

        # Build mapping from Ensembl ID to concatenated features
        self.feature_dict = {
            row["Ensembl ID"]: torch.tensor(row['Concat Embedding'], dtype=torch.float32)
            for _, row in node_features_df.iterrows()
        }

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        target_id = row["Target"]
        label = torch.tensor(row["Perturbed"], dtype=torch.float32).to(self.device)

        # Get query features
        try:
            query_feat = self.feature_dict[target_id].to(self.device)
        except KeyError:
            raise ValueError(f"Target node '{target_id}' not found in node_features_df.")

        return query_feat, label

def collate_function_mlp(batch):
    q_feats, labels = zip(*batch)
    q_feats = torch.stack(q_feats)  # (batch_size, feat_dim)
    labels = torch.stack(labels)    # (batch_size,)
    return q_feats, labels


class GNNDataset(Dataset):
    def __init__(self, label_df, active_nodes_df, node_features_df, node_idx_mapping, device='cpu'):
        self.label_df = label_df.reset_index(drop=True)
        self.device = device
        self.node_idx_mapping = node_idx_mapping  # Mapping from Ensembl ID to node index

        # Build mapping from (Cell Line) to list of active genes
        self.cell_line_to_active = (
            active_nodes_df.groupby("Cell Line")["Gene"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        target_id = row["Target"]
        label = torch.tensor(row["Perturbed"], dtype=torch.float32).to(self.device)

        # Map target Ensembl ID to node index
        try:
            target_idx = self.node_idx_mapping[target_id]
        except KeyError:
            raise ValueError(f"Target node '{target_id}' not found in node_idx_mapping.")

        active_idxs = []
        for ensembl_id in self.cell_line_to_active[row['Cell Line']]:
            try:
                active_idxs.append(self.node_idx_mapping[ensembl_id])
            except KeyError:
                # No embeddings for active gene --> not part of network
                pass

        return target_idx, label, active_idxs


def collate_function_gnn(batch):
    target_indices, labels, active_idxs = zip(*batch)
    target_indices = torch.tensor(target_indices, dtype=torch.long)
    labels = torch.stack(labels)
    return target_indices, labels, active_idxs


class GNNAttentionDataset(Dataset):
    def __init__(self, label_df, active_nodes_df, node_features_df, node_idx_mapping, device='cpu'):
        self.label_df = label_df.reset_index(drop=True)
        self.device = device
        self.node_idx_mapping = node_idx_mapping  # Mapping from Ensembl ID to node index

        # Build mapping from (Cell Line) to list of active genes
        self.cell_line_to_active = (
            active_nodes_df.groupby("Cell Line")["Gene"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        target_id = row["Target"]
        label = torch.tensor(row["Perturbed"], dtype=torch.float32).to(self.device)

        # Map target Ensembl ID to node index
        try:
            target_idx = self.node_idx_mapping[target_id]
        except KeyError:
            raise ValueError(f"Target node '{target_id}' not found in node_idx_mapping.")

        active_idxs = []
        for ensembl_id in self.cell_line_to_active[row['Cell Line']]:
            try:
                active_idxs.append(self.node_idx_mapping[ensembl_id])
            except KeyError:
                # No embeddings for active gene --> not part of network
                pass

        return target_idx, label, active_idxs

