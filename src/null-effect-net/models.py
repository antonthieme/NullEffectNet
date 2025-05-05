import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.3, neg_weight=9.0):
        super().__init__()
        self.neg_weight = neg_weight
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, q_feats):
        logits = self.mlp(q_feats)  # (batch_size, 1)
        return logits.squeeze(-1)   # (batch_size,)

    def predict_proba(self, q_feats):
        with torch.no_grad():
            logits = self.forward(q_feats)
            return torch.sigmoid(logits)

    def compute_loss(self, logits, labels):
        weights = torch.ones_like(labels, device=labels.device)
        weights[labels == 0] = self.neg_weight
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            weight=weights
        )


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=32, neg_weight=8.0, only_active=False):
        super().__init__()
        self.neg_weight = neg_weight
        self.only_active = only_active
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 1 logit
        )
    
    def forward(self, data, query_node_indices, active_node_indices=None):
        # data.x: (num_nodes, feature_dim)
        # data.edge_index: (2, num_edges)
        data.x = data.x.to(torch.float32)
        
        # Process in batch mode
        if self.only_active and active_node_indices is not None:
            # Assuming active_node_indices is a tuple of lists, one for each batch item
            # And query_node_indices is also a tuple/list with batch items
            batch_size = len(active_node_indices)
            all_logits = []
            
            for batch_idx in range(batch_size):
                # Get active nodes for this batch item
                batch_active_nodes = active_node_indices[batch_idx]
                
                # Convert to tensor if needed
                if not isinstance(batch_active_nodes, torch.Tensor):
                    batch_active_nodes = torch.tensor(batch_active_nodes, device=data.x.device)
                
                # Ensure indices are within bounds
                max_idx = data.x.size(0) - 1
                valid_indices = batch_active_nodes[batch_active_nodes <= max_idx]
                
                # Get node features for the active subgraph
                x_sub = data.x[valid_indices]
                
                # Create a mapping from original node indices to new indices in the subgraph
                node_mapper = {int(old_idx): new_idx for new_idx, old_idx in enumerate(valid_indices)}
                
                # Filter edges to only include those between active nodes
                mask = torch.isin(data.edge_index[0], valid_indices) & torch.isin(data.edge_index[1], valid_indices)
                filtered_edge_index = data.edge_index[:, mask]
                
                # Remap node indices in edge_index to match the new ordering
                remapped_edges = torch.zeros_like(filtered_edge_index)
                for i in range(filtered_edge_index.size(1)):
                    remapped_edges[0, i] = node_mapper[int(filtered_edge_index[0, i])]
                    remapped_edges[1, i] = node_mapper[int(filtered_edge_index[1, i])]
                
                # Get the query node for this batch item
                batch_query_idx = query_node_indices[batch_idx] if isinstance(query_node_indices, (list, tuple)) else query_node_indices[batch_idx:batch_idx+1]
                
                # Check if the query node is in the active set
                if isinstance(batch_query_idx, torch.Tensor) and batch_query_idx.numel() > 0:
                    query_in_active = torch.isin(batch_query_idx, valid_indices)
                    if not query_in_active.all():
                        # For simplicity, if query node isn't in active set, use the whole graph for this batch item
                        query_embed = self._process_full_graph(data, batch_query_idx)
                        all_logits.append(self.mlp(query_embed).squeeze(-1))
                        continue
                elif isinstance(batch_query_idx, (int, list)):
                    # Handle scalar or list case
                    query_idx_tensor = torch.tensor(batch_query_idx, device=data.x.device) if isinstance(batch_query_idx, list) else torch.tensor([batch_query_idx], device=data.x.device)
                    query_in_active = torch.isin(query_idx_tensor, valid_indices)
                    if not query_in_active.all():
                        query_embed = self._process_full_graph(data, query_idx_tensor)
                        all_logits.append(self.mlp(query_embed).squeeze(-1))
                        continue
                
                # Map query node to its index in the subgraph
                if isinstance(batch_query_idx, torch.Tensor):
                    remapped_query_idx = torch.tensor([node_mapper[int(idx)] for idx in batch_query_idx], device=data.x.device)
                else:
                    # Handle scalar case
                    remapped_query_idx = torch.tensor([node_mapper[int(batch_query_idx)]], device=data.x.device)
                
                # Apply GNN layers on the subgraph
                x = self.conv1(x_sub, remapped_edges)
                x = F.relu(x)
                x = self.conv2(x, remapped_edges)
                x = F.relu(x)
                
                # Get embedding for query node
                query_embedding = x[remapped_query_idx]
                
                # Get logits for this batch item
                batch_logits = self.mlp(query_embedding).squeeze(-1)
                all_logits.append(batch_logits)
            
            # Combine logits from all batch items
            if all(isinstance(l, torch.Tensor) and l.numel() == 1 for l in all_logits):
                return torch.cat([l.view(1) for l in all_logits])
            else:
                return torch.cat(all_logits)
                
        else:
            # Process the entire graph for all queries
            return self._process_full_graph_batch(data, query_node_indices)
    
    def _process_full_graph(self, data, query_node_idx):
        # Process the entire graph for a single query or list of queries
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        
        # Return embeddings for query nodes
        return x[query_node_idx]
    
    def _process_full_graph_batch(self, data, query_node_indices):
        # Process the entire graph for all queries in batch
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        
        # Handle different formats of query_node_indices
        if isinstance(query_node_indices, (list, tuple)):
            all_logits = []
            for query_idx in query_node_indices:
                if isinstance(query_idx, (list, tuple)):
                    query_idx = torch.tensor(query_idx, device=x.device)
                query_embed = x[query_idx]
                batch_logits = self.mlp(query_embed).squeeze(-1)
                all_logits.append(batch_logits)
            
            # Combine logits from all batch items
            if all(isinstance(l, torch.Tensor) and l.numel() == 1 for l in all_logits):
                return torch.cat([l.view(1) for l in all_logits])
            else:
                return torch.cat(all_logits)
        else:
            # Handle tensor case
            query_embeddings = x[query_node_indices]
            logits = self.mlp(query_embeddings)
            return logits.squeeze(-1)

    def predict_proba(self, data, query_node_indices, active_node_indices=None):
        with torch.no_grad():
            logits = self.forward(data, query_node_indices, active_node_indices)
            return torch.sigmoid(logits)

    def compute_loss(self, logits, labels):
        weights = torch.ones_like(labels, device=labels.device)
        weights[labels == 0] = self.neg_weight
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            weight=weights
        )


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        """
        x: (B, S, D)
        mask: (B, S), 1 if valid, 0 if padded
        """
        z = self.embed(x)  # (B, S, H)
        attn_scores = self.attn(z).squeeze(-1)  # (B, S)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=1)  # (B, S)
        pooled = torch.sum(z * attn_weights.unsqueeze(-1), dim=1)  # (B, H)
        return self.output(pooled), attn_weights  # (B, output_dim)


class GNNAttentionClassifier(nn.Module):
    def __init__(self, 
                 input_dim,       
                 pool_hidden_dim=64,
                 pool_out_dim=128,
                 gcn_hidden_dim=128,
                 gcn_out_dim=32,
                 neg_weight=8.0,
                 only_active=False):
        super().__init__()
        self.neg_weight = neg_weight
        self.only_active = only_active

        # Deep Sets pooling layer
        self.pooling = AttentionPooling(input_dim, pool_hidden_dim, pool_out_dim)

        # GCN layers
        self.conv1 = GCNConv(pool_out_dim, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_out_dim)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data, query_node_indices, active_node_indices=None, return_attn=False):
        # Compute pooled node embeddings; don't overwrite data.x
        pooled_x, attn_weights = self.pooling(data.set_features, data.set_mask)

        batch_size = len(query_node_indices)

        if not self.only_active or active_node_indices is None:
            # Full-graph GCN pass
            x = self.conv1(pooled_x, data.edge_index)
            x = F.relu(x)
            x = self.conv2(x, data.edge_index)
            x = F.relu(x)

            query_embeddings = x[query_node_indices]
            logits = self.mlp(query_embeddings)
            if return_attn:
                return logits.squeeze(-1), attn_weights
            return logits.squeeze(-1)

        # Active-node subgraph processing
        all_logits = []

        for i in range(batch_size):
            query_idx = query_node_indices[i].item()
            active_nodes = active_node_indices[i]

            if query_idx not in active_nodes:
                active_nodes = active_nodes + [query_idx]

            active_tensor = torch.tensor(active_nodes, device=pooled_x.device)

            sub_features = pooled_x[active_tensor]

            edge_mask = torch.isin(data.edge_index[0], active_tensor) & torch.isin(data.edge_index[1], active_tensor)
            sub_edges = data.edge_index[:, edge_mask]

            idx_map = {int(old): new for new, old in enumerate(active_tensor)}
            remapped_edges = torch.zeros_like(sub_edges)

            for j in range(sub_edges.size(1)):
                remapped_edges[0, j] = idx_map[int(sub_edges[0, j])]
                remapped_edges[1, j] = idx_map[int(sub_edges[1, j])]

            sub_x = self.conv1(sub_features, remapped_edges)
            sub_x = F.relu(sub_x)
            sub_x = self.conv2(sub_x, remapped_edges)
            sub_x = F.relu(sub_x)

            query_subgraph_idx = idx_map[query_idx]
            query_embedding = sub_x[query_subgraph_idx]

            logit = self.mlp(query_embedding).squeeze(-1)
            all_logits.append(logit)

        if return_attn:
            return torch.cat(all_logits), attn_weights
        return torch.cat(all_logits)

    def predict_proba(self, data, query_node_indices, active_node_indices=None):
        with torch.no_grad():
            logits = self.forward(data, query_node_indices, active_node_indices)
            return torch.sigmoid(logits)

    def compute_loss(self, logits, labels):
        weights = torch.ones_like(labels, device=labels.device)
        weights[labels == 0] = self.neg_weight
        return F.binary_cross_entropy_with_logits(logits, labels.float(), weight=weights)

