import torch
import torch.nn as nn

class SpatiotemporalTransformer(nn.Module):
    """
    Spatiotemporal Transformer Network using Cross-Attention.
    Processes atmospheric (core) and satellite features through parallel
    self-attention encoders, then uses cross-attention (core attending to sat)
    to fuse multi-modal representations before classification.
    """
    def __init__(self, num_core_feats: int, num_sat_feats: int, num_stale_feats: int = 0,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_core = num_core_feats
        self.num_sat = num_sat_feats
        self.num_stale = num_stale_feats
        
        # Embeddings to project 1D features into d_model space
        self.core_embedding = nn.Linear(num_core_feats, d_model)
        
        # Handle cases where satellite features might be empty
        if num_sat_feats > 0:
            self.sat_embedding = nn.Linear(num_sat_feats, d_model)
            encoder_layer_sat = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.15)
            self.sat_encoder = nn.TransformerEncoder(encoder_layer_sat, num_layers=num_layers)
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=0.15)
        
        if num_stale_feats > 0:
            self.stale_embedding = nn.Linear(num_stale_feats, d_model)
            
        encoder_layer_core = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.15)
        self.core_encoder = nn.TransformerEncoder(encoder_layer_core, num_layers=num_layers)
        
        # Output head
        out_dim = d_model * (2 if num_sat_feats > 0 else 1)
        if num_stale_feats > 0:
            out_dim += d_model
            
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the flattened input back into modalities based on feature counts
        core_x = x[:, :self.num_core]
        
        # Project and add sequence dimension: (batch, 1, d_model)
        core_emb = self.core_embedding(core_x).unsqueeze(1)
        core_enc = self.core_encoder(core_emb)
        
        features = []
        if self.num_sat > 0:
            sat_x = x[:, self.num_core:self.num_core+self.num_sat]
            sat_emb = self.sat_embedding(sat_x).unsqueeze(1)
            sat_enc = self.sat_encoder(sat_emb)
            
            # Cross attention: Core query attends to Satellite key/value
            cross_out, _ = self.cross_attn(query=core_enc, key=sat_enc, value=sat_enc)
            
            features.append(cross_out.squeeze(1))
            features.append(sat_enc.squeeze(1))
        else:
            features.append(core_enc.squeeze(1))
            
        if self.num_stale > 0:
            stale_x = x[:, self.num_core+self.num_sat:]
            stale_emb = self.stale_embedding(stale_x)
            features.append(stale_emb)
            
        concat = torch.cat(features, dim=-1)
        logits = self.head(concat).squeeze(-1)
        return logits
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
