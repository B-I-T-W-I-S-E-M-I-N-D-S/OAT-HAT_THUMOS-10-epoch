import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class AdaptiveSpanAttention(nn.Module):
    """
    Fast Adaptive Span Attention with efficient computation.
    """
    def __init__(self, embedding_dim, num_heads, dropout, max_span=64, min_span=8):
        super(AdaptiveSpanAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = min(num_heads, 4)  # Limit heads for speed
        self.max_span = min(max_span, 64)   # Limit span for speed
        self.min_span = min_span
        
        # Single efficient attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=False
        )
        
        # Lightweight span predictor
        self.span_predictor = nn.Linear(embedding_dim, 1)
        
        # Simple gating
        self.gate = nn.Linear(embedding_dim, embedding_dim)
        
        # Temporal decay (fixed for speed)
        self.register_buffer('temporal_decay', torch.tensor(0.95))
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, long_x, encoded_x):
        """
        Fast forward pass with minimal computation.
        """
        batch_size = long_x.shape[1]
        full_seq_len = long_x.shape[0]
        
        # Simple span prediction (single value for all heads)
        context_rep = torch.mean(encoded_x, dim=0)  # [batch, dim]
        span_ratio = torch.sigmoid(self.span_predictor(context_rep))  # [batch, 1]
        
        # Compute adaptive span
        adaptive_span = self.min_span + (self.max_span - self.min_span) * span_ratio
        adaptive_span = adaptive_span.clamp(self.min_span, min(self.max_span, full_seq_len))
        avg_span = int(adaptive_span.mean().item())
        
        # Select relevant history (same span for all batches for efficiency)
        start_idx = max(0, full_seq_len - avg_span)
        relevant_history = long_x[start_idx:]  # [span_len, batch, dim]
        
        # Apply temporal decay
        seq_len = relevant_history.shape[0]
        positions = torch.arange(seq_len, device=long_x.device, dtype=torch.float)
        decay_weights = self.temporal_decay ** (seq_len - 1 - positions)
        decay_weights = decay_weights.unsqueeze(1).unsqueeze(2)  # [seq_len, 1, 1]
        
        # Apply decay to history
        weighted_history = relevant_history * decay_weights
        
        # Single attention computation
        attended_output, _ = self.attention(
            query=encoded_x,
            key=weighted_history,
            value=weighted_history
        )
        
        # Simple gating
        gate_weights = torch.sigmoid(self.gate(attended_output))
        adaptive_features = attended_output * gate_weights
        
        # Residual connection and normalization
        output = self.layer_norm(adaptive_features + encoded_x)
        
        # Return simplified outputs
        attention_weights = torch.ones(encoded_x.shape[0], batch_size, device=long_x.device)
        actual_spans = [avg_span]
        
        return attention_weights, output, actual_spans


class HierarchicalContextEncoder(nn.Module):
    """
    Fast hierarchical encoder with efficient multi-scale processing.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(HierarchicalContextEncoder, self).__init__()
        
        # Simplified encoders
        self.fine_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=min(num_heads, 4),  # Limit heads
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        self.coarse_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=min(num_heads // 2, 2),  # Even fewer heads
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Simple fusion
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Lightweight feature selection
        self.feature_selector = nn.Linear(embedding_dim * 2, 2)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, attention_mask):
        """
        Fast forward pass with minimal computation.
        """
        seq_len, batch_size, dim = features.shape
        
        # Fine scale processing
        fine_features = self.fine_encoder(features)
        
        # Coarse scale processing (only if beneficial)
        if seq_len >= 8:
            # Simple downsampling
            coarse_features = features.permute(1, 2, 0)  # [batch, dim, seq]
            coarse_features = F.avg_pool1d(coarse_features, kernel_size=2, stride=2, padding=0)
            coarse_features = coarse_features.permute(2, 0, 1)  # [seq//2, batch, dim]
            
            # Encode
            coarse_encoded = self.coarse_encoder(coarse_features)
            
            # Simple upsampling
            coarse_upsampled = coarse_encoded.permute(1, 2, 0)  # [batch, dim, seq//2]
            coarse_upsampled = F.interpolate(coarse_upsampled, size=seq_len, mode='nearest')
            coarse_upsampled = coarse_upsampled.permute(2, 0, 1)  # [seq, batch, dim]
        else:
            coarse_upsampled = fine_features
        
        # Simple feature selection
        combined = torch.cat([fine_features, coarse_upsampled], dim=-1)
        selection_weights = torch.softmax(self.feature_selector(combined), dim=-1)
        
        # Weighted combination
        output = (fine_features * selection_weights[:, :, 0:1] + 
                 coarse_upsampled * selection_weights[:, :, 1:2])
        
        # Simple fusion
        fused = self.fusion(combined)
        output = output + fused
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        
        # Enhanced feature reduction with normalization
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        # Enhanced adaptive span attention
        self.adaptive_span_attention = AdaptiveSpanAttention(
            embedding_dim=n_embedding_dim,
            num_heads=min(4, n_enc_head),  # Limit heads for speed
            dropout=dropout,
            max_span=64,   # Reduced from 200
            min_span=8
        )
        
        # Enhanced hierarchical context encoder
        self.hierarchical_encoder = HierarchicalContextEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=min(4, n_enc_head),  # Limit heads for speed
            dropout=dropout
        )
        
        # Main encoder with reduced layers (since we have hierarchical processing)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                     nhead=n_enc_head, 
                                     dropout=dropout, 
                                     activation='gelu'), 
            max(1, n_enc_layer - 1),  # Reduce by 1 since hierarchical encoder adds complexity
            nn.LayerNorm(n_embedding_dim)
        )
                                            
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                     nhead=n_dec_head, 
                                     dropout=dropout, 
                                     activation='gelu'), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        
        # Simplified context integration
        self.context_integration = nn.Sequential(
            nn.Linear(n_embedding_dim * 2, n_embedding_dim),
            nn.GELU()
        )
        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )                               
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        
        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        
        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)
        
        # Conditional hierarchical processing (only for longer sequences)
        if pe_x.shape[0] > 8:
            attention_mask = torch.ones(pe_x.shape[0], pe_x.shape[1], device=pe_x.device)
            hierarchical_features = self.hierarchical_encoder(pe_x, attention_mask)
        else:
            hierarchical_features = pe_x
        
        # Standard encoder processing
        encoded_x = self.encoder(hierarchical_features)
        
        # Conditional adaptive attention (only for longer sequences)
        if pe_x.shape[0] > 16:
            _, adaptive_enhanced_features, _ = self.adaptive_span_attention(pe_x, encoded_x)
            # Integrate features
            integrated_features = self.context_integration(
                torch.cat([encoded_x, adaptive_enhanced_features], dim=-1)
            )
        else:
            integrated_features = encoded_x
        
        # Apply normalization
        integrated_features = self.norm1(integrated_features)
        
        # Decoder processing
        decoder_token = self.decoder_token.expand(-1, integrated_features.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, integrated_features) 
        
        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))
        
        decoded_x = decoded_x.permute([1, 0, 2])
        
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
