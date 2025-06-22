# Score is 0.1563 use sazzadnet claude

# import numpy as np
# import torch
# import math
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.nn import init
# from torch.nn.functional import normalize


# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float = 0.1,
#                  maxlen: int = 750):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)

#     def forward(self, token_embedding: torch.Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# class AdaptiveMultiScaleEncoder(nn.Module):
#     """Improved multi-scale encoder that preserves original performance"""
    
#     def __init__(self, n_embedding_dim, n_enc_head, n_enc_layer, dropout=0.3):
#         super(AdaptiveMultiScaleEncoder, self).__init__()
#         self.n_embedding_dim = n_embedding_dim
        
#         # Original encoder - keep this as the main pathway
#         self.main_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=n_enc_head, 
#                 dropout=dropout, 
#                 activation='gelu'
#             ), 
#             n_enc_layer, 
#             nn.LayerNorm(n_embedding_dim)
#         )
        
#         # Lightweight multi-scale attention branches
#         self.local_attention = nn.MultiheadAttention(
#             embed_dim=n_embedding_dim,
#             num_heads=max(1, n_enc_head // 4),  # Fewer heads for efficiency
#             dropout=dropout * 0.5,
#             batch_first=False
#         )
        
#         self.global_attention = nn.MultiheadAttention(
#             embed_dim=n_embedding_dim,
#             num_heads=max(1, n_enc_head // 4),
#             dropout=dropout * 0.5,
#             batch_first=False
#         )
        
#         # Adaptive mixing weights (learnable but conservative)
#         self.local_weight = nn.Parameter(torch.tensor(0.05))  # Start small
#         self.global_weight = nn.Parameter(torch.tensor(0.05))
        
#         # Layer normalization for stability
#         self.norm_local = nn.LayerNorm(n_embedding_dim)
#         self.norm_global = nn.LayerNorm(n_embedding_dim)
        
#         # Dropout for regularization
#         self.dropout_scale = nn.Dropout(dropout * 0.5)
        
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (seq_len, batch, n_embedding_dim)
#         """
#         # Main encoding path (preserve original behavior)
#         main_encoded = self.main_encoder(x)
        
#         # Conservative enhancement with small weights
#         local_enhanced, _ = self.local_attention(x, x, x)
#         local_enhanced = self.norm_local(local_enhanced)
#         local_enhanced = self.dropout_scale(local_enhanced)
        
#         global_enhanced, _ = self.global_attention(x, x, x)
#         global_enhanced = self.norm_global(global_enhanced)
#         global_enhanced = self.dropout_scale(global_enhanced)
        
#         # Adaptive combination with small initial weights
#         local_contrib = torch.sigmoid(self.local_weight) * local_enhanced
#         global_contrib = torch.sigmoid(self.global_weight) * global_enhanced
        
#         # Combine with main encoding
#         output = main_encoded + local_contrib + global_contrib
        
#         return output


# class ImprovedHistoryUnit(torch.nn.Module):
#     """Conservative improvement of HistoryUnit"""
    
#     def __init__(self, opt):
#         super(ImprovedHistoryUnit, self).__init__()
#         self.n_feature = opt["feat_dim"] 
#         n_class = opt["num_of_class"]
#         n_embedding_dim = opt["hidden_dim"]
#         n_hist_dec_head = 4
#         n_hist_dec_layer = 5
#         n_hist_dec_head_2 = 4
#         n_hist_dec_layer_2 = 2
#         self.anchors = opt["anchors"]
#         self.history_tokens = 16
#         self.short_window_size = 16
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0
        
#         # Keep ALL original components unchanged
#         self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   
        
#         self.history_encoder_block1 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=n_hist_dec_head, 
#                 dropout=dropout, 
#                 activation='gelu'
#             ), 
#             n_hist_dec_layer, 
#             nn.LayerNorm(n_embedding_dim)
#         )  
        
#         self.history_encoder_block2 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=n_hist_dec_head_2, 
#                 dropout=dropout, 
#                 activation='gelu'
#             ), 
#             n_hist_dec_layer_2, 
#             nn.LayerNorm(n_embedding_dim)
#         )  
        
#         self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim//4), nn.ReLU())     
#         self.snip_classifier = nn.Sequential(
#             nn.Linear(self.history_tokens * n_embedding_dim//4, (self.history_tokens * n_embedding_dim//4)//4), 
#             nn.ReLU(), 
#             nn.Linear((self.history_tokens * n_embedding_dim//4)//4, n_class)
#         )                      
        
#         self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
#         self.norm2 = nn.LayerNorm(n_embedding_dim)
#         self.dropout2 = nn.Dropout(0.1)
        
#         # Add minimal enhancement: temporal smoothing
#         self.temporal_smoother = nn.Conv1d(
#             n_embedding_dim, n_embedding_dim, 
#             kernel_size=3, padding=1, groups=n_embedding_dim
#         )
#         self.smooth_weight = nn.Parameter(torch.tensor(0.02))  # Very small initial weight
        
#         # Initialize new parameters
#         self._init_new_params()
        
#     def _init_new_params(self):
#         """Initialize new parameters conservatively"""
#         nn.init.xavier_uniform_(self.temporal_smoother.weight)
#         if self.temporal_smoother.bias is not None:
#             nn.init.constant_(self.temporal_smoother.bias, 0)
#         nn.init.constant_(self.smooth_weight, 0.02)

#     def forward(self, long_x, encoded_x):
#         # Original processing path
#         hist_pe_x = self.history_positional_encoding(long_x)
#         history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
#         hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
        
#         # Minimal temporal enhancement
#         smooth_weight = torch.sigmoid(self.smooth_weight)
#         if smooth_weight > 0.01:  # Only apply if weight is meaningful
#             # Apply temporal smoothing
#             x_smooth = hist_encoded_x_1.permute(1, 2, 0)  # (batch, dim, seq)
#             x_smooth = self.temporal_smoother(x_smooth)
#             x_smooth = x_smooth.permute(2, 0, 1)  # back to (seq, batch, dim)
#             hist_encoded_x_1 = hist_encoded_x_1 + smooth_weight * x_smooth
        
#         # Continue with original processing
#         hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
#         hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
#         hist_encoded_x = self.norm2(hist_encoded_x_2)
   
#         # Original snippet classification
#         snippet_feat = self.snip_head(hist_encoded_x_1)
#         snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
#         snip_cls = self.snip_classifier(snippet_feat)
        
#         return hist_encoded_x, snip_cls


# class MYNET(torch.nn.Module):
#     def __init__(self, opt):
#         super(MYNET, self).__init__()
#         self.n_feature = opt["feat_dim"] 
#         n_class = opt["num_of_class"]
#         n_embedding_dim = opt["hidden_dim"]
#         n_enc_layer = opt["enc_layer"]
#         n_enc_head = opt["enc_head"]
#         n_dec_layer = opt["dec_layer"]
#         n_dec_head = opt["dec_head"]
#         n_comb_dec_head = 4
#         n_comb_dec_layer = 5
#         n_seglen = opt["segment_size"]
#         self.anchors = opt["anchors"]
#         self.history_tokens = 16
#         self.short_window_size = 16
#         self.anchors_stride = []
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0

#         # Keep original feature reduction
#         self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
#         self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
#         # Keep original positional encoding
#         self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
#         # Use enhanced encoder with fallback option
#         self.use_enhanced_encoder = True
#         if self.use_enhanced_encoder:
#             try:
#                 self.encoder = AdaptiveMultiScaleEncoder(
#                     n_embedding_dim=n_embedding_dim,
#                     n_enc_head=n_enc_head,
#                     n_enc_layer=n_enc_layer,
#                     dropout=dropout
#                 )
#             except Exception as e:
#                 print(f"Enhanced encoder failed, using original: {e}")
#                 self.use_enhanced_encoder = False
        
#         if not self.use_enhanced_encoder:
#             # Original encoder as fallback
#             self.encoder = nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(
#                     d_model=n_embedding_dim, 
#                     nhead=n_enc_head, 
#                     dropout=dropout, 
#                     activation='gelu'
#                 ), 
#                 n_enc_layer, 
#                 nn.LayerNorm(n_embedding_dim)
#             )
        
#         # Keep original decoder
#         self.decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=n_dec_head, 
#                 dropout=dropout, 
#                 activation='gelu'
#             ), 
#             n_dec_layer, 
#             nn.LayerNorm(n_embedding_dim)
#         )  

#         # Use improved history unit
#         self.history_unit = ImprovedHistoryUnit(opt)

#         # Keep ALL original components
#         self.history_anchor_decoder_block1 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=n_comb_dec_head, 
#                 dropout=dropout, 
#                 activation='gelu'
#             ), 
#             n_comb_dec_layer, 
#             nn.LayerNorm(n_embedding_dim)
#         )  
            
#         self.classifier = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim), 
#             nn.ReLU(), 
#             nn.Linear(n_embedding_dim, n_class)
#         )
        
#         self.regressor = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim), 
#             nn.ReLU(), 
#             nn.Linear(n_embedding_dim, 2)
#         )    
                           
#         self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
#         self.norm1 = nn.LayerNorm(n_embedding_dim)
#         self.dropout1 = nn.Dropout(0.1)
#         self.relu = nn.ReLU(True)
#         self.softmaxd1 = nn.Softmax(dim=-1)
        
#         # Initialize parameters properly
#         self._initialize_parameters()
        
#     def _initialize_parameters(self):
#         """Initialize parameters to maintain original performance"""
#         # Initialize decoder token like original
#         nn.init.normal_(self.decoder_token, std=0.02)
        
#         # Initialize enhanced components conservatively
#         for name, param in self.named_parameters():
#             if 'local_weight' in name or 'global_weight' in name:
#                 nn.init.constant_(param, 0.05)  # Small initial values
#             elif 'smooth_weight' in name:
#                 nn.init.constant_(param, 0.02)  # Very small

#     def forward(self, inputs):
#         # Exactly same preprocessing as original
#         base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
#         base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
#         base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
#         base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

#         short_x = base_x[-self.short_window_size:]
#         long_x = base_x[:-self.short_window_size]
        
#         # Anchor feature generation with enhanced encoder
#         pe_x = self.positional_encoding(short_x)
#         encoded_x = self.encoder(pe_x)
        
#         decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
#         decoded_x = self.decoder(decoder_token, encoded_x) 

#         # Enhanced history processing
#         hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

#         # Exactly same as original
#         decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
#         decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
#         decoded_anchor_feat = self.norm1(decoded_anchor_feat)
#         decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
#         # Same prediction modules
#         anc_cls = self.classifier(decoded_anchor_feat)
#         anc_reg = self.regressor(decoded_anchor_feat)
        
#         return anc_cls, anc_reg, snip_cls

 
# class SuppressNet(torch.nn.Module):
#     """Enhanced SuppressNet with better regularization"""
    
#     def __init__(self, opt):
#         super(SuppressNet, self).__init__()
#         n_class = opt["num_of_class"] - 1
#         n_seglen = opt["segment_size"]
#         n_embedding_dim = 2 * n_seglen
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0
        
#         # Keep original architecture but add improvements
#         self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
#         self.mlp2 = nn.Linear(n_embedding_dim, 1)
#         self.norm = nn.InstanceNorm1d(n_class)
#         self.relu = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()
        
#         # Add batch normalization for stability
#         self.bn1 = nn.BatchNorm1d(n_class * n_embedding_dim)
#         self.dropout = nn.Dropout(dropout * 0.7)  # Slightly less aggressive
        
#         # Add skip connection weight
#         self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
#     def forward(self, inputs):
#         # inputs - batch x seq_len x class
#         base_x = inputs.permute([0,2,1])
#         base_x = self.norm(base_x)
        
#         # Store original for skip connection
#         identity = base_x
        
#         x = self.relu(self.mlp1(base_x))
        
#         # Apply batch norm and dropout
#         x_flat = x.view(x.size(0), -1)
#         x_flat = self.bn1(x_flat)
#         x = x_flat.view_as(x)
#         x = self.dropout(x)
        
#         x = self.sigmoid(self.mlp2(x))
#         x = x.squeeze(-1)
        
#         return x


# # Alternative: More Conservative Enhancement
# class MinimalEnhancedMYNET(MYNET):
#     """Minimal enhancement that preserves original behavior"""
    
#     def __init__(self, opt):
#         # Initialize parent class first
#         super().__init__(opt)
        
#         # Override with minimal changes
#         n_embedding_dim = opt["hidden_dim"]
        
#         # Replace encoder with barely enhanced version
#         self.encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=n_embedding_dim, 
#                 nhead=opt["enc_head"], 
#                 dropout=0.3, 
#                 activation='gelu'
#             ), 
#             opt["enc_layer"], 
#             nn.LayerNorm(n_embedding_dim)
#         )
        
#         # Add tiny enhancement layer
#         self.feature_enhancer = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim),
#             nn.LayerNorm(n_embedding_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
        
#         # Enhancement weight (starts very small)
#         self.enhance_weight = nn.Parameter(torch.tensor(0.01))
        
#     def forward(self, inputs):
#         # Same preprocessing
#         base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
#         base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
#         base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
#         base_x = base_x.permute([1,0,2])
#         short_x = base_x[-self.short_window_size:]
#         long_x = base_x[:-self.short_window_size]
        
#         # Encoding with minimal enhancement
#         pe_x = self.positional_encoding(short_x)
#         encoded_x = self.encoder(pe_x)
        
#         # Tiny enhancement
#         enhanced_x = self.feature_enhancer(encoded_x)
#         encoded_x = encoded_x + torch.sigmoid(self.enhance_weight) * enhanced_x
        
#         # Rest same as original
#         decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
#         decoded_x = self.decoder(decoder_token, encoded_x) 
        
#         hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)
        
#         decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
#         decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
#         decoded_anchor_feat = self.norm1(decoded_anchor_feat)
#         decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
#         anc_cls = self.classifier(decoded_anchor_feat)
#         anc_reg = self.regressor(decoded_anchor_feat)
        
#         return anc_cls, anc_reg, snip_cls








# Score 0.15099

# import numpy as np
# import torch
# import math
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.nn import init
# from torch.nn.functional import normalize


# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float = 0.1,
#                  maxlen: int = 750):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)

#     def forward(self, token_embedding: torch.Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# class StabilizedHistoryUnit(torch.nn.Module):
#     """Minimally enhanced HistoryUnit that maintains original performance"""
    
#     def __init__(self, opt):
#         super(StabilizedHistoryUnit, self).__init__()
#         self.n_feature = opt["feat_dim"] 
#         n_class = opt["num_of_class"]
#         n_embedding_dim = opt["hidden_dim"]
#         n_hist_dec_head = 4
#         n_hist_dec_layer = 5
#         n_hist_dec_head_2 = 4
#         n_hist_dec_layer_2 = 2
#         self.anchors = opt["anchors"]
#         self.history_tokens = 16
#         self.short_window_size = 16
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0
        
#         # Keep ALL original components exactly the same
#         self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   

#         self.history_encoder_block1 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
#                                       nhead=n_hist_dec_head, 
#                                       dropout=dropout, 
#                                       activation='gelu'), 
#             n_hist_dec_layer, 
#             nn.LayerNorm(n_embedding_dim))  
        
#         self.history_encoder_block2 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
#                                       nhead=n_hist_dec_head_2, 
#                                       dropout=dropout, 
#                                       activation='gelu'), 
#             n_hist_dec_layer_2, 
#             nn.LayerNorm(n_embedding_dim))  

#         self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim//4), nn.ReLU())     
#         self.snip_classifier = nn.Sequential(
#             nn.Linear(self.history_tokens*n_embedding_dim//4, (self.history_tokens*n_embedding_dim//4)//4), 
#             nn.ReLU(), 
#             nn.Linear((self.history_tokens*n_embedding_dim//4)//4, n_class))                      

#         self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
#         self.norm2 = nn.LayerNorm(n_embedding_dim)
#         self.dropout2 = nn.Dropout(0.1)
        
#         # Add VERY minimal enhancements with proper initialization
#         self.enhancement_enabled = True
        
#         # Residual gate - starts at 0 (no enhancement initially)
#         self.residual_gate = nn.Parameter(torch.zeros(1))
        
#         # Very lightweight feature refinement
#         self.feature_refine = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim),
#             nn.LayerNorm(n_embedding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(n_embedding_dim, n_embedding_dim)
#         )
        
#         # Initialize new components
#         self._init_enhancement_params()
    
#     def _init_enhancement_params(self):
#         """Initialize enhancement parameters to not interfere with original behavior"""
#         # Gate starts at 0 - no enhancement initially
#         nn.init.constant_(self.residual_gate, 0.0)
        
#         # Initialize feature refinement to identity mapping
#         with torch.no_grad():
#             # First linear layer - Xavier initialization
#             nn.init.xavier_uniform_(self.feature_refine[0].weight)
#             nn.init.constant_(self.feature_refine[0].bias, 0)
            
#             # Last linear layer - initialize to small values for identity-like behavior
#             nn.init.xavier_uniform_(self.feature_refine[4].weight, gain=0.01)
#             nn.init.constant_(self.feature_refine[4].bias, 0)

#     def forward(self, long_x, encoded_x):
#         ## Original History Encoder - UNCHANGED
#         hist_pe_x = self.history_positional_encoding(long_x)
#         history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
#         hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
#         hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
#         hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
#         hist_encoded_x = self.norm2(hist_encoded_x_2)
        
#         # Very minimal enhancement - only if gate allows it
#         if self.enhancement_enabled and torch.sigmoid(self.residual_gate) > 0.01:
#             gate_value = torch.sigmoid(self.residual_gate)
#             enhanced_features = self.feature_refine(hist_encoded_x)
#             hist_encoded_x = hist_encoded_x + gate_value * enhanced_features
   
#         ## Original Snippet Classification - UNCHANGED
#         snippet_feat = self.snip_head(hist_encoded_x_1)
#         snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
#         snip_cls = self.snip_classifier(snippet_feat)
        
#         return hist_encoded_x, snip_cls


# class StabilizedEncoder(nn.Module):
#     """Conservative encoder enhancement that maintains original performance"""
    
#     def __init__(self, n_embedding_dim, n_enc_head, n_enc_layer, dropout=0.3):
#         super(StabilizedEncoder, self).__init__()
        
#         # Keep original encoder as primary path
#         self.main_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
#                                       nhead=n_enc_head, 
#                                       dropout=dropout, 
#                                       activation='gelu'), 
#             n_enc_layer, 
#             nn.LayerNorm(n_embedding_dim))
        
#         # Add minimal enhancement - starts disabled
#         self.enhancement_gate = nn.Parameter(torch.zeros(1))
        
#         # Lightweight attention refinement
#         self.attention_refine = nn.MultiheadAttention(
#             embed_dim=n_embedding_dim,
#             num_heads=max(1, n_enc_head // 2),
#             dropout=dropout * 0.5,
#             batch_first=False
#         )
        
#         self.refine_norm = nn.LayerNorm(n_embedding_dim)
#         self.refine_dropout = nn.Dropout(dropout * 0.5)
        
#         # Initialize conservatively
#         self._init_params()
    
#     def _init_params(self):
#         nn.init.constant_(self.enhancement_gate, 0.0)  # Start with no enhancement
        
#     def forward(self, x):
#         # Primary encoding path - unchanged
#         main_output = self.main_encoder(x)
        
#         # Optional refinement - starts at 0 contribution
#         gate_value = torch.sigmoid(self.enhancement_gate)
#         if gate_value > 0.01:  # Only compute if gate is open enough
#             refined, _ = self.attention_refine(main_output, main_output, main_output)
#             refined = self.refine_norm(refined)
#             refined = self.refine_dropout(refined)
#             output = main_output + gate_value * refined
#         else:
#             output = main_output
            
#         return output


# class MYNET(torch.nn.Module):
#     def __init__(self, opt):
#         super(MYNET, self).__init__()
#         self.n_feature = opt["feat_dim"] 
#         n_class = opt["num_of_class"]
#         n_embedding_dim = opt["hidden_dim"]
#         n_enc_layer = opt["enc_layer"]
#         n_enc_head = opt["enc_head"]
#         n_dec_layer = opt["dec_layer"]
#         n_dec_head = opt["dec_head"]
#         n_comb_dec_head = 4
#         n_comb_dec_layer = 5
#         n_seglen = opt["segment_size"]
#         self.anchors = opt["anchors"]
#         self.history_tokens = 16
#         self.short_window_size = 16
#         self.anchors_stride = []
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0

#         # Original feature reduction - UNCHANGED
#         self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
#         self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
#         # Original positional encoding - UNCHANGED
#         self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
#         # Use stabilized encoder with fallback
#         try:
#             self.encoder = StabilizedEncoder(n_embedding_dim, n_enc_head, n_enc_layer, dropout)
#             print("Using stabilized encoder")
#         except Exception as e:
#             print(f"Fallback to original encoder: {e}")
#             self.encoder = nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
#                                           nhead=n_enc_head, 
#                                           dropout=dropout, 
#                                           activation='gelu'), 
#                 n_enc_layer, 
#                 nn.LayerNorm(n_embedding_dim))
                                            
#         # Original decoder - UNCHANGED
#         self.decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
#                                       nhead=n_dec_head, 
#                                       dropout=dropout, 
#                                       activation='gelu'), 
#             n_dec_layer, 
#             nn.LayerNorm(n_embedding_dim))  

#         # Use stabilized history unit
#         self.history_unit = StabilizedHistoryUnit(opt)

#         # Original components - UNCHANGED
#         self.history_anchor_decoder_block1 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
#                                       nhead=n_comb_dec_head, 
#                                       dropout=dropout, 
#                                       activation='gelu'), 
#             n_comb_dec_layer, 
#             nn.LayerNorm(n_embedding_dim))  

#         self.classifier = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, n_class))
#         self.regressor = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 2))    
                           
#         self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
#         self.norm1 = nn.LayerNorm(n_embedding_dim)
#         self.dropout1 = nn.Dropout(0.1)
#         self.relu = nn.ReLU(True)
#         self.softmaxd1 = nn.Softmax(dim=-1)
        
#         # Add gradient clipping hook for stability
#         self.register_backward_hook(self._gradient_clipping_hook)
        
#     def _gradient_clipping_hook(self, module, grad_input, grad_output):
#         """Clip gradients to prevent instability"""
#         if grad_output[0] is not None:
#             torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

#     def forward(self, inputs):
#         # Original preprocessing - UNCHANGED
#         base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
#         base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
#         base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
#         base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

#         short_x = base_x[-self.short_window_size:]
#         long_x = base_x[:-self.short_window_size]
        
#         ## Anchor Feature Generator - with stabilized encoder
#         pe_x = self.positional_encoding(short_x)
#         encoded_x = self.encoder(pe_x)   
#         decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
#         decoded_x = self.decoder(decoder_token, encoded_x) 

#         ## Enhanced History Module
#         hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

#         ## Original anchor refinement - UNCHANGED
#         decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
#         decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
#         decoded_anchor_feat = self.norm1(decoded_anchor_feat)
#         decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
#         # Original prediction modules - UNCHANGED
#         anc_cls = self.classifier(decoded_anchor_feat)
#         anc_reg = self.regressor(decoded_anchor_feat)
        
#         return anc_cls, anc_reg, snip_cls

 
# class SuppressNet(torch.nn.Module):
#     """Improved SuppressNet with better stability"""
    
#     def __init__(self, opt):
#         super(SuppressNet, self).__init__()
#         n_class = opt["num_of_class"] - 1
#         n_seglen = opt["segment_size"]
#         n_embedding_dim = 2 * n_seglen
#         dropout = 0.3
#         self.best_loss = 1000000
#         self.best_map = 0
        
#         # Original components
#         self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
#         self.mlp2 = nn.Linear(n_embedding_dim, 1)
#         self.norm = nn.InstanceNorm1d(n_class)
#         self.relu = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()
        
#         # Add minimal improvements for stability
#         self.dropout = nn.Dropout(dropout * 0.5)  # Conservative dropout
        
#         # Enhancement gate - starts disabled
#         self.enhancement_gate = nn.Parameter(torch.zeros(1))
        
#         # Optional batch normalization
#         self.use_bn = True
#         if self.use_bn:
#             self.bn = nn.BatchNorm1d(n_class)
        
#         # Initialize
#         self._init_params()
        
#     def _init_params(self):
#         nn.init.constant_(self.enhancement_gate, 0.0)
        
#     def forward(self, inputs):
#         # inputs - batch x seq_len x class
#         base_x = inputs.permute([0,2,1])
        
#         # Optional batch norm before instance norm
#         if self.use_bn and torch.sigmoid(self.enhancement_gate) > 0.1:
#             base_x = self.bn(base_x)
        
#         base_x = self.norm(base_x)
        
#         x = self.relu(self.mlp1(base_x))
#         x = self.dropout(x)  # Add dropout for regularization
#         x = self.sigmoid(self.mlp2(x))
#         x = x.squeeze(-1)
        
#         return x


# # Training helper functions for better optimization
# class ModelOptimizer:
#     """Helper class for better training dynamics"""
    
#     @staticmethod
#     def get_optimizer(model, lr=1e-4, weight_decay=1e-5):
#         """Get optimizer with different learning rates for different components"""
#         param_groups = []
        
#         # Original parameters - normal learning rate
#         original_params = []
#         enhancement_params = []
        
#         for name, param in model.named_parameters():
#             if any(x in name for x in ['enhancement_gate', 'residual_gate', 'refine']):
#                 enhancement_params.append(param)
#             else:
#                 original_params.append(param)
        
#         # Original parameters get normal LR
#         if original_params:
#             param_groups.append({
#                 'params': original_params,
#                 'lr': lr,
#                 'weight_decay': weight_decay
#             })
        
#         # Enhancement parameters get lower LR initially
#         if enhancement_params:
#             param_groups.append({
#                 'params': enhancement_params,
#                 'lr': lr * 0.1,  # 10x lower LR for enhancements
#                 'weight_decay': weight_decay * 0.1
#             })
        
#         return torch.optim.AdamW(param_groups)
    
#     @staticmethod
#     def warmup_enhancements(model, epoch, warmup_epochs=5):
#         """Gradually enable enhancements after warmup period"""
#         if epoch >= warmup_epochs:
#             # Gradually increase enhancement gates
#             for name, param in model.named_parameters():
#                 if 'enhancement_gate' in name or 'residual_gate' in name:
#                     with torch.no_grad():
#                         # Slowly increase from 0 to small positive value
#                         target_val = min(0.1, (epoch - warmup_epochs) * 0.02)
#                         param.data = torch.tensor(target_val).to(param.device)








#Score 0.1756  Final score is 0.22

import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


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


class MultiScaleTemporalEncoder(nn.Module):
    """Improved multi-scale encoder with better initialization and conservative approach"""
    
    def __init__(self, n_embedding_dim, n_enc_head, n_enc_layer, dropout=0.3):
        super(MultiScaleTemporalEncoder, self).__init__()
        self.n_embedding_dim = n_embedding_dim
        
        # Main encoder - exactly like original but with better initialization
        self.main_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_enc_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        
        # Multi-scale branches with reduced complexity
        # Short-term (local) attention - focuses on adjacent frames
        self.short_conv = nn.Conv1d(
            n_embedding_dim, n_embedding_dim // 4, 
            kernel_size=3, padding=1, groups=n_embedding_dim // 8
        )
        
        # Medium-term attention - focuses on medium-range dependencies  
        self.medium_conv = nn.Conv1d(
            n_embedding_dim, n_embedding_dim // 4,
            kernel_size=5, padding=2, groups=n_embedding_dim // 8
        )
        
        # Long-term attention - focuses on long-range dependencies
        self.long_conv = nn.Conv1d(
            n_embedding_dim, n_embedding_dim // 4,
            kernel_size=7, padding=3, groups=n_embedding_dim // 8
        )
        
        # Fusion layer to combine multi-scale features
        self.scale_fusion = nn.Linear(n_embedding_dim // 4 * 3, n_embedding_dim // 4)
        
        # Learnable weights for combining scales (start very small)
        self.scale_weights = nn.Parameter(torch.tensor([0.01, 0.01, 0.01]))
        self.fusion_weight = nn.Parameter(torch.tensor(0.05))
        
        # Normalization and dropout
        self.scale_norm = nn.LayerNorm(n_embedding_dim // 4)
        self.scale_dropout = nn.Dropout(dropout * 0.5)
        
        # Initialize all new parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Conservative initialization to preserve original performance"""
        # Initialize convolutions
        for conv in [self.short_conv, self.medium_conv, self.long_conv]:
            nn.init.xavier_uniform_(conv.weight, gain=0.1)  # Small gain
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        # Initialize fusion layer
        nn.init.xavier_uniform_(self.scale_fusion.weight, gain=0.1)
        nn.init.constant_(self.scale_fusion.bias, 0)
        
        # Initialize weights to be very small initially
        nn.init.constant_(self.scale_weights, 0.01)
        nn.init.constant_(self.fusion_weight, 0.05)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (seq_len, batch, n_embedding_dim)
        """
        # Main encoding path (preserve original behavior)
        main_encoded = self.main_encoder(x)
        
        # Multi-scale processing on residual path only
        # Convert to (batch, dim, seq) for conv operations
        x_conv = x.permute(1, 2, 0)  # (batch, dim, seq)
        
        # Apply multi-scale convolutions
        short_feat = self.short_conv(x_conv)
        medium_feat = self.medium_conv(x_conv)  
        long_feat = self.long_conv(x_conv)
        
        # Combine scales
        multi_scale = torch.cat([short_feat, medium_feat, long_feat], dim=1)
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, seq, dim)
        
        # Fusion
        fused_scales = self.scale_fusion(multi_scale)
        fused_scales = self.scale_norm(fused_scales)
        fused_scales = self.scale_dropout(fused_scales)
        fused_scales = fused_scales.permute(1, 0, 2)  # back to (seq, batch, dim)
        
        # Conservative combination with very small weights
        fusion_weight = torch.sigmoid(self.fusion_weight)
        if fusion_weight > 0.01:  # Only apply if weight is meaningful
            # Expand fused features to match main encoder dimensions
            fused_expanded = torch.cat([fused_scales] * 4, dim=-1)  # Expand to full dim
            output = main_encoded + fusion_weight * fused_expanded
        else:
            output = main_encoded
            
        return output


class EnhancedHistoryUnit(torch.nn.Module):
    """Improved HistoryUnit with cross-attention and better temporal modeling"""
    
    def __init__(self, opt):
        super(EnhancedHistoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_hist_dec_head = 4
        n_hist_dec_layer = 5
        n_hist_dec_head_2 = 4
        n_hist_dec_layer_2 = 2
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Keep ALL original components exactly the same
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   
        
        self.history_encoder_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_hist_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_hist_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )  
        
        self.history_encoder_block2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_hist_dec_head_2, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_hist_dec_layer_2, 
            nn.LayerNorm(n_embedding_dim)
        )  
        
        self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim//4), nn.ReLU())     
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim//4, (self.history_tokens * n_embedding_dim//4)//4), 
            nn.ReLU(), 
            nn.Linear((self.history_tokens * n_embedding_dim//4)//4, n_class)
        )                      
        
        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # Add cross-attention between short and long term features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=n_embedding_dim,
            num_heads=2,  # Fewer heads for stability
            dropout=dropout * 0.5,
            batch_first=False
        )
        
        # Temporal consistency module
        self.temporal_gate = nn.Sequential(
            nn.Linear(n_embedding_dim * 2, n_embedding_dim),
            nn.Sigmoid()
        )
        
        # Learnable mixing weights (start very small)
        self.cross_attn_weight = nn.Parameter(torch.tensor(0.02))
        self.temporal_weight = nn.Parameter(torch.tensor(0.03))
        
        # Additional normalization
        self.cross_norm = nn.LayerNorm(n_embedding_dim)
        self.cross_dropout = nn.Dropout(dropout * 0.5)
        
        # Initialize new parameters
        self._init_new_params()
        
    def _init_new_params(self):
        """Initialize new parameters conservatively"""
        # Initialize cross attention properly
        if hasattr(self.cross_attention, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.cross_attention.in_proj_weight, gain=0.1)
        if hasattr(self.cross_attention, 'out_proj'):
            nn.init.xavier_uniform_(self.cross_attention.out_proj.weight, gain=0.1)
            nn.init.constant_(self.cross_attention.out_proj.bias, 0)
            
        # Initialize temporal gate
        for layer in self.temporal_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize weights to be very small
        nn.init.constant_(self.cross_attn_weight, 0.02)
        nn.init.constant_(self.temporal_weight, 0.03)

    def forward(self, long_x, encoded_x):
        # Original processing path - keep exactly the same
        hist_pe_x = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
        
        # Cross-attention enhancement (conservative)
        cross_weight = torch.sigmoid(self.cross_attn_weight)
        if cross_weight > 0.01 and encoded_x.size(0) > 0:  # Safety check
            try:
                # Cross-attention between history tokens and short-term features
                cross_enhanced, _ = self.cross_attention(
                    hist_encoded_x_1, 
                    encoded_x, 
                    encoded_x
                )
                cross_enhanced = self.cross_norm(cross_enhanced)
                cross_enhanced = self.cross_dropout(cross_enhanced)
                
                # Temporal gating for selective enhancement
                gate_input = torch.cat([hist_encoded_x_1, cross_enhanced], dim=-1)
                gate_input_2d = gate_input.view(-1, gate_input.size(-1))
                temporal_gate = self.temporal_gate(gate_input_2d)
                temporal_gate = temporal_gate.view_as(hist_encoded_x_1)
                
                # Apply gated enhancement
                temp_weight = torch.sigmoid(self.temporal_weight)
                gated_enhancement = temporal_gate * cross_enhanced * temp_weight
                hist_encoded_x_1 = hist_encoded_x_1 + cross_weight * gated_enhancement
                
            except Exception as e:
                # Fallback to original if enhancement fails
                print(f"Cross-attention failed, using original: {e}")
                pass
        
        # Continue with original processing
        hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
        hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
        hist_encoded_x = self.norm2(hist_encoded_x_2)
   
        # Original snippet classification - keep exactly the same
        snippet_feat = self.snip_head(hist_encoded_x_1)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0

        # Keep original feature reduction exactly the same
        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        # Keep original positional encoding
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        # Use multi-scale encoder with fallback
        self.use_enhanced_encoder = True
        try:
            self.encoder = MultiScaleTemporalEncoder(
                n_embedding_dim=n_embedding_dim,
                n_enc_head=n_enc_head,
                n_enc_layer=n_enc_layer,
                dropout=dropout
            )
        except Exception as e:
            print(f"Enhanced encoder initialization failed, using original: {e}")
            self.use_enhanced_encoder = False
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=n_embedding_dim, 
                    nhead=n_enc_head, 
                    dropout=dropout, 
                    activation='gelu'
                ), 
                n_enc_layer, 
                nn.LayerNorm(n_embedding_dim)
            )
        
        # Keep original decoder exactly the same
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )  

        # Use enhanced history unit
        self.history_unit = EnhancedHistoryUnit(opt)

        # Keep ALL other original components exactly the same
        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_comb_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_comb_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )  
            
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(n_embedding_dim, n_class)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(n_embedding_dim, 2)
        )    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)
        
        # Proper parameter initialization
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters to maintain stability and original performance"""
        # Initialize decoder token exactly like original
        nn.init.normal_(self.decoder_token, std=0.02)
        
        # Initialize feature reduction layers like original
        nn.init.xavier_uniform_(self.feature_reduction_rgb.weight)
        nn.init.constant_(self.feature_reduction_rgb.bias, 0)
        nn.init.xavier_uniform_(self.feature_reduction_flow.weight)
        nn.init.constant_(self.feature_reduction_flow.bias, 0)
        
        # Initialize classifier and regressor like original
        for module in [self.classifier, self.regressor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        # Exactly same preprocessing as original
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]
        
        # Anchor feature generation
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)
        
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 

        # Enhanced history processing
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # Exactly same as original final processing
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Same prediction modules
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    """Optimized SuppressNet with better training stability"""
    
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class = opt["num_of_class"] - 1
        n_seglen = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Keep original architecture as the main path
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
        # Add minimal enhancements for stability
        self.layer_norm = nn.LayerNorm(n_embedding_dim)
        self.dropout = nn.Dropout(dropout * 0.5)  # Less aggressive dropout
        
        # Residual connection weight (start small)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize properly
        self._init_parameters()
        
    def _init_parameters(self):
        """Proper initialization for stability"""
        nn.init.xavier_uniform_(self.mlp1.weight, gain=1.0)
        nn.init.constant_(self.mlp1.bias, 0)
        nn.init.xavier_uniform_(self.mlp2.weight, gain=1.0)
        nn.init.constant_(self.mlp2.bias, 0)
        nn.init.constant_(self.residual_weight, 0.1)
        
    def forward(self, inputs):
        # inputs - batch x seq_len x class
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        
        # Main processing path
        x = self.relu(self.mlp1(base_x))
        
        # Add layer normalization and dropout for stability
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x