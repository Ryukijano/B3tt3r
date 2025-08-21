import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import partial
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.geometry import inv, geotrf

class SpatialMemory:
    """
    Sophisticated spatial memory implementation inspired by SPANNer3R
    Supports working memory, long-term memory, similarity checking, and memory pruning
    """
    def __init__(self, norm_q, norm_k, norm_v, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False, num_patches=None):
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.init_mem()
    
    def init_mem(self):
        """Initialize memory storage"""
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None
        self.lm = 0  # long-term memory counter
        self.wm = 0  # working memory counter
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        """Add key features to memory"""
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)
        return self.mem_k
    
    def add_mem_v(self, feat):
        """Add value features to memory"""
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)
        return self.mem_v

    def add_mem_c(self, feat):
        """Add confidence values to memory"""
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)
        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        """Add 3D points to memory"""
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        """Add image features to memory"""
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    def add_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):
        """Add features to memory with counters"""
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count += 1
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.add_mem_k(feat_k)
        self.add_mem_v(feat_v)
        self.add_mem_pts(pts_cur)
        self.add_mem_img(img_cur)
    
    def check_sim(self, feat_k, thresh=0.7):
        """Check similarity with working memory to avoid redundant features"""
        if self.mem_k is None or thresh == 1.0:
            return False
        
        wmem_size = self.wm * self.num_patches
        if wmem_size == 0 or wmem_size > self.mem_k.shape[1]:
            return False

        # Get working memory features
        wm_feat = self.mem_k[:, -wmem_size:]
        
        # Reshape only if dimensions match
        if wmem_size % self.num_patches == 0:
            wm = wm_feat.reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])
        else:
            # If reshape not possible, use direct comparison
            wm = wm_feat.unsqueeze(1)

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        
        if wm.dim() == 4:  # Properly reshaped
            wm_norm = F.normalize(wm, p=2, dim=-1)
            corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)
            mean_corr = torch.mean(corr, dim=-1)
        else:  # Direct comparison
            wm_norm = F.normalize(wm, p=2, dim=-1)
            corr = torch.einsum('bpc,btc->bpt', feat_k_norm, wm_norm)
            mean_corr = torch.mean(corr, dim=-2)

        if mean_corr.max() > thresh:
            return True
        return False

    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None):
        """Add memory with similarity checking and pruning"""
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]

        if self.check_sim(feat_k, thresh=self.sim_thresh):
            return
        
        self.add_mem(feat_k, feat_v, pts_cur, img_cur)
        self.wm += 1

        if self.wm > self.work_mem_size:
            self.wm -= 1
            if self.long_mem_size == 0:
                # Remove oldest memory if no long-term memory
                self.mem_k = self.mem_k[:, self.num_patches:]
                self.mem_v = self.mem_v[:, self.num_patches:]
                self.mem_count = self.mem_count[:, self.num_patches:]
                self.mem_attn = self.mem_attn[:, self.num_patches:]
            else:
                self.lm += self.num_patches
        
        if self.lm > self.long_mem_size:
            self.memory_prune()
            self.lm = self.top_k - self.wm * self.num_patches
    
    def memory_read(self, feat, res=True):
        """Read features from memory using attention mechanism"""
        if self.mem_k is None:
            return feat
            
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), 
                               self.norm_k(self.mem_k.reshape(self.mem_k.shape[0], -1, self.mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        
        if self.mem_c is not None:
            affinity = affinity * self.mem_c.view(self.mem_c.shape[0], 1, -1)  
        
        attn = torch.softmax(affinity, dim=-1)

        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)
                
        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn < self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True) 
        
        out = torch.einsum('bpx,bxc->bpc', attn, 
                          self.norm_v(self.mem_v.reshape(self.mem_v.shape[0], -1, self.mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        # Update attention statistics
        total_attn = torch.sum(attn, dim=-2)
        if self.mem_attn is not None:
            self.mem_attn += total_attn[..., None]
        
        return out
    
    def memory_prune(self):
        """Prune long-term memory based on attention weights"""
        if self.mem_attn is None or self.mem_count is None:
            return
            
        weights = self.mem_attn / (self.mem_count + 1e-8)
        weights[self.mem_count < self.work_mem_size + 5] = 1e8

        top_k_values, top_k_indices = torch.topk(weights, min(self.top_k, weights.shape[1]), dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))

        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)

        if self.mem_pts is not None:
            top_k_indices_pts = top_k_indices.unsqueeze(-1).expand(-1, -1, self.mem_pts.shape[-2], self.mem_pts.shape[-1])
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_pts)
            
        if self.mem_imgs is not None:
            top_k_indices_imgs = top_k_indices.unsqueeze(-1).expand(-1, -1, self.mem_imgs.shape[-2], self.mem_imgs.shape[-1])
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_imgs)

class Bett3R(AsymmetricMASt3R):
    """
    BETT3R model: Combines MASt3R's stereo vision capabilities with SPANNer3R's spatial memory
    Unlike SPANNer3R, Bett3R doesn't require temporal ordering of images
    """
    def __init__(self, *args, use_feat=False, mem_pos_enc=False, memory_dropout=0.15, 
                 long_mem_size=4000, work_mem_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Memory configuration
        self.use_feat = use_feat
        self.mem_pos_enc = mem_pos_enc
        
        # Initialize memory encoder components similar to SPANNer3R
        self.set_memory_encoder(memory_dropout=memory_dropout)
        self.set_attn_head()
        
        # Initialize spatial memory with sophisticated mechanisms
        self.spatial_memory = None  # Will be initialized during forward pass
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.memory_dropout = memory_dropout

    def set_memory_encoder(self, enc_depth=6, enc_embed_dim=1024, out_dim=1024, 
                          enc_num_heads=16, mlp_ratio=4, memory_dropout=0.15):
        """Initialize memory encoding components"""
        from croco.models.blocks import Block
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # Value encoder for processing features before storing in memory
        self.value_encoder = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, rope=getattr(self, 'rope', None) if self.mem_pos_enc else None)
            for i in range(enc_depth)])
        
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)
        
        # Normalization layers for memory operations
        self.norm_q = nn.LayerNorm(out_dim)
        self.norm_k = nn.LayerNorm(out_dim)
        self.norm_v = nn.LayerNorm(out_dim)
        self.mem_dropout = nn.Dropout(memory_dropout) if memory_dropout > 0 else None
        
    def set_attn_head(self, enc_embed_dim=1024+768, out_dim=1024):
        """Initialize attention heads for feature encoding"""
        self.attn_head_1 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )
        
        self.attn_head_2 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim), 
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

    def encode_value(self, x, pos):
        """Encode features for memory storage"""
        for block in self.value_encoder:
            x = block(x, pos)
        x = self.value_norm(x)
        x = self.value_out(x)
        return x

    def encode_feat_key(self, feat1, feat2, num=1):
        """Encode feature keys for memory operations"""
        feat = torch.cat((feat1, feat2), dim=-1)
        feat_k = getattr(self, f'attn_head_{num}')(feat)
        return feat_k

    def forward(self, view1, view2, return_memory=False):
        """
        Forward pass processing image pairs with spatial memory enhancement
        
        Args:
            view1, view2: Input image views (dict with 'img' key)
            return_memory: Whether to return memory state
            
        Returns:
            Predictions with memory-enhanced features
        """
        # Initialize spatial memory if not exists
        if self.spatial_memory is None:
            self.spatial_memory = SpatialMemory(
                self.norm_q, self.norm_k, self.norm_v, 
                mem_dropout=self.mem_dropout,
                long_mem_size=self.long_mem_size,
                work_mem_size=self.work_mem_size
            )

        # Extract images and shapes
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # Encode image pairs using MASt3R backbone
        feat1, pos1, _ = self._encode_image(img1, shape1)
        feat2, pos2, _ = self._encode_image(img2, shape2)

        # Decode features using MASt3R decoder
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        
        # Generate predictions using MASt3R heads
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # Encode features for memory operations
        feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
        feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)
        
        # Encode values for memory storage
        cur_v1 = self.encode_value(dec1[-1], pos1)
        cur_v2 = self.encode_value(dec2[-1], pos2)
        
        # Memory enhancement: read from memory before storing
        if self.spatial_memory.mem_k is not None:
            feat_enhanced1 = self.spatial_memory.memory_read(feat_k1, res=True)
            feat_enhanced2 = self.spatial_memory.memory_read(feat_k2, res=True)
            
            # Re-decode with memory-enhanced features
            dec1_enhanced, dec2_enhanced = self._decoder(feat_enhanced1, pos1, feat_enhanced2, pos2)
            
            # Generate enhanced predictions
            with torch.cuda.amp.autocast(enabled=False):
                res1_enhanced = self._downstream_head(1, [tok.float() for tok in dec1_enhanced], shape1)
                res2_enhanced = self._downstream_head(2, [tok.float() for tok in dec2_enhanced], shape2)
                
            # Use enhanced predictions
            res1, res2 = res1_enhanced, res2_enhanced

        # Update spatial memory with current features
        if not self.training:
            # Use similarity checking during inference
            self.spatial_memory.add_mem_check(feat_k1, cur_v1 + feat_k1)
            self.spatial_memory.add_mem_check(feat_k2, cur_v2 + feat_k2)
        else:
            # Add directly during training
            self.spatial_memory.add_mem(feat_k1, cur_v1 + feat_k1)
            self.spatial_memory.add_mem(feat_k2, cur_v2 + feat_k2)

        if return_memory:
            return res1, res2, self.spatial_memory
        
        return res1, res2

    def reset_memory(self):
        """Reset spatial memory - useful between different scenes"""
        if self.spatial_memory is not None:
            self.spatial_memory.init_mem()