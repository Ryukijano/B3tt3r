import torch
import torch.nn as nn
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.geometry import inv, geotrf

class SpatialMemory(nn.Module):
    """Implements spatial memory for storing and retrieving 3D features"""
    def __init__(self, feature_dim=256, memory_size=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.position_embeddings = nn.Parameter(torch.randn(memory_size, 3))

    def update(self, features, positions):
        """Update memory with new features at given 3D positions"""
        # Normalize positions
        positions_norm = positions / positions.norm(dim=-1, keepdim=True)
        embeddings_norm = self.position_embeddings / self.position_embeddings.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        sim = positions_norm @ embeddings_norm.T
        attn = torch.softmax(sim, dim=-1)

        # Update memory features through attention
        self.memory = attn @ features
        self.position_embeddings = attn @ positions

        return self.memory

    def query(self, query_positions):
        """Retrieve features from memory at query positions"""
        sim = torch.cdist(query_positions, self.position_embeddings) 
        attn = torch.softmax(-sim, dim=-1)
        retrieved = attn @ self.memory
        return retrieved

class Bett3R(AsymmetricMASt3R):
    """BETT3R model extending MASt3R with spatial memory"""
    def __init__(self, *args, memory_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize spatial memory
        self.spatial_memory = SpatialMemory(
            feature_dim=self.encoder.embed_dim,
            memory_size=memory_size
        )

    def forward(self, batch):
        """Forward pass with spatial memory integration"""
        # Get base features from MASt3R
        base_feats = super().forward(batch)
        
        # Extract camera poses and 3D points
        poses = batch['camera_pose']
        pts3d = batch['pts3d']

        # Transform points to world coordinates
        world_pts = []
        world_feats = []
        for i in range(len(poses)):
            pose = poses[i]
            pts = pts3d[i]
            # Transform to world space
            world_pt = geotrf(inv(pose), pts)
            world_pts.append(world_pt)
            
            # Get corresponding features
            feat = base_feats[i]
            world_feats.append(feat)

        world_pts = torch.cat(world_pts, dim=0)
        world_feats = torch.cat(world_feats, dim=0)

        # Update spatial memory
        memory_feats = self.spatial_memory.update(world_feats, world_pts)
        
        # Enhance base features with memory
        enhanced_feats = []
        for i in range(len(poses)):
            pose = poses[i]
            pts = pts3d[i]
            
            # Query memory at current points
            retrieved = self.spatial_memory.query(pts)
            
            # Combine with base features
            enhanced = base_feats[i] + retrieved
            enhanced_feats.append(enhanced)

        return enhanced_feats