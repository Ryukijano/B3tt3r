import numpy as np
from scipy.spatial import cKDTree as KDTree
import open3d as o3d

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median

def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)
    
    return comp, comp_median

def compute_iou(pred_vox, target_vox):
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    iou = len(intersection) / len(union)
    return iou

def evaluate_combined_model(gt_points, rec_points, gt_normals=None, rec_normals=None, pred_vox=None, target_vox=None):
    comp_ratio = completion_ratio(gt_points, rec_points)
    acc, acc_median, normal_dot_mean, normal_dot_median = accuracy(gt_points, rec_points, gt_normals, rec_normals)
    comp, comp_median, comp_normal_dot_mean, comp_normal_dot_median = completion(gt_points, rec_points, gt_normals, rec_normals)
    iou = compute_iou(pred_vox, target_vox) if pred_vox is not None and target_vox is not None else None

    return {
        "completion_ratio": comp_ratio,
        "accuracy": acc,
        "accuracy_median": acc_median,
        "normal_dot_mean": normal_dot_mean,
        "normal_dot_median": normal_dot_median,
        "completion": comp,
        "completion_median": comp_median,
        "completion_normal_dot_mean": comp_normal_dot_mean,
        "completion_normal_dot_median": comp_normal_dot_median,
        "iou": iou
    }
