import numpy as np

def compute_homography_3d(pts_pred, pts_gt):
    # pts_pred, pts_gt: [3, N]
    N = pts_pred.shape[1]
    A = np.zeros((3 * N, 16))
    
    for i in range(N):
        X = np.append(pts_pred[:, i], 1.0)
        x_p, y_p, z_p = pts_gt[:, i]
        
        A[3*i, 0:4] = -X
        A[3*i, 12:16] = x_p * X
        
        A[3*i+1, 4:8] = -X
        A[3*i+1, 12:16] = y_p * X
        
        A[3*i+2, 8:12] = -X
        A[3*i+2, 12:16] = z_p * X
        
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(4, 4)
    if H[3, 3] < 0:
        H = -H
    return H

def ransac_homography_3d(pts_pred, pts_gt, num_iters=500, threshold=0.1):
    N = pts_pred.shape[1]
    best_inliers = []
    best_H = None
    
    for _ in range(num_iters):
        # Randomly sample 5 points
        idx = np.random.choice(N, 5, replace=False)
        pts_pred_sample = pts_pred[:, idx]
        pts_gt_sample = pts_gt[:, idx]
        
        H = compute_homography_3d(pts_pred_sample, pts_gt_sample)
        
        # Evaluate all points
        pts_pred_homo = np.vstack((pts_pred, np.ones((1, N))))
        pts_est_homo = H @ pts_pred_homo
        pts_est = pts_est_homo[:3, :] / (pts_est_homo[3, :] + 1e-8)
        
        errors = np.linalg.norm(pts_est - pts_gt, axis=0)
        inliers = np.where(errors < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            
    # Final refinement using all best inliers
    if len(best_inliers) >= 5:
        best_H = compute_homography_3d(pts_pred[:, best_inliers], pts_gt[:, best_inliers])
        
    return best_H, best_inliers

# Test with synthetic data
np.random.seed(0)
pts_pred = np.random.rand(3, 100) * 10
H_true = np.random.rand(4, 4)
pts_gt_homo = H_true @ np.vstack((pts_pred, np.ones(100)))
pts_gt = pts_gt_homo[:3] / pts_gt_homo[3]

# Add outliers
pts_gt[:, :20] += np.random.rand(3, 20) * 5

H_est, inliers = ransac_homography_3d(pts_pred, pts_gt)

pts_est_homo = H_est @ np.vstack((pts_pred, np.ones(100)))
pts_est = pts_est_homo[:3] / pts_est_homo[3]

err = np.linalg.norm(pts_est[:, inliers] - pts_gt[:, inliers], axis=0).mean()
print(f"DLT RANSAC Error (on inliers): {err:.6f}")
print(f"Num Inliers: {len(inliers)}/100")
