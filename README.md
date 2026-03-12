Plane-Aware Structure-from-Motion
=============
This repository implements Plane-Aware Structure-from-Motion (PASfM) on top of a COLMAP reconstruction. The method improves geometric consistency by leveraging dominant planar structures in the scene.

The pipeline works as follows:

- Plane Detection:
Planar correspondences are detected from image pairs using homography estimation with RANSAC, and merged into global planes using shared feature tracks.

- Plane Refinement: 
3D points assigned to each plane are robustly fitted with RANSAC + SVD to estimate accurate plane parameters.

- Dynamic Snap Optimization:
Plane-associated points are iteratively moved toward their planes while monitoring reprojection error, preventing instability during optimization.

- Iterative Refinement: 
The system alternates between bundle adjustment and plane snapping, improving planar alignment while preserving camera pose quality.
