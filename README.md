# ERP
# Smart Home Fall Detection System

##  Introduction
This project focuses on the development of an elderly fall detection system using computer vision and machine learning techniques. 
It explores multiple approaches including:
- **3D Convolutional Neural Networks (3DCNN)** for spatio-temporal feature extraction from video frames.
- **Skeleton-based features** (keypoints and joint angles) combined with unsupervised clustering.
- **Support Vector Machine (SVM)** for supervised classification of pose-based features.

The work is part of a Master's thesis on intelligent elderly care and safety monitoring.

---

##  Project Structure
  Train 3DCNN model
  │── 1.3DCNN_model.py 
  │── 2.Only_Mediapipe.py 
  │── 3.Model_mediepipe.py 
  │── 3DCNN_performance.py
  Cluster analysis of key points of posture
  │── 4.1 Extract_keypoint.py
  │── 4.2 PCA_visualization.py
  │── 4.3 Clustering_evaluation.py
  Cluster analysis of Angle + Time of posture
  │── 5.1 Angles_per_frame.py
  │── 5.2 Time_window_features.py
  │── 5.3 Time_Clustering.py
  SVM classification
  │── SVM_classification,py
  
  │── README.md # Project description (this file)
