# Cardboard Quality Control System

A comprehensive quality control system for cardboard inspection using computer vision and machine learning techniques. This project was developed as part of the **Advanced Measurement Systems for Control Applications** course at Politecnico di Milano.

## 🎯 Overview

This system performs automated quality control inspection of cardboard pieces through two main analysis modules:

1. **Contour Quality Check**: Analyzes cardboard contours, vertices, and internal cuts against golden samples
2. **Surface Defect Detection**: AI-powered surface defect detection using Few-Shot Learning with Vision Transformers

## ✨ Features

### Contour Analysis
- Real-time cardboard contour detection
- Vertex detection based on slope changes
- Internal cut/hole analysis
- Shape comparison with golden samples
- Defect localization and classification

### Surface Defect Detection
- Few-Shot Learning with Vision Transformers (ViT)
- Prototypical Networks for classification
- Real-time inference with confidence scores
- Synthetic defect generation for training augmentation
- Data augmentation with geometric and photometric transformations

## 🚀 How to Use

1. **Contour Check**: Navigate through test images and click "Analyze Contours"
2. **Surface Check**: Navigate through test images and click "Analyze Surface"
3. View detailed analysis results and visualizations
4. Check the "System Information" tab for model details

## 🔧 Technical Details

- **Architecture**: Vision Transformer (ViT-Base) + Prototypical Networks
- **Training**: Few-shot learning with data augmentation
- **Contour Detection**: Multi-threshold detection with morphological operations
- **Real-time Processing**: Optimized for fast inference

## 👥 Authors

- **Mattia Ogliar Badessi**
- **Luca Molettieri**

**Course**: Advanced Measurement Systems for Control Applications
**Institution**: Politecnico di Milano
**Academic Year**: 2024-2025