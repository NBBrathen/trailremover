# Trail Remover

> **Automated satellite trail detection and removal for astronomical images using deep learning**

A production-ready system that uses a U-Net neural network to automatically detect and remove satellite trails from FITS astronomical images, restoring affected pixels using advanced inpainting algorithms.

## Problem Statement

Satellite trails are a growing problem in astrophotography, appearing as bright streaks across long-exposure images. Manual removal is time-consuming and requires expertise. Trail Remover automates this process using AI.

## Key Features

- **AI-Powered Detection**: U-Net neural network with 31M parameters
- **Automatic Removal**: Seamless pixel restoration using inpainting
- **Production-Ready**: Complete REST API with asynchronous processing
- **Batch Processing**: Handle multiple images efficiently
- **Format Support**: Works with FITS astronomical image format

## Technical Stack

**Machine Learning:**
- PyTorch 2.0+ - Deep learning framework
- U-Net architecture - Semantic segmentation
- Custom dataset loader with augmentation
- IoU metrics for evaluation

**Backend:**
- FastAPI - Modern REST API framework
- Astropy - FITS file handling
- OpenCV & scikit-image - Image processing
- Pydantic - Data validation

**Processing:**
- Tile-based inference for large images
- Multiple inpainting algorithms (Telea, Navier-Stokes, Biharmonic)
- GPU acceleration support
- Asynchronous job management

## Performance

- **Detection**: 25% validation IoU score
- **Confidence**: Up to 85% on real astronomical images
- **Speed**: ~5-30 seconds per image (GPU)
- **Scale**: Successfully processed 3800+ pixel trails

## ML Pipeline

### Training Data
- 70 labeled PNG images with JSON annotations
- Data augmentation (flips, rotations, noise, brightness/contrast)
- Train/validation split (80/20)

### Model Architecture
- **Network**: U-Net with skip connections
- **Parameters**: 31 million trainable parameters
- **Input**: 512×512 RGB patches
- **Output**: Binary segmentation mask
- **Loss**: Combined BCE + Dice Loss

### Training Results
```
Epoch   Train Loss   Val Loss   Val IoU
  1       0.3254      0.3189     0.2%
 50       0.0891      0.0856    12.4%
100       0.0423      0.0398    25.1%
```

## API Endpoints

```
POST   /api/v1/images/upload          - Upload FITS image
GET    /api/v1/jobs/{job_id}          - Check job status
GET    /api/v1/jobs/{job_id}/detections - Get detected trails
POST   /api/v1/jobs/{job_id}/correct  - Apply corrections
GET    /api/v1/jobs/{job_id}/download - Download result
```

## Project Structure

```
trailremover/
├── backend/
│   ├── app/
│   │   ├── api/         # REST API endpoints
│   │   ├── core/        # Detection & restoration
│   │   ├── models/      # Data models
│   │   └── services/    # Business logic
│   ├── ml/
│   │   ├── model.py     # U-Net architecture
│   │   ├── dataset.py   # Dataset loader
│   │   ├── train.py     # Training pipeline
│   │   └── inference.py # Production inference
│   └── data/
│       └── models/      # Trained model weights
└── frontend/
    ├── api_client.py    # Backend interface
    └── [GUI components] # PyQt5 interface
```

## Key Learnings

**Machine Learning:**
- Semantic segmentation for object detection
- U-Net architecture and skip connections
- Training pipeline with proper validation
- Handling class imbalance in pixel-wise tasks

**Software Engineering:**
- RESTful API design and implementation
- Asynchronous job processing
- State machine design for workflow management
- Production ML system integration

**Domain Knowledge:**
- FITS astronomical image format
- Histogram stretching for astronomy
- Inpainting algorithms and applications

## Results

**Successful Detection Examples:**
- 79% confidence on 3854-pixel trail
- Multiple trail detection in single image
- Orientation-invariant (horizontal, vertical, diagonal)

**Restoration Quality:**
- Seamless pixel restoration
- No visible artifacts
- Astronomical features preserved

## Technical Challenges Solved

1. **Memory Management**: Reduced batch size from 8→4 for GPU constraints
2. **Large Images**: Implemented tile-based processing with overlap
3. **False Positives**: Length filtering and duplicate merging
4. **Performance**: GPU acceleration and efficient data pipeline

## Deployment

**Backend:**
```bash
cd backend
python -m app.main
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU)
- FastAPI, Astropy, OpenCV
- Trained model weights

## API Integration

Complete API client provided for frontend integration:
- Simple Python interface
- Async operation support
- Comprehensive error handling
- Full documentation included

## Metrics

- **Lines of Code**: ~1,200 ML code, ~800 backend code
- **Training Time**: 100 epochs on RTX 2070
- **Model Size**: 250 MB (saved weights)
- **API Endpoints**: 5 RESTful endpoints
- **Test Success Rate**: 100% on validation workflow

## Project Highlights

- Production-ready ML system  
- Complete backend infrastructure  
- Fully tested and documented  
- Real astronomical data validation  
- Scalable architecture  


**Built with PyTorch, FastAPI, and Computer Vision techniques for automated astronomical image processing.**