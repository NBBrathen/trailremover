# Trail Remover

> **Automated satellite trail detection and removal for astronomical images using deep learning**

A desktop application that uses a U-Net neural network to automatically detect and remove satellite trails from FITS astronomical images, restoring affected pixels using local background estimation with noise matching.

## Problem Statement

Satellite trails are a growing problem in astrophotography, appearing as bright streaks across long-exposure images due to the increasing number of satellites in orbit. Manual removal is time-consuming and requires expertise. Trail Remover automates this process using AI.

## Key Features

- **AI-Powered Detection**: U-Net neural network trained on astronomical images
- **Intelligent Restoration**: Local background estimation with noise matching for seamless removal
- **Desktop GUI**: User-friendly PyQt5 interface for batch processing
- **Batch Processing**: Handle multiple FITS images efficiently
- **Progress Tracking**: Real-time progress bar during processing
- **Save Results**: Download corrected images to any folder

## Screenshots

The application displays the original image alongside the processed result, showing detected trails and the restoration quality.

## Technical Stack

**Machine Learning:**
- PyTorch 2.0+ - Deep learning framework
- U-Net architecture - Semantic segmentation (31M parameters)
- Tile-based inference for large images
- GPU acceleration support (CUDA)

**Backend:**
- FastAPI - REST API with async job processing
- Astropy - FITS file handling
- OpenCV - Image processing
- NumPy/SciPy - Numerical operations

**Frontend:**
- PyQt5 - Cross-platform desktop GUI
- Threaded processing - Non-blocking UI during uploads
- Real-time progress updates

## Restoration Algorithm

The restoration uses **local background estimation with noise matching**:

1. **Mask Expansion**: Trail mask is dilated to capture faint edges
2. **Block Processing**: Image processed in 64x64 blocks for efficiency
3. **Local Statistics**: For each block, compute median and std of nearby non-trail pixels
4. **Noise Matching**: Fill trail pixels with `local_median + gaussian_noise(local_std)`
5. **Iterative Passes**: Multiple passes with increasing mask expansion catch halos

This approach preserves the natural noise texture of the sky background, unlike traditional inpainting which creates smooth "scars".

## Project Structure

```
trailremover/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/   # REST API endpoints
│   │   ├── core/
│   │   │   ├── detection.py    # Trail detection logic
│   │   │   └── restoration.py  # Pixel restoration
│   │   ├── models/             # Pydantic data models
│   │   ├── services/
│   │   │   ├── image_processor.py
│   │   │   └── job_manager.py
│   │   ├── config.py           # Settings
│   │   └── main.py             # FastAPI app
│   ├── ml/
│   │   ├── model.py            # U-Net architecture
│   │   ├── dataset.py          # Dataset loader
│   │   ├── train.py            # Training pipeline
│   │   └── inference.py        # Production inference
│   ├── data/models/            # Trained model weights
│   └── requirements.txt
├── frontend/
│   ├── main_window.py          # Main GUI application
│   ├── api_client.py           # Backend API client
│   ├── load_image.ui           # Qt Designer UI
│   └── loading_screen.ui
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Git LFS (for model weights)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/trailremover.git
cd trailremover

# Install Git LFS and pull model
git lfs install
git lfs pull

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install PyQt5 requests astropy opencv-python matplotlib numpy
```

### GPU Support (Optional)
For faster processing with NVIDIA GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Start the Backend
```bash
cd backend
python -m app.main
```
You should see:
```
INFO: ML detector initialized successfully
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 2. Start the Frontend
In a new terminal:
```bash
cd frontend
python main_window.py
```

### 3. Process Images
1. Click **"Load Images"**
2. Browse to a folder containing FITS files
3. Click **"Upload"** - processing begins automatically
4. View results by clicking on any image in the list
5. Click **"Save"** to download corrected images

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/images/upload` | Upload FITS image |
| GET | `/api/v1/jobs/{job_id}` | Check job status |
| GET | `/api/v1/jobs/{job_id}/detections` | Get detected trails |
| POST | `/api/v1/jobs/{job_id}/correct` | Apply corrections |
| GET | `/api/v1/jobs/{job_id}/download` | Download result |

## Configuration

Edit `backend/app/config.py` to customize:

```python
CONFIDENCE_THRESHOLD = 0.5   # Detection sensitivity (lower = more detections)
MIN_TRAIL_PIXELS = 50        # Minimum trail size
```

Edit `backend/app/services/image_processor.py` for restoration:

```python
restore_pixels_local_background_iterative(
    image_data,
    trails,
    sample_radius=30,   # Pixels to sample for background
    base_expand=8,      # Initial mask expansion
    passes=3            # Number of restoration passes
)
```

## Training (Optional)

To train on your own data:

1. Place labeled images in `backend/data/raw/`
   - PNG images with corresponding JSON (LabelMe format)
   - Label trails as "trail", "satellite", or "streak"

2. Run training:
```bash
cd backend
python -m ml.train
```

## Troubleshooting

### "Model not found" error
```bash
cd backend
git lfs pull
```

### PyTorch 2.6 compatibility
The code includes `weights_only=False` for torch.load() compatibility.

### Slow processing on CPU
- Process 2-3 images at a time
- Consider installing CUDA PyTorch for GPU acceleration

### Trail still visible after processing
- Increase `base_expand` parameter (try 12-15)
- Increase `passes` parameter (try 4-5)

## Performance

- **Detection Speed**: ~3-15 seconds per image (GPU), ~30-60 seconds (CPU)
- **Model Size**: 250 MB
- **Memory Usage**: ~2GB GPU VRAM for inference

## Team

- **Backend/ML**: Trail detection, restoration algorithms, API development
- **Frontend**: PyQt5 GUI, user experience

---

**Built with PyTorch, FastAPI, and PyQt5 for automated astronomical image processing.**