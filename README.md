
# lane_detection
lane detection is a computer vision project that identifies lane markings on roads using image processing and deep learning techniques. the project typically involves edge detection, deep learning-based models like cnn and rnn to accurately detect lanes
## Usage:

### 1. Set up the environment 
`conda env create -f environment.yml`

To activate the environment:

Window: `conda activate carlane`

Linux, MacOS: `source activate carlane`

### 2. Run the pipeline:
```bash
python3 main.py INPUT_IMAGE OUTPUT_IMAGE_PATH
python3 main.py --video INPUT_VIDEO OUTPUT_VIDEO_PATH
```

