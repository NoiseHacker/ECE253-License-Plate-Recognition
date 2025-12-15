# ECE253 License Plate Recognition

## Code Source and Attribution

This project uses the **OpenALPR pre-compiled 64-bit binaries for Windows** for license plate recognition and evaluation.  
OpenALPR source:https://github.com/openalpr/openalpr.
All image degradation and enhancement code in this repository is implemented by our team for academic use. In order to run OpenALPR, you must first install the Windows Visual C++ runtime libraries. The installer (vcredist_x64.exe) is included in the package.

## Project Description

This repository contains the code for the ECE 253 (Digital Image Processing) Final Project. We study how image degradation affects license plate recognition and how image enhancement methods can recover recognition performance.

## Directory Structure

```
contrast/
├── clean/ # Original clean images
├── degraded/ # Images after synthetic degradation
└── enhance/ # Enhanced images (two methods per degraded image)
alpr.exe    # OpenALPR pre-compiled binaries (Windows)
degradation.py # Generate degraded images from clean images
enhancement.py # Enhance degraded images
README.md
vcredist_x64.exe # The installer of Windows Visual C++ runtime libraries.
```

## Requirements

- Windows 64-bit
- Python ≥ 3.8
- Python packages: `pip install opencv-python numpy matplotlib`
- Microsoft Visual C++ x64 Runtime (required by OpenALPR)

## How to Run

### 1. Contrast

#### 1. Generate Degraded Images

- `python degradation.py`
- Input: `contrast/clean/`  
- Output: `contrast/degraded/`

#### 2. Enhance Degraded Images

- `python enhancement.py`
- Input: `contrast/degraded/`  
- Output: `contrast/enhance/`  
- Each degraded image produces two enhanced results using different methods.

### 2. Deblurring

```
python motion_deblur_compare.py blur/orig/filename
```

The output files are in blur/out

### 3. License Plate Recognition

Example using OpenALPR CLI:

```
PS C:\Users\openalpr_64> ./alpr -c us contrast/clean/us1.jpg
plate0: 6 results
    - 6XTS599    confidence: 93.7566
    - 6XT3599    confidence: 84.367
    - 6XTSS99    confidence: 84.2843
    - 6XT599     confidence: 84.205
    - 6XT3S99    confidence: 74.8946
    - 6XTS99     confidence: 74.7327
```

Detailed command line usage:

```
PS C:\Users\openalpr_64> ./alpr --help

USAGE: 

   alpr  [-c <country_code>] [--config <config_file>] [-n <topN>] [--seek
         <integer_ms>] [-p <pattern code>] [--clock] [-d] [-j] [--]
         [--version] [-h] <image_file_path>


Where: 

   -c <country_code>,  --country <country_code>
     Country code to identify (either us for USA or eu for Europe). 
     Default=us

   --config <config_file>
     Path to the openalpr.conf file

   -n <topN>,  --topn <topN>
     Max number of possible plate numbers to return.  Default=10

   --seek <integer_ms>
     Seek to the specified millisecond in a video file. Default=0

   -p <pattern code>,  --pattern <pattern code>
     Attempt to match the plate number against a plate pattern (e.g., md
     for Maryland, ca for California)

   --clock
     Measure/print the total time to process image and all plates. 
     Default=off

   -d,  --detect_region
     Attempt to detect the region of the plate image.  [Experimental] 
     Default=off

   -j,  --json
     Output recognition results in JSON format.  Default=off

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.

   <image_file_path>
     Image containing license plates


   OpenAlpr Command Line Utility
```

## Notes

- `degradation.py` degrades images.
- `enhancement.py` restores degraded images.
- OpenALPR is used for evaluation only.
