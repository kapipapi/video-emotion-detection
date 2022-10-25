# How to run?
1. Create **python 3.8** virtual environment (`virtualenv`), with `virtualenv venv`
2. Install `requirements.txt` with `pip install -r requirements.txt`
3. Download RAVDESS dataset with `here will be bash code for downloading dataset`
4. Prepare video files for training/testing
5. Download pre-trained weights from ...

# Proper project tree
```commandline
video-emotion-detection
├── train.py
├── test.py
├── venv
│   ├── ...
├── utils
│   ├── ...
├── dataset
│   ├── Actor_01
│   │   ├── 01-01-01-01-01-01-01.mp4
│   │   ├── ...
│   ├── Actor_02
│   │   ├── 01-01-01-01-01-01-02.mp4
│   │   ├── ...
...
```