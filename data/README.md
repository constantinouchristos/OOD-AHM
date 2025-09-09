# Dataset Reconstruction Guide

This dataset has been split into multiple smaller files (25 MB each) for easier uploading and distribution.  
To use the dataset, you need to reassemble the parts back into the original `dataset.zip`.

---

## 1. Linux / macOS
Open a terminal in the folder containing the parts and run:

```bash
cat dataset.zip.part* > dataset.zip
