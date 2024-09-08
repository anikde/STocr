## Getting Started
This repository contains modules for recognition for 7 different languages.

### Installation
Requires Python >= 3.9 and PyTorch >= 1.10 (until 1.13). The default requirements files will install the latest versions of the dependencies (as of August 21, 2023).

#### Updating dependency version pins
```bash
conda create -n STocr python=3.9
pip install pip-tools
make clean-reqs reqs  # Regenerate all the requirements files
```

```bash
# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]
pip install fire==0.6.0
pip install numpy==1.26.4
```
Tested on CUDA 11.7 with python 3.9.

### Inference 
Following command is used to get inference on a set of images from desired model options available in [assets](https://github.com/anikde/STocr/releases/tag/v1.0.0).
```
python your_script.py \
--checkpoint /path/to/checkpoint.ckpt \
--language hindi \
--image_dir /path/to/images \
--save_dir /path/to/save
```
To check the argument usage ```python infer.py --help```.

### Acknowledgments

Text Recognition - [PARseq](https://github.com/baudm/parseq)
