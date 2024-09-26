## Getting Started
This repository contains recognition modules for 8 different languages.
| Language |
|----------|
| Assamese | 
| Bengali  |
| Gujarati |
| Hindi    |
| Marathi  |
| Odia     |
| Punjabi  |
| Tamil    | 

### Versions
Version 1 at commit at 060efa6567a0ede9b0d08e14d922b488783548ac
### Installation
Requires Python >= 3.9 and PyTorch >= 2.0. The default requirements files will install the latest versions of the dependencies (as of February 22, 2024).
#### Updating dependency version pins
```bash
conda create -n STocr python=3.9
pip install pip-tools
make clean-reqs reqs  # Regenerate all the requirements files
```

```bash
# Use specific platform build. Other PyTorch 2.0 options: cu118, cu121, rocm5.7
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]
pip install fire==0.6.0
pip install numpy==1.26.4
```


### Inference 
Following command is used to get inference on a set of images from desired model options available in [assets](https://github.com/anikde/STocr/releases/tag/v1.0.0).
```
python your_script.py \
--checkpoint /path/to/checkpoint.ckpt \
--language hindi \
--image_dir /path/to/images \
--save_dir /path/to/save
```
For english you can directly run ```python infer.py --checkpoint  --image_dir images --language english --save_dir output ```
To check the argument usage ```python infer.py --help```.

### Acknowledgments

Text Recognition - [PARseq](https://github.com/baudm/parseq)