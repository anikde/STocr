## Getting Started
This repository contains modules for recognition for 7 different languages.

### Installation
Requires Python >= 3.9 and PyTorch >= 1.10 (until 1.13). The default requirements files will install the latest versions of the dependencies (as of August 21, 2023).
```bash
# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]
```
#### Updating dependency version pins
```bash
pip install pip-tools
make clean-reqs reqs  # Regenerate all the requirements files
```

