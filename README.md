
## Introduction



## Installation

Make sure to install CUDA 11 or higher properly.

```shell
pip3 install -U pip
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

After this, test the installation by running the following command:

```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, then the installation is successful.

## Runing on computecanada


The explnations of arguments can be found in the `lop.py` file.

