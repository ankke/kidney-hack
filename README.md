# kidney-hack

https://www.kaggle.com/c/hubmap-kidney-segmentation/data

## setup

### prerequisites: 
1. python 3.7
2. pipenv
3. gdal

```pipenv install -d``` or ```pipenv install -d --python <python_version>```

### jupyter kernel setup:

```pipenv shell```

```python -m ipykernel install --user --name=<virtual_env_name>```

kernel should be available on the list: Kernel -> Change kernel

### dataset

```kaggle competitions download -c hubmap-kidney-segmentation```

and unzip in project root directory