## Starcraft Super Resolution

Project to upscale Starcraft images using Super Resolution techniques.

---

#### Requirements

- `Python 3.8`, install requirements via

```
pip install -r starcraft_super_resolution/requirements.txt
pip install -e starcraft_super_resolution/
```

- AWS CLI tool (see https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

---

##### Model Settings

The design of the model used is specified via the command line
`python3 /starcraft_super_resolution/run.py --name {name} --dataset {dataset} --layers {layers} --scaling {scaling} --epochs {epochs} --batch_size {batch_size} --epochs_per_save {epochs_per_save} --overwrite {overwrite}`
where

```
name: The name of your model, used when saving data and displaying via tensorboard.
dataset: Which dataset to use in training, "units" or "frames".
layers: String that specifies the model architecture.  Example: "layer_1_kernel_size,layer_1_num_filters layer_2_kernel_size,layer_2_num_filters final_layer_kernel_size".  The number of filters is not specifies for the final layer since this is specified by whether the final image should be greyscale or not.
scaling: To what scale do we want to upres an image.
epochs: Number of epochs to train on.
batch_size: Batch Size used in training.
epochs_per_save:  How many epochs to train on between saving images/models.
greyscale:  True or False, specifies if color images will be used or not.
overwrite:  Whether to overwrite previous model data.
```

#### Execute

##### Local

- Decide on model settings and run

  `python3 /starcraft_super_resolution/run.py --name {name} --dataset {dataset} --layers {layers} --scaling {scaling} --epochs {epochs} --batch_size {batch_size} --epochs_per_save {epochs_per_save} --overwrite {overwrite}`


  for a specific simple example

  
  `python3 ./starcraft_super_resolution/run.py --name test_model --dataset units --layers 12,3 5 --scaling 3 --epochs 10 --batch_size 32 --epochs_per_save 2 --overwrite False`
- To view results launch `tensorboard --logdir starcraft_super_resolution/upres/data/output --max_reload_threads 1`.

##### EC2

- Set your AWS variables in a `.env` file in the base folder. See `.env_sample`.
- Configure your `boto` access keys. https://aws.amazon.com/developers/getting-started/python/.
- Edit the `Model Settings` variables in `launch.py`.
- `python3 starcraft_super_resolution/launch.py`.
- Output gives SSH command & link to hosted tensorboard instance.

---

#### Sources

https://arxiv.org/pdf/1501.00092.pdf  
https://arxiv.org/pdf/1808.03344.pdf
