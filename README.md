## Starcraft Super Resolution
Project to upscale Starcraft images using Super Resolution techniques.

---
#### Install Package and Requirements
```
pip install -r  starcraft_super_resolution/requirements.txt
pip install -e  starcraft_super_resolution/
```

---
#### Execute

##### Local  
- Decide on model settings and run  
`python3 /starcraft_super_resolution/run.py --name color_units --dataset units --layers 69,9 128,1 5 --scaling 5 --epochs 200 --batches 10000 --overwrite True`  
  The above command will create a super resolution model where the first interior layer will have `69` filters and a `9x9` kernel.  For more details on the options see the docstring in `upres/modeling/sr_model.py`.
- To view results launch `tensorboard --logdir starcraft_upres/upres/data/output`.

##### EC2  
- Set your AWS variables in a `.env` file in the base folder.  See `.env_sample`.
- Configure your `boto` access keys.  https://aws.amazon.com/developers/getting-started/python/.
- Edit the `Model Settings` variables in `launch.py`.
- `python3 starcraft_super_resolution/launch.py`.
- Output gives SSH command & link to hosted tensorboard instance.

---

#### Sources  
https://arxiv.org/pdf/1501.00092.pdf  
https://arxiv.org/pdf/1808.03344.pdf
