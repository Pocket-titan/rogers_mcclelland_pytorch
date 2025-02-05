# PyTorch models of semantic cognition

Based on the paper "Semantic Cognition: A Parallel Distributed Processing Approach" from 2003 by Rogers and McClelland, whose model is itself an adaptation of an earlier one by Rumelhart (1990).

Inspired by the PyTorch implementation from https://github.com/jeffreyallenbrooks/rogers-and-mcclelland.

## Running the model

The necessary packages change over time, but be sure to

```bash
pip/conda install torch torchvision tensorboard pandas scipy numpy seaborn scikit-learn matplotlib
```

To train the model and generate log data, run

```python
python feedforward.py
```

To view the logs in TensorBoard (it's nice to keep this running in a seperate process), run

```bash
tensorboard --logdir logs/fit
```

and go to `localhost:6006` in your browser. One run will show up in TensorBoard as two sets of data; one containing the scalar metrics (in `${run_name}/train`) and another containing the images (in `${run_name}/`).

To clear the logs (the folder tends to fill up fast), run

```bash
python clear_logs.py
```
