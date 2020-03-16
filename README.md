# Tensorflow (Keras) model of semantic cognition

Based on the paper "Semantic Cognition: A Parallel Distributed Processing Approach" from 2003 by Rogers, T. and McClelland, J..
Inspired by the PyTorch implementation from https://github.com/jeffreyallenbrooks/rogers-and-mcclelland.

## Running the model

The necessary packages change over time, but be sure to

```bash
pip install tensorflow tensorboard
```

To train the model and generate log data, run

```python
python main.py
```

To view the logs in TensorBoard after execution, run

```bash
tensorboard --logdir logs/fit
```

and go to `localhost:6006` in your browser.
