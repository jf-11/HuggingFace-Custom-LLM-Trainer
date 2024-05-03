# HuggingFace-LLM-Custom-Training

<p>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>
</p>

<img width="274" alt="logo" src="logo.png">

This repository provides a setup for training large language models (LLMs) using Hugging Face's Transformers library. It is designed to handle distributed training across multiple GPUs and incorporates techniques such as LoRA and quantization for efficient training.

## Configuration

The training process is driven by a JSON configuration file (`config.json`) which specifies all necessary parameters and settings. Here's an overview of the arameters:

- `run_name`: Identifier for the training run.
- `wandb_log`: Boolean to enable logging with Weights & Biases.
- `wandb_project`: Name of the Weights & Biases project for logging.
- `seed`: Seed for random number generation to ensure reproducibility.
- `path_to_data`: Directory path where training and validation data are stored.
- `out_dir`: Output directory for saving trained model checkpoints.
- `model_name`: Identifier for the pre-trained model from Hugging Face.
- `device`: Specifies the computing device, such as 'cuda' for GPU.
- `gradient_acc_steps`: Number of steps to accumulate gradients before updating model weights.
- `grad_clip`: Maximum norm of the gradients for gradient clipping.
- `dtype`: Data type of model parameters (e.g., 'bfloat16').
- `quantization`: Enables model quantization to reduce memory usage, with options like `false`, `4bit`, or `8bit`.
- `lora_rank`: Rank for the LoRA (Low-Rank Adaptation) layers.
- `lora_alpha`: Alpha value for scaling the LoRA layers.
- `target_modules`: List of model modules to apply LoRA modifications.
- `compile`: Boolean to enable model compilation for performance optimization.
- `max_len`: Maximum sequence length for input data.
- `num_workers`: Number of worker threads for loading data.
- `batch_size`: Number of samples per batch.
- `lr`: Base learning rate for the optimizer.
- `betas`: Coefficients used for computing running averages of gradient and its square.
- `eps`: Term added to the denominator to improve numerical stability in the optimizer.
- `weight_decay`: Weight decay (L2 penalty) parameter.
- `lr_schedule`: Boolean to enable learning rate scheduling.
- `warmup`: Number of steps to linearly increase the learning rate from zero.
- `epochs`: Total number of training epochs to run.

## Files

- `config.json`: Contains all configuration settings for the training.
- `train.py`: Main script for training the model. It reads the configuration, sets up the model, data, and training process.
- `utils.py`: Contains helper functions and classes such as the dataset class, parameter analysis, and learning rate scheduler.

## Usage

To start training, you need to run the `train.py` script with the path to the configuration file. Here are the commands for different setups:

- **Single GPU:**
  `python train.py config.json`

- **Distributed Training (DDP) on 2 GPUs:**
  `torchrun --standalone --nproc_per_node=2 --nnodes=1 train.py config.json`

## Data Loading and Format in the Dataset Class

### Data Loading
Data for training is loaded using Python's `pickle` module, which allows for the serialization and deserialization of Python object structures. In the training script (`train.py`), data is expected to be stored in `.pkl` files, specifically for training and validation datasets.

### Expected Data Format
The data should be in a dictionary format where each key is an integer index, and the value is another dictionary with two keys: `text` and `label`. Here is the expected structure:
```
  {
      0: {'text': "Example input text", 'label': "Corresponding output text"},
      1: {'text': "Another input text", 'label': "Corresponding output text"},
      ...
  }
```

## Notes

- Make sure the paths in `config.json` for data and output directories are correctly set up before starting the training.
- Adjust the `num_workers` in the DataLoader for optimized data loading.
- Monitor the training progress with Weights & Biases if `wandb_log` is set to `true`.
- Adapt the Dataset class for your needs.

## Acknowledgments

This repository was inspired by and, in some parts, based on the following resources:

- [Andrej Karpathy's Llama2 Training Script](https://github.com/karpathy/llama2.c/blob/master/train.py): The structure and some functionalities of the training script are based on concepts and code snippets from Andrej Karpathy's implementation.
