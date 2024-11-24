# Scriptr

Scriptr is a fine-tuned Gemma 2 2B instruct model designed for generating YouTube scripts based on provided keywords. This project leverages state-of-the-art natural language processing techniques to create high-quality and relevant scripts efficiently.

## Demo

![Scriptr Script for ML](https://github.com/user-attachments/assets/89809657-ccce-4bdb-8d00-bd4c1f881359)


## Features
- Generates YouTube scripts from simple keyword inputs.
- Supports inference on both CPU and GPU environments.

## Getting Started

### Prerequisites
Ensure you have Python installed along with the following dependencies:
- `torch`
- `bitsandbytes`
- `transformers`
- `peft`
- `accelerate`
- `trl`

### Installation and Usage
1. Navigate to the `Scriptr` directory.
2. Install the required dependencies:
```
pip install torch bitsandbytes transformers peft accelerate trl
```
3. Run the `inference.py` file:
```
python inference.py
```
- **CPU Environment**: Expect slower performance.
- **GPU Environment**: Significantly faster inference.

## Notes
- The first run may take some time as the model initializes.
- Ensure a compatible GPU is available for optimal performance.
