# Neural Style Transfer

A TensorFlow implementation of Neural Style Transfer that combines the content of one image with the artistic style of another using deep convolutional neural networks.

## Overview

This project implements the neural style transfer technique described in ["A Neural Algorithm of Artistic Style" by Gatys et al.](https://arxiv.org/abs/1508.06576). The algorithm uses a pre-trained VGG19 network to extract features from both content and style images, then optimizes a generated image to match the content representation of one image and the style representation of another.

## Features

- **Content Preservation**: Maintains the structural content of the input image
- **Style Transfer**: Applies artistic style from reference style images
- **Customizable Parameters**: Adjustable content/style loss weights (alpha/beta)
- **GPU Acceleration**: Optimized for Google Colab with A100 GPU support
- **Progress Visualization**: Saves intermediate results during training

## Requirements

```python
tensorflow>=2.x
numpy
matplotlib
PIL (Pillow)
scipy
```

## Project Structure

```
neural-style-transfer/
├── images/
│   ├── cat.jpg          # Content image
│   ├── stone_style.jpg  # Style image
│   ├── louvre.jpg       # Alternative content
│   └── monet_800600.jpg # Alternative style
├── output/              # Generated images saved here
├── pretrained-model/    # VGG19 weights (if using local)
└── neural_style_transfer.ipynb
```

## Usage

### 1. Setup Environment

The notebook is designed to run on Google Colab with GPU acceleration:

```python
# Runtime > Change runtime type > Hardware accelerator: GPU (A100)
```

### 2. Load Images

Place your content and style images in the `images/` directory. The default setup uses:
- **Content**: `images/cat.jpg`
- **Style**: `images/stone_style.jpg`

### 3. Configure Parameters

Adjust the hyperparameters in the notebook:

```python
img_size = 400           # Image dimensions
alpha = 10              # Content loss weight
beta = 40               # Style loss weight
epochs = 5000           # Training iterations
learning_rate = 0.001   # Optimizer learning rate
```

### 4. Run the Algorithm

Execute all cells in the notebook. The process:
1. Loads and preprocesses images
2. Initializes VGG19 model
3. Computes content and style costs
4. Optimizes the generated image
5. Saves results every 250 epochs

## Algorithm Details

### Content Cost

The content cost ensures the generated image preserves the content structure:

```python
J_content = ||a_C - a_G||² / (4 × n_H × n_W × n_C)
```

Where `a_C` and `a_G` are feature representations from layer `block5_conv4`.

### Style Cost

The style cost is computed using Gram matrices from multiple layers:

```python
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2), 
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]
```

### Total Cost

```python
J_total = α × J_content + β × J_style
```

## Key Functions

### `compute_content_cost(content_output, generated_output)`
Calculates the content preservation loss using deep layer features.

### `compute_layer_style_cost(a_S, a_G)`
Computes style cost for a single layer using Gram matrix differences.

### `compute_style_cost(style_image_output, generated_image_output)`
Combines style costs from multiple layers with specified weights.

### `train_step(generated_image)`
Performs one optimization step using gradient descent.

## Results

The algorithm generates intermediate results saved as:
- `output/image_0.jpg` (initial noisy image)
- `output/image_250.jpg`, `output/image_500.jpg`, etc.
- Final stylized image after all epochs

## Customization

### Change Images
Replace content/style images by updating the file paths:
```python
content_image = np.array(Image.open("images/your_content.jpg").resize((img_size, img_size)))
style_image = np.array(Image.open("images/your_style.jpg").resize((img_size, img_size)))
```

### Adjust Style Layers
Modify `STYLE_LAYERS` to emphasize different aspects:
- Lower layers: Fine textures and colors
- Higher layers: Abstract patterns and shapes

### Tune Hyperparameters
- **α (alpha)**: Higher values preserve more content details
- **β (beta)**: Higher values apply stronger style transfer
- **Learning rate**: Affects convergence speed and stability

## Performance

- **Training Time**: ~20-30 minutes for 5000 epochs on A100 GPU
- **Memory Usage**: ~8-12GB GPU memory for 400×400 images
- **Convergence**: Visible results after ~1000 epochs

## Troubleshooting

### Common Issues

1. **VGG19 Loading Error**: Use `weights='imagenet'` instead of local path
2. **GPU Memory**: Reduce `img_size` or batch operations
3. **Slow Training**: Ensure GPU acceleration is enabled in Colab

### Tips

- Start with lower resolution (200×200) for faster experimentation
- Try different content/style weight ratios for various effects
- Use high-contrast style images for more dramatic results

## References

- [Gatys et al. - A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [TensorFlow Neural Style Transfer Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [VGG19 Architecture](https://arxiv.org/abs/1409.1556)

## License

This project is open source and available under the MIT License.