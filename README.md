# Conditional-Generative-Adversarial-Network
Train a GAN to transform images from one domain to another domain

# Sample usage
```python
import src

dataset = Dataset(x_path='<SRC_IMAGES / SRC_NPY>', y_path='<TARGET_IMAGES / TARGET_NPY>', img_size=<INTEGER REPRESENTING DIM OF SQUARE IMG>, resize_required=<True / FALSE>, load=<True / False>)
gan = GAN(dataset)
gan.generate('<PATH OF FOLDER CONTAINING TEST IMGS>', '<OUTPUT_DIR>')
```
