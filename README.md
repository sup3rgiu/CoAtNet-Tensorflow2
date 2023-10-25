# CoAtNet Tensorflow2
A [CoAtNet](https://arxiv.org/pdf/2106.04803v2.pdf) (**CoAtNet-0, CoAtNet-1, CoAtNet-2, CoAtNet-3, CoAtNet-4**) implementation using TensorFlow-2.0.


## Usage

```python
from coatnet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4

model = coatnet_0(image_size=(224,224), num_classes=1000, seed=42)
```

## Report

From the original paper:
| Model     | Eval Size       | #Params |
| --------- | --------------- | ------- |
| CoAtNet-0 | 224<sup>2</sup> | 25M     |
| CoAtNet-1 | 224<sup>2</sup> | 42M     |
| CoAtNet-2 | 224<sup>2</sup> | 75M     |
| CoAtNet-3 | 224<sup>2</sup> | 168M    |
| CoAtNet-4 | 384<sup>2</sup> | 275M    |

<br/>

From `model.summary()` ('Trainable params') of this implementation:
| Model     | Eval Size       | #Params |
| --------- | --------------- | ------- |
| CoAtNet-0 | 224<sup>2</sup> | 24.8M     |
| CoAtNet-1 | 224<sup>2</sup> | 42.6M     |
| CoAtNet-2 | 224<sup>2</sup> | 75.7M     |
| CoAtNet-3 | 224<sup>2</sup> | 169M    |
| CoAtNet-4 | 384<sup>2</sup> | 283M    |

## References
1. The original CoAtNet paper: https://arxiv.org/pdf/2106.04803v2.pdf
