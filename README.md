# CoAtNet Tensorflow2
A [CoAtNet](https://arxiv.org/pdf/2106.04803v2.pdf) (**CoAtNet-0, CoAtNet-1, CoAtNet-2, CoAtNet-3, CoAtNet-4**) implementation using TensorFlow-2.0.

![img](https://github.com/sup3rgiu/CoAtNet-Tensorflow2/assets/7725068/8e312b6b-2e57-4dd4-ab7f-9301292d3a1d)


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

## Citation
```bibtex
@article{dai2021coatnet,
  title={CoAtNet: Marrying Convolution and Attention for All Data Sizes},
  author={Dai, Zihang and Liu, Hanxiao and Le, Quoc V and Tan, Mingxing},
  journal={arXiv preprint arXiv:2106.04803},
  year={2021}
}
```
