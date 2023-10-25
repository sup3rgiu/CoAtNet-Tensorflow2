import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # Suppress TF warnings (must be set before importing Tensorflow)

import tensorflow as tf
import keras
from coatnet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# To count only trainable_weights
# Useful to show only the count of trainable parameters in the 'Param #' column when using model.summary()
def count_params(self):
    """Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    """
    if not self.built:
        if getattr(self, "_is_graph_network", False):
            with keras.src.utils.tf_utils.maybe_init_scope(self):
                self._maybe_build(self.inputs)
        else:
            raise ValueError(
                "You tried to call `count_params` "
                f"on layer {self.name}"
                ", but the layer isn't built. "
                "You can build it manually via: "
                f"`{self.name}.build(batch_input_shape)`."
            )
    return keras.src.utils.layer_utils.count_params(self.trainable_weights)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.layers.Layer.count_params = count_params

    model = coatnet_0(image_size=(224,224), num_classes=1000, seed=42)

    print(model.summary(expand_nested=False))