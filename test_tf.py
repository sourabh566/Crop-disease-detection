import tensorflow as tf
print("TensorFlow imported successfully!")
print("Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
