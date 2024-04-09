import tensorflow as tf

# class NoisyTeacherEnforcer:
#     def __init__(self, noise_prob=0.4, mean=0.0, stddev=1.0):
#         self.noise_prob = noise_prob
#         self.mean = mean
#         self.stddev = stddev

#     def add_noise_to_tensor(self, tensor):
#         shape = tf.shape(tensor)
#         noise_mask = tf.random.uniform(shape) < self.noise_prob
#         noise = tf.random.normal(shape, mean=self.mean, stddev=self.stddev)
#         tensor_noisy = tf.where(noise_mask, tensor + noise, tensor)
#         return tensor_noisy

class NoisyTeacherEnforcer:
    def __init__(self, noise_prob=0.3, noise_scale=1.0):
        self.noise_prob = noise_prob
        self.noise_scale = noise_scale

    def add_noise_to_tensor(self, tensor):
        tensor = tf.cast(tensor, tf.float32)
        shape = tf.shape(tensor)
        noise_mask = tf.random.uniform(shape) < self.noise_prob
        # Generate uniform noise
        noise = tf.random.uniform(shape, minval=-self.noise_scale, maxval=self.noise_scale)
        tensor_noisy = tf.where(noise_mask, tensor + noise, tensor)
        return tensor_noisy