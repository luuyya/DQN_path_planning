import tensorflow as tf
import numpy as np
from io import BytesIO  # Python 3.x
import PIL.Image  # 用于图像处理

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert image to a format that can be used by TensorFlow
                img = PIL.Image.fromarray(np.uint8(img))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                image_string = buffer.getvalue()

                # Log the image using TensorFlow 2.x
                tf.summary.image(name=f"{tag}/{i}", data=np.expand_dims(np.array(PIL.Image.open(BytesIO(image_string))), axis=0), step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        counts, bin_edges = np.histogram(values, bins=bins)

        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step)
            self.writer.flush()
