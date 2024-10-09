import tensorflow as tf
import numpy as np

class Logger(object):

    def __init__(self, log_dir):
        """创建一个写入日志到 log_dir 的 summary writer。"""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """记录一个标量变量。"""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """记录一组图像。"""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # 使用 TensorFlow 2.x 记录图像
                tf.summary.image(name=f"{tag}/{i}", data=np.expand_dims(img, axis=0), step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """记录张量值的直方图。"""
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step)
            self.writer.flush()
