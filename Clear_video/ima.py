import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, VGG16
import cv2

def prepare_video_dataset(low_quality_dir, high_quality_dir, num_frames=300, img_size=(224, 224)):
    low_quality_data = []
    high_quality_data = []

    video_files = [f for f in os.listdir(low_quality_dir) if f.endswith(('.mp4', '.avi', '.MP4'))]
    print(f"Found {len(video_files)} video files")

    for video_name in video_files:
        low_quality_path = os.path.join(low_quality_dir, video_name)
        high_quality_path = os.path.join(high_quality_dir, video_name)

        if not os.path.exists(high_quality_path):
            print(f"Warning: High quality video not found for {video_name}")
            continue

        low_cap = cv2.VideoCapture(low_quality_path)
        high_cap = cv2.VideoCapture(high_quality_path)

        frame_count = 0
        while frame_count < num_frames:
            ret_low, frame_low = low_cap.read()
            ret_high, frame_high = high_cap.read()

            if not ret_low or not ret_high:
                break

            frame_low = cv2.resize(frame_low, img_size)
            frame_high = cv2.resize(frame_high, img_size)

            low_quality_data.append(frame_low)
            high_quality_data.append(frame_high)

            frame_count += 1

        low_cap.release()
        high_cap.release()

    return np.array(low_quality_data), np.array(high_quality_data)

def create_advanced_enhancement_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (using EfficientNetB0)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Decoder
    x = base_model.output
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x)
    
    # Attention mechanism
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    # Final convolution
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    
    # Residual connection
    outputs = layers.Add()([inputs, outputs])
    
    return models.Model(inputs, outputs)

class VideoEnhancementModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(VideoEnhancementModel, self).__init__()
        self.enhancement_model = create_advanced_enhancement_model(input_shape)
        self.vgg = VGG16(include_top=False, weights='imagenet')
        self.vgg.trainable = False

    def compile(self, optimizer, loss_weights):
        super(VideoEnhancementModel, self).compile()
        self.optimizer = optimizer
        self.loss_weights = loss_weights

    def call(self, inputs):
        return self.enhancement_model(inputs)

    @tf.function
    def train_step(self, data):
        low_quality, high_quality = data

        with tf.GradientTape() as tape:
            enhanced = self.enhancement_model(low_quality, training=True)
            loss = self._compute_loss(high_quality, enhanced)

        gradients = tape.gradient(loss, self.enhancement_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.enhancement_model.trainable_variables))

        return {"loss": loss}

    def _compute_loss(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        perceptual_loss = self._perceptual_loss(y_true, y_pred)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        total_loss = (
            self.loss_weights['mse'] * mse_loss +
            self.loss_weights['perceptual'] * perceptual_loss +
            self.loss_weights['ssim'] * ssim_loss
        )

        return total_loss

    def _perceptual_loss(self, y_true, y_pred):
        y_true = tf.keras.applications.vgg16.preprocess_input(y_true * 255.0)
        y_pred = tf.keras.applications.vgg16.preprocess_input(y_pred * 255.0)
        
        true_features = self.vgg(y_true)
        pred_features = self.vgg(y_pred)
        
        return tf.reduce_mean(tf.square(true_features - pred_features))

def main():
    low_quality_dir = r"C:\Users\prakh\OneDrive\Desktop\blur new"
    high_quality_dir = r"C:\Users\prakh\OneDrive\Desktop\clear new"
    
    low_quality_data, high_quality_data = prepare_video_dataset(low_quality_dir, high_quality_dir, num_frames=300)
    
    low_quality_data = low_quality_data.astype(np.float32) / 255.0
    high_quality_data = high_quality_data.astype(np.float32) / 255.0
    
    model = VideoEnhancementModel(low_quality_data.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss_weights={'mse': 0.5, 'perceptual': 0.3, 'ssim': 0.2}
    )

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((low_quality_data, high_quality_data))
    dataset = dataset.shuffle(1000).batch(16)

    # Train the model
    model.fit(dataset, epochs=50, verbose=1)
    
    model.enhancement_model.save("advanced_video_enhancement_model.h5")

if __name__ == "__main__":
    main()