import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import layers, models, optimizers, applications
import glob
import os
import matplotlib.pyplot as plt

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the model architecture
def create_deblur_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    skip2 = x
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model(inputs, outputs)

# Create VGG16 model for perceptual loss
vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False
perceptual_model = models.Model(vgg.input, vgg.get_layer('block3_conv3').output)

# Custom loss function
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    y_true_features = perceptual_model(y_true)
    y_pred_features = perceptual_model(y_pred)
    
    perceptual_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
    
    return mse_loss + 0.1 * perceptual_loss

# Function to load video paths
def load_video_paths(blur_folder, clear_folder):
    blur_video_paths = sorted(glob.glob(os.path.join(blur_folder, "*")))
    clear_video_paths = sorted(glob.glob(os.path.join(clear_folder, "*")))
    
    if len(blur_video_paths) != len(clear_video_paths):
        raise ValueError("Number of blur and clear videos do not match.")
    
    print(f"Found {len(blur_video_paths)} pairs of videos.")
    
    for blur, clear in zip(blur_video_paths, clear_video_paths):
        print(f"Blur: {os.path.basename(blur)} | Clear: {os.path.basename(clear)}")
    
    return list(zip(blur_video_paths, clear_video_paths))

# Frame generator
def frame_generator(video_pairs, batch_size=4):
    while True:
        for blur_path, clear_path in video_pairs:
            blur_cap = cv2.VideoCapture(blur_path)
            clear_cap = cv2.VideoCapture(clear_path)
            
            while True:
                blur_frames = []
                clear_frames = []
                
                for _ in range(batch_size):
                    blur_ret, blur_frame = blur_cap.read()
                    clear_ret, clear_frame = clear_cap.read()
                    
                    if not blur_ret or not clear_ret:
                        break
                    
                    blur_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2RGB)
                    clear_frame = cv2.cvtColor(clear_frame, cv2.COLOR_BGR2RGB)
                    
                    blur_frame = cv2.resize(blur_frame, (256, 256))
                    clear_frame = cv2.resize(clear_frame, (256, 256))
                    
                    blur_frames.append(blur_frame)
                    clear_frames.append(clear_frame)
                
                if len(blur_frames) == 0:
                    break
                
                yield np.array(blur_frames, dtype=np.float32) / 255.0, np.array(clear_frames, dtype=np.float32) / 255.0
            
            blur_cap.release()
            clear_cap.release()

# Training function
def train_model(blur_folder, clear_folder, num_epochs=100, batch_size=4):
    # Load video paths
    video_pairs = load_video_paths(blur_folder, clear_folder)
    
    # Split into train and validation
    split = int(0.8 * len(video_pairs))
    train_pairs = video_pairs[:split]
    val_pairs = video_pairs[split:]
    
    # Create generators
    train_gen = frame_generator(train_pairs, batch_size)
    val_gen = frame_generator(val_pairs, batch_size)
    
    # Create and compile the model
    model = create_deblur_model()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=custom_loss)
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    
    # Estimate steps per epoch
    steps_per_epoch = sum([int(cv2.VideoCapture(pair[0]).get(cv2.CAP_PROP_FRAME_COUNT)) for pair in train_pairs]) // batch_size
    validation_steps = sum([int(cv2.VideoCapture(pair[0]).get(cv2.CAP_PROP_FRAME_COUNT)) for pair in val_pairs]) // batch_size
    
    # Train the model
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=[early_stopping, reduce_lr])
    
    # Save the model
    model.save('enhanced_deblur_model_tf.h5')
    print("Training completed. Model saved as 'enhanced_deblur_model_tf.h5'")
    
    return history

# Main execution
if __name__ == "__main__":
    blur_folder = r"C:\Users\prakh\OneDrive\Desktop\blur new"
    clear_folder = r"C:\Users\prakh\OneDrive\Desktop\clear new"
    
    history = train_model(blur_folder, clear_folder)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()