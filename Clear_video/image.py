import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split

def print_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def prepare_video_dataset(low_quality_dir, high_quality_dir, num_frames=30, img_size=(224, 224)):
    print("Starting prepare_video_dataset function")
    print_memory_usage()
    low_quality_data = []
    high_quality_data = []

    try:
        video_files = [f for f in os.listdir(low_quality_dir) if f.endswith(('.mp4', '.avi', '.MP4'))]
        print(f"Found {len(video_files)} video files")

        for video_name in video_files:
            print(f"Processing video: {video_name}")
            low_quality_path = os.path.join(low_quality_dir, video_name)
            high_quality_path = os.path.join(high_quality_dir, video_name)

            if not os.path.exists(high_quality_path):
                print(f"Warning: High quality video not found for {video_name}")
                continue

            low_cap = cv2.VideoCapture(low_quality_path)
            high_cap = cv2.VideoCapture(high_quality_path)

            if not low_cap.isOpened() or not high_cap.isOpened():
                print(f"Error: Could not open video files for {video_name}")
                continue

            frame_count = 0
            while frame_count < num_frames:
                ret_low, frame_low = low_cap.read()
                ret_high, frame_high = high_cap.read()

                if not ret_low or not ret_high:
                    print(f"Reached end of video {video_name} after {frame_count} frames")
                    break

                frame_low = cv2.resize(frame_low, img_size)
                frame_high = cv2.resize(frame_high, img_size)

                low_quality_data.append(frame_low)
                high_quality_data.append(frame_high)

                frame_count += 1

            low_cap.release()
            high_cap.release()

            print(f"Processed {frame_count} frames from {video_name}")

            if len(low_quality_data) >= num_frames:
                break

        print(f"Total processed frames: {len(low_quality_data)}")
        print_memory_usage()
        
        if len(low_quality_data) == 0:
            raise ValueError("No frames were processed. Check your video files and directories.")
        
        return np.array(low_quality_data), np.array(high_quality_data)
    except Exception as e:
        print(f"Error in prepare_video_dataset: {str(e)}")
        raise

def create_video_enhancement_model(input_shape):
    print(f"Creating model with input shape: {input_shape}")
    
    # Using EfficientNetB0 as the base model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    
    print(f"Shape after base model: {x.shape}")  # Should be (None, 7, 7, 1280)
    
    # Decoder (Upsampling the feature maps)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)  # (None, 14, 14, 256)
    print(f"Shape after first Conv2DTranspose: {x.shape}")
    
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)  # (None, 28, 28, 128)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)   # (None, 56, 56, 64)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)   # (None, 112, 112, 32)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x)   # (None, 224, 224, 16)
    
    # Final convolution to produce the RGB output with shape (224, 224, 3)
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    
    print(f"Shape of outputs: {outputs.shape}")
    
    # Residual connection (Adding the original input to the output)
    outputs = layers.Add()([inputs, outputs])
    
    # Create the final model
    model = models.Model(inputs, outputs)
    return model


# Load VGG16 model globally
vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

# Define perceptual loss using the preloaded VGG16 model
def perceptual_loss(y_true, y_pred):
    # Preprocess true and predicted images for VGG16
    y_true_vgg = tf.keras.applications.vgg16.preprocess_input(y_true)
    y_pred_vgg = tf.keras.applications.vgg16.preprocess_input(y_pred)
    
    # Extract features from both true and predicted images using VGG16
    true_features = vgg_model(y_true_vgg)
    pred_features = vgg_model(y_pred_vgg)
    
    # Calculate the perceptual loss as Mean Squared Error (MSE) between features
    loss = tf.reduce_mean(tf.square(true_features - pred_features))
    return loss

def combined_loss(y_true, y_pred):
    # Mean Squared Error (MSE) Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Perceptual Loss (using the preloaded VGG16 model)
    perceptual = perceptual_loss(y_true, y_pred)
    
    # Combine both losses with appropriate weights
    combined = mse_loss + 0.1 * perceptual
    
    return combined



def train_video_enhancement_model(model, low_quality_data, high_quality_data, epochs=50, batch_size=16):
    print("Starting train_video_enhancement_model function")
    print_memory_usage()
    
    X_train, X_val, y_train, y_val = train_test_split(low_quality_data, high_quality_data, test_size=0.2, random_state=42)

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # Compile model using the combined loss
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mae'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    print("Model training completed")
    print_memory_usage()
    return model, history


def enhance_video(model, input_video_path, output_video_path, img_size=(224, 224)):
    print(f"Enhancing video: {input_video_path}")
    print_memory_usage()
    
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, img_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)

        # Enhance frame
        enhanced_frame = model.predict(frame_batch)
        enhanced_frame = np.clip(enhanced_frame[0] * 255, 0, 255).astype(np.uint8)

        # Resize back to original dimensions
        enhanced_frame = cv2.resize(enhanced_frame, (width, height))

        out.write(enhanced_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
            print_memory_usage()

    cap.release()
    out.release()
    print("Video enhancement completed")
    print_memory_usage()

if __name__ == "__main__":
    print("Script started.")
    print_memory_usage()
    
    # Define your directories
    low_quality_dir = r"C:\Users\prakh\OneDrive\Desktop\blur"
    high_quality_dir = r"C:\Users\prakh\OneDrive\Desktop\clear"

    print(f"Low quality directory: {low_quality_dir}")
    print(f"High quality directory: {high_quality_dir}")

    try:
        print("Starting dataset preparation...")
        low_quality_data, high_quality_data = prepare_video_dataset(low_quality_dir, high_quality_dir, num_frames=100)  # Limit to 100 frames
        print("Dataset preparation completed.")
        print(f"Low quality data shape: {low_quality_data.shape}")
        print(f"High quality data shape: {high_quality_data.shape}")
        print_memory_usage()

        print("Normalizing data...")
        low_quality_data = low_quality_data.astype(np.float32) / 255.0
        high_quality_data = high_quality_data.astype(np.float32) / 255.0
        print("Data normalization completed.")
        print_memory_usage()

        print("Creating model...")
        model = create_video_enhancement_model(low_quality_data.shape[1:])
        print("Model created.")
        print_memory_usage()

        print("Starting model training...")
        trained_model, training_history = train_video_enhancement_model(model, low_quality_data, high_quality_data, epochs=10, batch_size=8)  # Reduced epochs and batch size
        print("Model training completed.")
        print_memory_usage()

        print("Saving model...")
        trained_model.save("video_enhancement_model.h5")
        print("Model saved.")
        print_memory_usage()

        print("Enhancing video...")
        input_video_path = "path/to/your/input/video.mp4"  # Update this path
        output_video_path = "enhanced_video.mp4"
        enhance_video(trained_model, input_video_path, output_video_path)

        print("Video enhancement completed. Enhanced video saved as:", output_video_path)
        print_memory_usage()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    print("Script finished.")
    print_memory_usage()