import tensorflow as tf
import cv2
import numpy as np

def deblur_video(input_video_path, output_video_path):
    # Load the trained model
    model = tf.keras.models.load_model('enhanced_deblur_model_tf.h5', compile=False)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        # Perform deblurring
        deblurred_frame = model.predict(np.expand_dims(frame_normalized, axis=0))[0]

        # Post-process the deblurred frame
        deblurred_frame = (deblurred_frame * 255).clip(0, 255).astype(np.uint8)
        deblurred_frame = cv2.cvtColor(deblurred_frame, cv2.COLOR_RGB2BGR)
        deblurred_frame = cv2.resize(deblurred_frame, (width, height))

        # Write the frame
        out.write(deblurred_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Deblurred video saved as '{output_video_path}'")

# Usage
input_video_path = "path/to/blur_video.mp4"
output_video_path = "path/to/deblurred_video.mp4"
deblur_video(input_video_path, output_video_path)