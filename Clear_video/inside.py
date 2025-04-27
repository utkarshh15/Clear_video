import cv2
import numpy as np
import tensorflow as tf

def enhance_video(model_path, input_video_path, output_video_path, img_size=(224, 224), batch_size=16):
    # Load the trained model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame (resize and normalize)
        frame_resized = cv2.resize(frame, img_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frames_buffer.append(frame_normalized)
        
        # Process the batch once we have enough frames
        if len(frames_buffer) == batch_size:
            batch = np.array(frames_buffer)
            enhanced_frames = model.predict(batch)  # Run the model on the batch
            
            # Process each enhanced frame in the batch
            for enhanced_frame in enhanced_frames:
                enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)
                
                # Resize back to original dimensions and write to output
                enhanced_frame_resized = cv2.resize(enhanced_frame, (width, height))
                out.write(enhanced_frame_resized)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            frames_buffer = []  # Clear buffer after processing
    
    # If there are remaining frames in the buffer, process them as a smaller batch
    if len(frames_buffer) > 0:
        batch = np.array(frames_buffer)
        enhanced_frames = model.predict(batch)
        
        for enhanced_frame in enhanced_frames:
            enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)
            enhanced_frame_resized = cv2.resize(enhanced_frame, (width, height))
            out.write(enhanced_frame_resized)
            
            frame_count += 1
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release video objects
    cap.release()
    out.release()
    print("Video enhancement completed")

if __name__ == "__main__":
    model_path = r"C:\Users\prakh\OneDrive\Desktop\image clear\advanced_video_enhancement_model.h5"
    input_video_path = r"C:\Users\prakh\OneDrive\Desktop\clear new\coca.mp4"
    output_video_path = "coca.MP4"
    
    enhance_video(model_path, input_video_path, output_video_path)
