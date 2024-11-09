import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable
import numpy as np
from utils.general import non_max_suppression
from models.experimental import attempt_load


def load_model(weights_path, device):
    # Create an empty YOLOv5 model
    model = attempt_load(weights_path, device)
    model.eval()
    return model


def process_detections(detections, conf_thres):
    car_detections = []
    for det in detections:
        if det[5] == 2:  # 2 corresponds to the "car" class in the COCO dataset
            if det[4] >= conf_thres:
                car_detections.append(det[:4].cpu().numpy().astype(int))
    return car_detections


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    for det in detections:
        x1, y1, x2, y2 = det
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


model_path = 'model_pretrained/AOD_net_epoch_relu_10.pth'
net = torch.load(model_path, map_location=torch.device('cpu'))
cuda_available = torch.cuda.is_available()

# ===== Load input image =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "yolov5s.pt"
conf_thres = 0.5
model = load_model(weights_path, device)

# Open input video file
input_video_path = 'fogroad.mp4'  # Replace with your video path
output_video_path = 'output_video.mp4'  # Path to save the processed video as MP4
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Ensure FPS is correctly captured
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
print(f"FPS: {fps}, Total Frames: {total_frames}")

# List to store processed frames
processed_frames = []

frame_count = 0  # Initialize a frame counter

while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        original_frame = frame.copy()  # Copy the original frame for comparison display
        frame = cv2.resize(frame, (640, 640))  # Resize frame to match model input size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).unsqueeze_(0)
        varIn = Variable(frame)

        # Perform dehazing
        dehazed_frame = net(varIn)
        dehazed_frame = dehazed_frame.cpu().data.numpy().squeeze(0).transpose((1, 2, 0))
        dehazed_frame = (dehazed_frame * 255).astype(np.uint8)

        # Convert from RGB back to BGR
        dehazed_frame = cv2.cvtColor(dehazed_frame, cv2.COLOR_RGB2BGR)

        # Prepare the frame for YOLOv5 detection
        img = torch.from_numpy(dehazed_frame).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)

        # Perform object detection
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres)

        # Extract vehicle detection results
        detections = process_detections(pred[0], conf_thres)

        # Draw detection results on the frame
        frame = draw_detections(dehazed_frame, detections)

        # Resize the processed frame back to original size
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Store the processed frame in the list
        processed_frames.append(frame)

        # Print progress
        print(f"Processed frame {frame_count} / {total_frames}")

        # Show the result (optional)
        cv2.imshow('Original Frame', original_frame)
        cv2.imshow("Car Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# After processing, save the video with all frames at once
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Write all frames to the output video
for processed_frame in processed_frames:
    out.write(processed_frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()