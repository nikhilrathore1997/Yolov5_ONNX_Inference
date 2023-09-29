#Writen By Nikhil
import onnxruntime as ort
import os
import cv2
import torch
import numpy as np
import argparse

from yolov5.utils.general import non_max_suppression, xyxy2xywh

def load_model(model_path):
    providers = ['CPUExecutionProvider']  # Add the enabled providers
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session

def main(model_path, labels_file, input_dir, output_file):
    labels = open(labels_file, "r").read().strip().split('\n')
    INPUT_H = 320
    INPUT_W = 320
    num_channels = 3  # RGB

    session = load_model(model_path)

    for input_img in os.listdir(input_dir):
        image_path = os.path.join(input_dir, input_img)
        image = cv2.imread(image_path)
        img_original = cv2.imread(image_path)

        img_width, img_height = img_original.shape[0:2]

        width_ratio = img_width / INPUT_H
        height_ratio = img_height / INPUT_W

        image = cv2.resize(image, (INPUT_W, INPUT_H))  # Resize to the model's input size
        image = image / 255.0  # Normalize pixel values to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.array(image, dtype=np.float32)
        
        outputs = session.run(None, {session.get_inputs()[0].name: image})  # Return output in list format
        output_ = torch.from_numpy(np.asarray(outputs))  # Convert into tensor format

        out = non_max_suppression(output_, conf_thres=0.2, iou_thres=0.6)[0]

        for i, (X, Y, X2, Y2, score, class_id) in enumerate(out):
            x = int(X) * width_ratio
            y = int(Y) * height_ratio
            x2 = int(X2) * width_ratio
            y2 = int(Y2) * height_ratio
            w = int(x2 - x)
            h = int(y2 - y)
            shape_ = (int(x), int(y), int(w), int(h))
            class_id = round(float(class_id))
            score = float(score)

            img_original = cv2.rectangle(img_original, shape_, (255, 0, 0), 1)
            img_original = cv2.rectangle(img_original, (int(x), int(y), int(w), 15), (255, 0, 0), -1)
            img_original = cv2.putText(img_original, str(score)[0:4], (int(x), int(y + 12)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(output_file+"/"+input_img, img_original)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output Directory")

    args = parser.parse_args()
    main(args.model, args.labels, args.input_dir, args.output_file)
