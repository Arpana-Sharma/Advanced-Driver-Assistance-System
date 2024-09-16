from ultralytics import YOLO
import cv2
import pandas as pd
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import supervision as sv
from scipy.spatial import distance

sam_checkpoint = 'C://Users/91638/Downloads/sam_vit_b_01ec64.pth'
model_type = 'vit_b'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

model = YOLO('C://Users/91638/Downloads/best.pt')

video_path = 'C://Users/91638/Downloads/video.avi'
video_out_path = 'C://Users/91638/Downloads/video_SAM_detected.avi'

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

CLASSES = ['Auto', 'Car', 'HV', 'LCV', 'MTW', 'Others']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=image, conf=0.5)
    mask_predictor.set_image(image)

    scene = image.copy()
    source_image = image.copy()
    segmented_image = image.copy()

    all_detections = []
    polygon_data = []

    for i, bbox in enumerate(results[0].boxes):
        box = np.array(bbox.xyxy)
        veh_class = int(bbox.cls)
        class_name = CLASSES[veh_class]
        x1, y1, x2, y2 = box[0].tolist()

        masks, scores, logits = mask_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True
        )

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array([veh_class] * len(masks))
        )

        detections = detections[detections.area == np.max(detections.area)]
        all_detections.append(detections)

        for mask in detections.mask:
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 15:  
                    continue
                contour_points = contour.reshape(-1, 2).tolist() 
                data = {'class': class_name}
                for idx, (x, y) in enumerate(contour_points):
                    data[f'x{idx}'] = x
                    data[f'y{idx}'] = y
                polygon_data.append(data)

    df = pd.DataFrame(polygon_data)

    combined_detections = sv.Detections(
        xyxy=np.concatenate([d.xyxy for d in all_detections]),
        mask=np.concatenate([d.mask for d in all_detections]),
        class_id=np.concatenate([d.class_id for d in all_detections])
    )

    source_image = box_annotator.annotate(scene=source_image, detections=combined_detections)
    segmented_image = mask_annotator.annotate(scene=segmented_image, detections=combined_detections)

    result_data = []

    for i, bbox in enumerate(results[0].boxes):
        box = np.array(bbox.xyxy)
        veh_class = int(bbox.cls)
        class_name = CLASSES[veh_class]

        bbox_points = [(box[0, 0], box[0, 1]), (box[0, 2], box[0, 1]), (box[0, 2], box[0, 3]), (box[0, 0], box[0, 3])]

        bbox_points_sorted = sorted(bbox_points, key=lambda pt: pt[1], reverse=True)

        highest_y_points = sorted(bbox_points_sorted[:2], key=lambda pt: pt[0])

        (x1, y1), (x2, y2) = highest_y_points

        min_dist_left = float('inf')
        nearest_point_left = None

        min_dist_right = float('inf')
        nearest_point_right = None

        for idx, row in df.iterrows():
            if row['class'] == class_name:
                for idx in range(len(row) // 2):
                    x_col = f'x{idx}'
                    y_col = f'y{idx}'
                    if x_col in row and y_col in row:
                        polygon_point = (row[x_col], row[y_col])

                        if not (np.isnan(polygon_point[0]) or np.isnan(polygon_point[1])):
                            dist_left = distance.euclidean((x1, y1), polygon_point)
                            if dist_left < min_dist_left:
                                min_dist_left = dist_left
                                nearest_point_left = polygon_point

                            dist_right = distance.euclidean((x2, y2), polygon_point)
                            if dist_right < min_dist_right:
                                min_dist_right = dist_right
                                nearest_point_right = polygon_point
        if nearest_point_left is not None and nearest_point_right is not None:
            result_data.append({
                'class': class_name,
                'x-left': int(nearest_point_left[0]),
                'y-left': int(nearest_point_left[1]),
                'x-right': int(nearest_point_right[0]),
                'y-right': int(nearest_point_right[1])
            })

    result_df = pd.DataFrame(result_data)

    def update_mtw_rows(row):
        if row['class'] == 'MTW':
            if row['y-left'] > row['y-right']:
                max_y = row['y-left']
                corresponding_x = row['x-left']
            else:
                max_y = row['y-right']
                corresponding_x = row['x-right']

            row['x-left'] = corresponding_x
            row['y-left'] = max_y
            row['x-right'] = corresponding_x
            row['y-right'] = max_y
        return row

    result_df = result_df.apply(update_mtw_rows, axis=1)

    for index, row in result_df.iterrows():
        cv2.circle(segmented_image, (int(row['x-left']), int(row['y-left'])), 4, (0, 255, 0), -1)
        cv2.circle(segmented_image, (int(row['x-right']), int(row['y-right'])), 4, (0, 255, 0), -1)

    for mask in combined_detections.mask:
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.reshape(-1, 2)
            cv2.polylines(segmented_image, [contour], isClosed=True, color=(0, 0, 255), thickness=2)

    output_frame = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    out.write(output_frame)

cap.release()
out.release()
