import yolov_model as ym
import cv2
import numpy as np

def preprocess(image_path,file_name):
    model = ym.load()
    # Load the image
    image = cv2.imread(image_path)
    results = model(image, verbose=False)
    objects = results[0].boxes.data.cpu().numpy()
    colors = np.random.uniform(0, 255, size=(len(objects), 3))

    for i in range(len(objects)):
        x1, y1, x2, y2, _, _ = objects[i]
        color = colors[i]

        # Create a copy of the image
        overlay = image.copy()

        # Draw a filled rectangle on the copy of the image
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)

        # Blend the original image and the copy of the image
        alpha = 0.3  # Define the transparency factor
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Calculate the center of the rectangle
        center_x = int(0.9*(x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Write the number in the center of the rectangle on the final image
        cv2.putText(image, str(i+1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 20)

    cv2.imwrite('static/temp/'+file_name, image)
    return objects
    