import cv2
import iris
import matplotlib.pyplot as plt

import yaml

with open("active_contour.yaml", "r") as file:
    config_data = yaml.safe_load(file)

print(config_data)  # Check if the config is correctly loaded

# 1. Create IRISPipeline object
# iris_pipeline = iris.IRISPipeline(config="active_contour.yaml")
iris_pipeline = iris.IRISPipeline(config=config_data)

# 2. Load IR image of an eye
img_pixels = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)

# 3. Perform inference
# Options for the `eye_side` argument are: ["left", "right"]
output = iris_pipeline(img_data=img_pixels, eye_side="left")

# 4. Print iris pipeline
for i in iris_pipeline.params.pipeline:
    print(i)

# 5. Display the output
iris_visualizer = iris.visualisation.IRISVisualizer()
canvas = iris_visualizer.plot_segmentation_map(
    ir_image=iris.IRImage(img_data=img_pixels, eye_side="right"),
    segmap=iris_pipeline.call_trace['segmentation'],
)
plt.show()

