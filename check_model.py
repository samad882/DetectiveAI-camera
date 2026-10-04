import sys
sys.path.append('src')
from detection import Detector
model = Detector('models/bestonnx.onnx')
classes = model.model.get_modelmeta().custom_metadata_map.get("names", "")
print("Model classes metadata:", classes)
print("Unique classes:", len(set(model.classes)))
print("Total classes:", len(model.classes))
