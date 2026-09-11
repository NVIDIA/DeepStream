## Detection

### Helper to get a subset from the complete set
```python
python3 create_coco_subset.py /media/scratch.metropolis3/yuw/datasets/coco/annotations/instances_val2017.json /tmp/val2017_vehicle.50.json --supercategory vehicle --num_images 50
```

### Eval usage:
```python
python3 detection_eval.py --val-json=/tmp/val2017_vehicle.50.json --image-dir=/media/scratch.metropolis3/yuw/datasets/coco/val2017 --output=/tmp/tao/pred_val2017_vehicle.50.json --host 10.111.53.46 --dump-vis-path=/tmp/tao/vis_val2017_vehicle.50
```

### COCO Prediction schema
```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "examples": [
      [
        {
            "image_id": 1,
            "bbox": [366.95068359, 386.24633789, 17.97363281, 17.46826172],
            "score": 0.9428234,
            "category_id": 2
        },
        {
            "image_id": 1,
            "bbox": [548.27636719, 32.32040405, 5.94726562, 5.95733643],
            "score": 0.93891287,
            "category_id": 2
        }
      ]
    ],
    "items": {
      "type": "object",
      "properties": {
        "image_id": {
          "type": "integer",
          "description": "Image identifier from source_id"
        },
        "bbox": {
          "type": "array",
          "items": {
            "type": "number"
          },
          "minItems": 4,
          "maxItems": 4,
          "description": "Bounding box coordinates [x, y, w, h] from detection_boxes"
        },
        "score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Detection confidence score from detection_scores"
        },
        "category_id": {
          "type": "integer",
          "description": "Category identifier from detection_classes"
        }
      },
      "required": ["image_id", "bbox", "score", "category_id"]
    }
  }
```

## Semantic Segmentation

### Usage
```python
python3 semantic_segmentation_eval.py --config eg.experiment.seg.yaml --host 127.0.0.1 --port 8800
```

### --config is tao-deploy experiment.yaml config, we only need to dataset section
```yaml
# Example config YAML structure:
dataset:
  segment:
    root_dir: /path/to/dataset
    validation_split: val
    palette:
      - label_id: 0
        mapping_class: background
        rgb: [0, 0, 0]
        seg_class: background
      - label_id: 1
        mapping_class: foreground
        rgb: [255, 255, 255]
        seg_class: foreground
```

## Classification

### Prerequisites
```bash
pip install scikit-learn
```

### Full evaluation:
```python
# eg.experiment.cls.yaml is from tao-deploy
python3 classification_eval.py --config eg.experiment.cls.yaml
```

### Run evaluation against tao-deploy inference result
first run tao-deploy inference on the classification to get the result.csv file, which has below format:
```csv
/path/to/image1.jpg,n01440764,0.9
/path/to/image2.jpg,n01443537,0.8
```

```python
# results.csv is dumped by tao-deploy classification inference
python3 classification_eval.py --config eg.experiment.cls.yaml --csv-path results.csv
```

### Run subset, limited per-class evaluation(e.g., 2 images per class):
```python
# eg.experiment.cls.yaml is from tao-deploy
python3 classification_eval.py --config eg.experiment.cls.yaml --max-images-per-class 2
```

### Run subset, random subset evaluation(e.g., 1000 random images):
```python
python3 classification_eval.py --config eg.experiment.cls.yaml --total-images 1000
```

