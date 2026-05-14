# Quick Normalization Reference

## The Two Critical Parameters

When generating DeepStream nvinfer configs, **BOTH** of these parameters must exactly match your model's training preprocessing:

### 1. `net-scale-factor`
**Purpose**: Scales pixel values (typically from [0-255] range)

**Common values**:
- `0.00392156862745098` (1/255) - Most common, scales to [0,1]
- `0.5` - Scales by half
- `1.0` - No scaling

### 2. `offsets` (per-channel mean subtraction)
**Purpose**: Subtracts mean values from each color channel after scaling

**Format**: `R;G;B` (Red;Green;Blue)

**Common values**:
- **Not specified** or `0;0;0` - No mean subtraction
- `127.5;127.5;127.5` - For [-1,1] normalization
- `123.675;116.28;103.53` - ImageNet mean values

---

## Quick Decision Tree

### Step 1: Check your training code

```python
# Example 1: Simple [0,1] scaling
image = image / 255.0
```
**Config**: `net_scale_factor: 0.00392156862745098`, **no offsets**

---

```python
# Example 2: [-1,1] normalization
image = (image / 255.0 - 0.5) * 2
# or: image = (image - 127.5) / 127.5
```
**Config**: `net_scale_factor: 0.00392156862745098`, `offsets: "127.5;127.5;127.5"`

---

```python
# Example 3: ImageNet normalization
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
**Config**: `net_scale_factor: 0.00392156862745098`, `offsets: "123.675;116.28;103.53"`
*(mean values × 255)*

---

```python
# Example 4: Custom scaling
image = image * 0.5
```
**Config**: `net_scale_factor: 0.5`, **no offsets**

---

```python
# Example 5: Custom scaling with mean subtraction
image = (image * 0.01) - custom_mean
```
**Config**: `net_scale_factor: 0.01`, `offsets: "{R};{G};{B}"` *(your custom mean values)*

---

```python
# Example 6a: Division by std (uniform std across channels)
image = image / 255.0
image = (image - mean) / std  # If std=0.229 for all channels
```
**Config**: `net_scale_factor: 0.0171302428` *(1/(255×0.229))*, `offsets: "2.118;1.991;1.773"` *(mean/std)*

⚠️ **Note**: This only works if std is the same for all channels. For per-channel std (e.g., ImageNet `std=[0.229, 0.224, 0.225]`), you need to bake normalization into the ONNX model or use a custom preprocessing plugin, as DeepStream only supports a single scale factor.

---

```python
# Example 6b: Multiplication by scale factor d
image = image * d / 255.0  # Where d might be 1/std or a custom constant
```
**Config**: `net_scale_factor: d/255` *(if d=0.5, then 0.5/255 = 0.00196078)*

**Note**: The scale factor becomes `d/255` instead of `1/255`. This pattern is used when training applies a custom scale factor `d`.

---

## The Formula

DeepStream applies:
```
output_pixel = (input_pixel × net_scale_factor) - offset
```

### Examples:

**[0,1] normalization** (no offsets):
```
output = pixel × 0.00392156862745098
output = pixel / 255
Result: [0, 1] range
```

**[-1,1] normalization** (with offsets):
```
output = (pixel × 0.00392156862745098) - 0.5
output = (pixel / 255) - 0.5
Result: [-0.5, 0.5] range → typically then multiplied by 2 in the model
```

**ImageNet normalization**:
```
For red channel:
output = (pixel × 0.00392156862745098) - 0.485
output = (pixel / 255) - 0.485
```

---

## Common Mistakes

### ❌ Mistake 1: Using default values without checking training
```yaml
# Default assumes [0,1] normalization
net-scale-factor: 0.00392156862745098
# No offsets
```
**Problem**: If your model was trained with ImageNet normalization or [-1,1] range, this will give wrong results!

### ❌ Mistake 2: Forgetting per-channel offsets
```python
# Training code used:
image = (image / 255) - 0.5  # [-1,1] normalization
```
```yaml
# But config only sets scale factor:
net-scale-factor: 0.00392156862745098
# Missing: offsets: "127.5;127.5;127.5"
```
**Problem**: Mean subtraction is missing, results will be incorrect!

### ❌ Mistake 3: Wrong offset values
```yaml
# Model trained with ImageNet mean
net-scale-factor: 0.00392156862745098
offsets: "127.5;127.5;127.5"  # WRONG! Should be ImageNet means
```
**Problem**: Using wrong per-channel values will shift colors incorrectly!

---

## Verification Checklist

Before deploying your config:

- [ ] I checked my training code for the exact preprocessing steps
- [ ] I verified the `net-scale-factor` matches the scaling in training
- [ ] I verified the `offsets` match any mean subtraction in training
- [ ] If training had NO mean subtraction, I did NOT set offsets
- [ ] If training had per-channel means, I set the correct RGB values
- [ ] I tested with a known input and compared results

---

## When in Doubt

1. **Test with known inputs**: Use images with known pixel values
2. **Compare outputs**: Run the same image through training and inference
3. **Check model documentation**: Look for preprocessing specifications
4. **Ask the model creator**: They should know the exact normalization used

---

## Real-World Examples

### YOLOv5
```yaml
net-scale-factor: 0.00392156862745098  # Scales to [0,1]
# No offsets - training used simple /255 normalization
```

### ResNet-50 (ImageNet pre-trained)
```yaml
net-scale-factor: 0.00392156862745098  # Scales to [0,1]
offsets: "123.675;116.28;103.53"  # ImageNet per-channel means
```

### TAO ChangeNet ([-1,1] range)
```yaml
net-scale-factor: 0.00392156862745098  # Scales to [0,1]
offsets: "127.5;127.5;127.5"  # Subtracts 0.5 per channel
```

---

## Remember

> **Both `net-scale-factor` AND `offsets` are equally critical!**
>
> Using the wrong scale factor OR wrong per-channel offsets will result in poor inference accuracy, regardless of how good your model is.

For detailed information, see:
- [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - Comprehensive guide
- [EXAMPLE_DEEPSTREAM_CONFIG.md](EXAMPLE_DEEPSTREAM_CONFIG.md) - Usage examples

