# Standard Deviation Normalization Calculator

## Understanding Std Normalization

### Case 1: Division by Std (Most Common)

When training uses standard deviation normalization by **dividing** by std:
```python
normalized = (pixel/255 - mean) / std
```

This can be rewritten as:
```python
normalized = pixel * (1/(255*std)) - (mean/std)
```

Which maps to DeepStream parameters:
- `net_scale_factor = 1/(255 * std)`
- `offsets = mean/std`

**Example**: If `std = 0.229`, then `net_scale_factor = 1/(255 × 0.229) = 0.0171302428`

---

### Case 2: Multiplication by Scale Factor (Alternative)

If training uses a scaling factor `d` (where `d` might be related to std):
```python
normalized = pixel * d / 255
```

This maps to DeepStream parameters:
- `net_scale_factor = d/255`
- `offsets` depend on any mean subtraction

**Example**: If `d = 0.5`, then `net_scale_factor = 0.5/255 = 0.00196078431`

**Note**: This is less common but may be used in custom preprocessing schemes where `d` could be `1/std` or another scaling factor.

## The Challenge

**DeepStream limitation**: Only supports a **single** `net_scale_factor` value, but many models use **per-channel** std values.

**Example**: ImageNet uses `std=[0.229, 0.224, 0.225]` - different for each channel!

## Solutions

### Solution 1: Uniform Std (Best for DeepStream)

If your model uses the **same std for all channels**, you can directly convert:

#### Example: Uniform std = 0.229
```python
# Training normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.229, 0.229]  # Same for R, G, B
```

#### Calculation:
```python
net_scale_factor = 1 / (255 * 0.229) = 0.0171302428

offsets_R = 0.485 / 0.229 = 2.118122
offsets_G = 0.456 / 0.229 = 1.991266
offsets_B = 0.406 / 0.229 = 1.772926
```

#### DeepStream Config:
```yaml
net-scale-factor: 0.0171302428
offsets: 2.118122;1.991266;1.772926
```

---

### Solution 2: Average Std (Approximation)

If std values are close but not identical, use the average:

#### Example: ImageNet std = [0.229, 0.224, 0.225]
```python
# Training normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]  # Different per channel
```

#### Calculation:
```python
avg_std = (0.229 + 0.224 + 0.225) / 3 = 0.226

net_scale_factor = 1 / (255 * 0.226) = 0.0173510648

offsets_R = 0.485 / 0.226 = 2.146018
offsets_G = 0.456 / 0.226 = 2.017699
offsets_B = 0.406 / 0.226 = 1.796460
```

#### DeepStream Config:
```yaml
net-scale-factor: 0.0173510648
offsets: 2.146018;2.017699;1.796460
```

⚠️ **Trade-off**: This is an approximation. Accuracy may be slightly reduced because we're not using the exact per-channel std values.

---

### Solution 3: Ignore Std (Mean-only, Less Accurate)

Use only mean subtraction, ignore std division:

#### Calculation:
```python
net_scale_factor = 1 / 255 = 0.00392156862745098

offsets_R = 0.485 * 255 = 123.675
offsets_G = 0.456 * 255 = 116.28
offsets_B = 0.406 * 255 = 103.53
```

#### DeepStream Config:
```yaml
net-scale-factor: 0.00392156862745098
offsets: 123.675;116.28;103.53
```

⚠️ **Limitation**: This does NOT include std normalization, only mean subtraction. Results will be less accurate.

---

### Solution 4: Bake into ONNX (Recommended for Per-Channel Std)

The most accurate solution: Include preprocessing in the ONNX model itself.

#### PyTorch Example:
```python
import torch
import torch.nn as nn

class ModelWithPreprocessing(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        # Register mean and std as buffers
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # x is in [0, 255] range
        x = x / 255.0  # Scale to [0, 1]
        x = (x - self.mean) / self.std  # Per-channel normalization
        return self.base_model(x)

# Wrap your model
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
wrapped_model = ModelWithPreprocessing(original_model, mean, std)

# Export to ONNX
torch.onnx.export(wrapped_model, dummy_input, "model_with_preprocessing.onnx")
```

#### DeepStream Config:
```yaml
# No normalization needed - it's in the model!
net-scale-factor: 1.0
# No offsets
```

**Benefits**:
- ✅ Exact per-channel std normalization
- ✅ No accuracy loss
- ✅ Simpler DeepStream config

---

## Quick Calculator

### Method 1: Division by Std (Standard Approach)

**Given Training Parameters**:
- **mean**: [mean_r, mean_g, mean_b]
- **std**: [std_r, std_g, std_b]

**Training formula**: `(pixel/255 - mean) / std`

#### If std is uniform (std_r = std_g = std_b = d):

```
net_scale_factor = 1 / (255 × d)
offset_r = mean_r / d
offset_g = mean_g / d
offset_b = mean_b / d
```

**Example**: If `d = 0.229`, `mean = [0.485, 0.456, 0.406]`
```
net_scale_factor = 1 / (255 × 0.229) = 0.0171302428
offsets = 2.118;1.991;1.773
```

#### If std varies per channel (approximation):

```
avg_std = (std_r + std_g + std_b) / 3
net_scale_factor = 1 / (255 × avg_std)
offset_r = mean_r / avg_std
offset_g = mean_g / avg_std
offset_b = mean_b / avg_std
```

---

### Method 2: Multiplication by Scale Factor

**Given Training Parameters**:
- **scale_factor**: d (a custom scaling constant)

**Training formula**: `pixel × d / 255`

#### Calculation:

```
net_scale_factor = d / 255
```

**Example**: If `d = 0.5`
```
net_scale_factor = 0.5 / 255 = 0.00196078431
```

**Example**: If `d = 1/0.229` (inverse of std)
```
net_scale_factor = (1/0.229) / 255 = 4.366812 / 255 = 0.0171302428
```

**Note**: The scale factor becomes `d/255` instead of `1/255`.

---

## Common Datasets

### ImageNet
```
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

**Recommended**: Bake into ONNX or use average std approximation

### COCO (if normalized)
```
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
Same as ImageNet

### CIFAR-10
```
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
```

**Average std**: 0.2009

---

## Decision Tree

```
Do you use std normalization in training?
│
├─ No → Use simple config:
│       net_scale_factor: 1/255
│       offsets: mean × 255 (if mean subtraction used)
│
└─ Yes → Is std the same for all channels?
    │
    ├─ Yes → Use combined config:
    │        net_scale_factor: 1/(255×std)
    │        offsets: mean/std
    │
    └─ No → Choose approach:
           │
           ├─ Approximation (easier):
           │  net_scale_factor: 1/(255×avg_std)
           │  offsets: mean/avg_std
           │  ⚠️ Slight accuracy loss
           │
           ├─ Mean-only (less accurate):
           │  net_scale_factor: 1/255
           │  offsets: mean × 255
           │  ⚠️ Missing std normalization
           │
           └─ Bake into ONNX (best):
              Include preprocessing in model
              ✅ Full accuracy, per-channel std
```

---

## Verification

To verify your config is correct:

1. **Test with known input**:
   ```python
   test_pixel = 128  # Middle gray

   # Training preprocessing:
   train_value = (test_pixel/255 - mean) / std

   # DeepStream preprocessing:
   ds_value = (test_pixel * net_scale_factor) - offset

   # These should match!
   ```

2. **Compare outputs**: Run inference on the same image in training and DeepStream, compare results

3. **Check accuracy**: Use validation dataset to measure accuracy drop

---

## Python Helper Script

```python
def calculate_deepstream_params(mean, std, use_average=False):
    """
    Calculate DeepStream normalization parameters.

    Args:
        mean: List of per-channel means [R, G, B] (in [0,1] range)
        std: List of per-channel stds [R, G, B]
        use_average: If True, use average std (approximation)

    Returns:
        tuple: (net_scale_factor, offsets_string)
    """
    mean_r, mean_g, mean_b = mean
    std_r, std_g, std_b = std

    # Check if std is uniform
    is_uniform = (std_r == std_g == std_b)

    if is_uniform:
        # Exact calculation for uniform std
        net_scale_factor = 1.0 / (255 * std_r)
        offset_r = mean_r / std_r
        offset_g = mean_g / std_g
        offset_b = mean_b / std_b
        print("✅ Uniform std detected - using exact calculation")
    elif use_average:
        # Approximation using average std
        avg_std = (std_r + std_g + std_b) / 3
        net_scale_factor = 1.0 / (255 * avg_std)
        offset_r = mean_r / avg_std
        offset_g = mean_g / avg_std
        offset_b = mean_b / avg_std
        print(f"⚠️  Using average std approximation: {avg_std:.6f}")
    else:
        print("❌ Per-channel std detected. Consider baking into ONNX!")
        print("   Using average std approximation...")
        avg_std = (std_r + std_g + std_b) / 3
        net_scale_factor = 1.0 / (255 * avg_std)
        offset_r = mean_r / avg_std
        offset_g = mean_g / avg_std
        offset_b = mean_b / avg_std

    offsets = f"{offset_r:.6f};{offset_g:.6f};{offset_b:.6f}"

    print(f"\nDeepStream Config:")
    print(f"  net-scale-factor: {net_scale_factor}")
    print(f"  offsets: {offsets}")

    return net_scale_factor, offsets

# Example usage:
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
calculate_deepstream_params(mean, std, use_average=True)
```

---

## Summary

- **Best accuracy**: Bake std normalization into ONNX model
- **Good approximation**: Use average std (if stds are similar)
- **Uniform std**: Direct conversion works perfectly
- **Mean-only**: Simplest but least accurate (ignores std)

Always verify your configuration with test data before deploying!

