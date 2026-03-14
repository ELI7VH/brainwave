---
queue: corpo/brand-helper
date: 2026-03-14
status: pending-push
---

# Logo Processing: brainwave

## What was done
- Source: `blog-media/brainwave-logo.png` (robot with fedora, green brain, dark circle frame)
- Cropped to circle with hat extending past the circle boundary
- Method: PIL/numpy brightness threshold (<200) to detect hat pixels above circle top, alpha-composited onto transparent background
- Output squared for avatar use (1715x1715), anchored to bottom
- Saved as `repos/brainwave/logo.png` (transparent PNG)
- Hero banner copied as-is to `repos/brainwave/hero.png`

## Technique (reusable)
```python
# Circle mask for body, brightness-threshold mask for elements extending past circle
# 1. Detect circle center/radius from source
# 2. Draw filled ellipse mask
# 3. For pixels above circle: include if brightness < 200 (non-background)
# 4. Composite, crop to bbox, square-pad
```

## Parameters to tune
- `cy = h * 0.54` — circle vertical center (shifts with different compositions)
- `r = min(w,h) * 0.42` — circle radius
- brightness threshold `200` — depends on background color

## Assets
- Logo: https://github.com/ELI7VH/brainwave/blob/main/logo.png
- Hero: https://github.com/ELI7VH/brainwave/blob/main/hero.png
