# ⬡ Edge Detection Studio — Streamlit

A beautiful, interactive edge detection app built with Streamlit + OpenCV.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **3 Detection Methods**: Canny, Sobel, Laplacian
- **Real-time parameter tuning** via sliders & radio buttons
- **6+ edge colors**: Cyan, Violet, Orange, Red, White, Yellow, Green, Hot Pink
- **3-panel view**: Original · Edge Map · Overlay
- **Full-resolution preview**
- **One-click export**: overlay PNG, edge map PNG, original PNG
- **Live stats**: resolution, edge density %, method, color

## Methods

| Method | Best for |
|---|---|
| Canny | General purpose — clean, thin edges |
| Sobel | Gradient direction, smooth edges |
| Laplacian | Fine detail, texture edges |
