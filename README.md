# RF-DETR-Medium WebGPU — Browser-Based Real-Time Object Detection

> Run **RF-DETR-Medium** — a state-of-the-art real-time detection transformer from Roboflow — **100% locally in your browser** using WebGPU acceleration. No server, no API calls, no cloud inference.

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Model](https://img.shields.io/badge/model-RF--DETR--Medium-orange)
![Runtime](https://img.shields.io/badge/runtime-WebGPU-purple)
![Library](https://img.shields.io/badge/library-Transformers.js%20v4-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Facts](#key-facts)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [WebGPU Browser Support](#webgpu-browser-support)
- [Deployment](#deployment)
- [Model Details](#model-details)
- [Performance Benchmarks](#performance-benchmarks)
- [API Reference](#api-reference)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project runs the **RF-DETR-Medium** object detection model entirely in the browser using **WebGPU** for GPU-accelerated inference. It leverages [Transformers.js](https://github.com/huggingface/transformers.js) (v4) and the [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) WebGPU backend to execute the model locally — no data ever leaves the user's device.

The demo supports:
- 📷 **Live webcam detection** with real-time bounding box overlays
- 🎬 **Video file upload** for offline detection on recorded video
- 🎚️ **Adjustable confidence threshold** (0.00 – 1.00)
- 🏷️ **COCO label filtering** (e.g., show only "person, car")
- ⏯️ **Pause / Play** controls
- 📊 **Live FPS counter**

---

## Key Facts

| Property | Value |
|---|---|
| **Model** | RF-DETR-Medium (`Roboflow/rf-detr-medium`) |
| **ONNX Model** | `onnx-community/rfdetr_medium-ONNX` |
| **Architecture** | rf_detr (DINOv2 ViT backbone + Deformable DETR) |
| **Task** | Object Detection (80 COCO classes) |
| **Parameters** | 33.7M |
| **Input Resolution** | 576×576 |
| **COCO AP50** | 73.6 |
| **COCO AP50:95** | 54.7 |
| **RF100-VL AP50** | 87.4 |
| **RF100-VL AP50:95** | 61.2 |
| **Server Latency (T4, TensorRT FP16)** | 4.4 ms |
| **License** | Apache 2.0 |
| **Paper** | [arXiv:2511.09554](https://arxiv.org/abs/2511.09554) — Accepted at ICLR 2026 |
| **JS Library** | Transformers.js v4 (`@huggingface/transformers`) |
| **Inference Backend** | ONNX Runtime Web (WebGPU EP) |
| **Precision** | fp32 (default for WebGPU) |

---

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Webcam /   │────▶│  Canvas (frame │────▶│  RF-DETR-Medium   │────▶│  Bounding     │
│  Video File │     │  extraction)     │     │  ONNX (WebGPU)    │     │  Box Overlay  │
└─────────────┘     └──────────────────┘     └───────────────────┘     └──────────────┘
                           │                         ▲
                           │ │
                    requestAnimationFrame     onnx-community/
 (60 FPS loop)           rfdetr_medium-ONNX
```

1. **Frame Capture** — A hidden `<canvas>` element draws the current video frame (from webcam or file).
2. **Model Inference** — The canvas is passed to the Transformers.js `object-detection` pipeline, which runs the ONNX model on the GPU via WebGPU.
3. **Result Rendering** — Detected objects (label, score, bounding box) are drawn as colored overlays on a `<canvas>` positioned above the `<video>` element.
4. **Loop** — The process repeats via `requestAnimationFrame` for real-time continuous detection.

---

## Live Demo

**Hugging Face Space:** [https://huggingface.co/spaces/webml-community/RF-DETR-Medium-WebGPU](https://huggingface.co/spaces/webml-community/RF-DETR-Medium-WebGPU)

---

## Tech Stack

| Component | Technology | Version / Source |
|---|---|---|
| ML Library | Transformers.js | `@huggingface/transformers@next` (v4) |
| Inference Runtime | ONNX Runtime Web | WebGPU execution provider |
| Model Format | ONNX | `onnx-community/rfdetr_medium-ONNX` |
| Frontend | Vanilla HTML / CSS / JS | No build tools, no frameworks |
| CDN | jsDelivr | `https://cdn.jsdelivr.net/npm/@huggingface/transformers@next` |
| UI Icons | Inline SVG | Feather-style icons |

---

## Quick Start

### Prerequisites

- A **WebGPU-enabled browser** (see [Browser Support](#webgpu-browser-support))
- A webcam (optional — video file upload also supported)

### Option A: Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/webml-community/RF-DETR-Medium-WebGPU
cd RF-DETR-Medium-WebGPU

# Serve with any static file server, e.g.:
npx serve .
# or
python -m http.server 8000
```

Open `http://localhost:8000` in a WebGPU-enabled browser.

### Option B: Use as a Single HTML File

The entire app is a single `index.html` file (with `style.css`). Drop both files into any static host:

```
vercel deploy # Vercel
netlify deploy   # Netlify
# or any static host (GitHub Pages, Cloudflare Pages, etc.)
```

### Option C: Programmatic Usage (NPM)

```bash
npm install @huggingface/transformers
```

```js
import { pipeline } from '@huggingface/transformers';

// Load the model with WebGPU acceleration
const detector = await pipeline(
  'object-detection',
  'onnx-community/rfdetr_medium-ONNX',
  { device: 'webgpu', dtype: 'fp32' }
);

// Run detection on an image
const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, { threshold: 0.75, percentage: true });
console.log(output);
// → [{ score: 0.96, label: 'cat', box: { xmin: 0.12, ymin: 0.15, xmax: 0.48, ymax: 0.82 } }, ...]
```

---

## Project Structure

```
RF-DETR-Medium-WebGPU/
├── index.html       # Main application (HTML + embedded JS module)
├── style.css        # Styling
├── README.md        # This file
└── .gitattributes   # LFS config
```

### `index.html` — Key Sections

| Section | Lines (approx.) | Description |
|---|---|---|
| HTML Structure | 1–60 | Video element, canvas overlay, controls, status overlay |
| Imports | 61 | `import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@next'` |
| State Variables | 63–80 | `detector`, `threshold`, `allowedLabels`, `paused`, `webcamStream` |
| Color System | 82–92 | 6-color cycling palette for bounding box labels |
| Resize / DPR | 94–120 | High-DPI overlay canvas sizing + `object-fit: contain` rect math |
| Source Switching | 122–165 | Webcam ↔ file toggle, `getUserMedia`, object URL management |
| Model Loading | 170–185 | `pipeline('object-detection', 'onnx-community/rfdetr_medium-ONNX', { device: 'webgpu', dtype: 'fp32' })` |
| Warmup | 187–192 | Single inference pass to trigger shader compilation |
| Render Loop | 195–215 | `requestAnimationFrame` → draw frame → infer → draw boxes |
| Box Drawing | 217–255 | Canvas2D rendering with rounded rects, label backgrounds, text |

---

## Configuration

### Adjustable Parameters (UI)

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Threshold** | 0.50 | 0.00 – 1.00 | Minimum confidence score for displaying a detection |
| **Allowed Labels** | (none) | COCO class names | Comma-separated filter, e.g. `person, car, dog` |

### Pipeline Options (Code)

```js
const detector = await pipeline('object-detection', 'onnx-community/rfdetr_medium-ONNX', {
  device: 'webgpu', // 'webgpu' | 'wasm' (fallback)
  dtype: 'fp32',       // 'fp32' | 'fp16' | 'q8' | 'q4'
});

const results = await detector(inputCanvas, {
  threshold: 0.5,      // confidence threshold
  percentage: true,    // return bbox coords as percentages (0–1) vs pixels
});
```

### Quantization Options

| dtype | Size (approx.) | Precision | Use Case |
|---|---|---|---|
| `fp32` | ~135 MB | Full | Default for WebGPU; best accuracy |
| `fp16` | ~67 MB | Half | Faster load, slightly lower accuracy |
| `q8` | ~34 MB | 8-bit | Default for WASM fallback; good balance |
| `q4` | ~17 MB | 4-bit | Fastest load; most accuracy loss |

---

## WebGPU Browser Support

WebGPU global support is approximately **70%** as of October 2024.

| Browser | WebGPU Status | How to Enable |
|---|---|---|
| **Chrome / Edge** (Windows) | ✅ Supported (v113+) | Enabled by default (v121+ for fp16) |
| **Chrome / Edge** (macOS) | ✅ Supported | Enabled by default |
| **Chrome / Edge** (Android) | ✅ Supported (v121+) | Enabled by default |
| **Chrome / Edge** (iOS) | ❌ Not supported | — |
| **Firefox** | ⚠️ Experimental | Enable `dom.webgpu.enabled` flag |
| **Safari** (macOS/iOS) | ⚠️ Experimental | Enable `WebGPU` feature flag |

> **Note:** If WebGPU is unavailable, the app will fail to load the model. For a WASM fallback, change `device: 'webgpu'` to `device: 'wasm'` and `dtype` to `'q8'`.

---

## Deployment

### Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from project root
vercel

# Production deployment
vercel --prod
```

The project is a static site — no server-side rendering or API routes required.

### Deploy to Netlify

```bash
netlify deploy --dir=. --prod
```

### Deploy to GitHub Pages

```bash
git push origin main
# Enable GitHub Pages in repo settings → Pages → Source: main branch
```

### Deploy to Cloudflare Pages

```bash
npx wrangler pages deploy .
```

### Environment Variables

None required. The model is fetched client-side from the Hugging Face Hub CDN.

---

## Model Details

### RF-DETR-Medium

| Property | Value |
|---|---|
| **Hugging Face Model** | [`Roboflow/rf-detr-medium`](https://huggingface.co/Roboflow/rf-detr-medium) |
| **ONNX Export** | [`onnx-community/rfdetr_medium-ONNX`](https://huggingface.co/onnx-community/rfdetr_medium-ONNX) |
| **Architecture** | RF-DETR (DINOv2 backbone + Deformable DETR decoder) |
| **Backbone** | DINOv2 Vision Transformer |
| **Detection Head** | Deformable DETR (single-scale feature maps) |
| **NAS Method** | Weight-sharing neural architecture search (scheduler-free) |
| **Training Data** | Microsoft COCO (80 classes) |
| **Pre-training** | DINOv2 self-supervised ViT |
| **Paper** | [arXiv:2511.09554](https://arxiv.org/abs/2511.09554) |
| **Conference** | ICLR 2026 |
| **Authors** | Isaac Robinson, Peter Robicheaux, Matvei Popov, Deva Ramanan, Neehar Peri |
| **Code** | [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr) |

### Model Family

| Variant | Params (M) | Resolution | COCO AP50:95 | Latency (ms) | License |
|---|---|---|---|---|---|
| RF-DETR-N | 30.5 | 384×384 | 48.4 | 2.3 | Apache 2.0 |
| RF-DETR-S | 32.1 | 512×512 | 53.0 | 3.5 | Apache 2.0 |
| **RF-DETR-M** | **33.7** | **576×576** | **54.7** | **4.4** | **Apache 2.0** |
| RF-DETR-L | 33.9 | 704×704 | 56.5 | 6.8 | Apache 2.0 |
| RF-DETR-XL | 126.4 | 700×700 | 58.6 | 11.5 | PML 1.0 |
| RF-DETR-2XL | 126.9 | 880×880 | 60.1 | 17.2 | PML 1.0 |

> Latency measured on NVIDIA T4 with TensorRT 10.4, CUDA 12.4, FP16, batch size 1.

---

## Performance Benchmarks

### COCO Object Detection

| Metric | RF-DETR-Medium | YOLOv11x (comparison) |
|---|---|---|
| **AP50** | 73.6 | — |
| **AP50:95** | 54.7 | 54.7 |
| **Latency (T4)** | 4.4 ms | 10.5 ms |
| **Params** | 33.7M | — |
| **Resolution** | 576×576 | — |

### RF100-VL Domain Shift| Metric | RF-DETR-Medium |
|---|---|
| **AP50** | 87.4 |
| **AP50:95** | 61.2 |

### Key Comparisons (from paper)

- RF-DETR (nano) beats **D-FINE (nano)** by **+5.3 AP** at similar latency on COCO.
- RF-DETR (2x-large) outperforms **GroundingDINO (tiny)** by **+1.2 AP** on RF100-VL while running **20× faster**.
- RF-DETR (2x-large) is the **first real-time detector to exceed 60 AP** on COCO.
- RF-DETR-L outperforms **YOLOv11x** (56.5 vs 54.7 AP50:95) at lower latency (6.8 vs 10.5 ms).

---

## API Reference

### `pipeline('object-detection', model, options)`

Creates an object detection pipeline.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `task` | `string` | ✅ | Must be `'object-detection'` |
| `model` | `string` | ✅ | Model ID on Hugging Face Hub |
| `options.device` | `string` | ❌ | `'webgpu'` or `'wasm'` (default: `'wasm'`) |
| `options.dtype` | `string` | ❌ | `'fp32'`, `'fp16'`, `'q8'`, `'q4'` |

### `detector(input, options)`

Runs inference on an image or canvas.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input` | `string \| HTMLCanvasElement \| ImageData` | ✅ | Input image/frame |
| `options.threshold` | `number` | ❌ | Confidence threshold (default: 0.5) |
| `options.percentage` | `boolean` | ❌ | Return bbox as percentages (default: false) |

**Returns:** `Array<{ score: number, label: string, box: { xmin, ymin, xmax, ymax } }>`

---

## License

- **Model:** Apache 2.0 ([Roboflow/rf-detr-medium](https://huggingface.co/Roboflow/rf-detr-medium))
- **ONNX Export:** Apache 2.0 ([onnx-community/rfdetr_medium-ONNX](https://huggingface.co/onnx-community/rfdetr_medium-ONNX))
- **Transformers.js:** Apache 2.0 ([huggingface/transformers.js](https://github.com/huggingface/transformers.js))
- **Demo Code:** See Space repository for details

---

## Acknowledgements

- **Roboflow** — For developing RF-DETR and releasing it under Apache 2.0
- **Hugging Face** — For Transformers.js, the ONNX community exports, and the Hub- **webml-community** — For the WebGPU demo Space
- **ONNX Runtime** — For the WebGPU execution provider
- **DINOv2** — For the pre-trained vision transformer backbone

### Citation

```bibtex
@article{robinson2025rfdetr,
  title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author={Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  journal={arXiv preprint arXiv:2511.09554},
  year={2025}
}
```

---

*Built with [Transformers.js](https://github.com/huggingface/transformers.js) · Powered by [WebGPU](https://www.w3.org/TR/webgpu/) · Model by [Roboflow](https://roboflow.com/)*
---

## 🚀 Live Demo: `/RF-DETR-Medium-WebGPU`

A **static Space** (no server backend) that runs RF-DETR-Medium entirely client-side in the browser using WebGPU acceleration.

| Property | Value |
|---|---|
| **SDK** | Static (pure client-side HTML/JS) |
| **Likes** | 19 |
| **Created** | 16 Feb 2026 |

Everything runs on the user's GPU via the browser's WebGPU API — no server inference is involved.

---

## 📦 ONNX Model: `onnx-community/rfdetr_medium-ONNX`

This is the model the Space uses. It's an ONNX-format export of RF-DETR-Medium, specifically packaged for **Transformers.js** (the JavaScript library).

| Property | Value |
|---|---|
| **Model** | [hf.co/onnx-community/rfdetr_medium-ONNX](https://hf.co/onnx-community/rfdetr_medium-ONNX) |
| **Task** | Object Detection |
| **Library** | transformers.js |
| **Architecture** | rf_detr |
| **License** | Apache-2.0 |
| **Downloads** | 2.3K |
| **Likes** | 7 |

### Usage in JavaScript (Transformers.js)

```js
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline(
  'object-detection',
  'onnx-community/rfdetr_medium-ONNX'
);

const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, { threshold: 0.75 });
console.log(output);
```

To run it **on WebGPU**, simply add the `device` option:

```js
const detector = await pipeline(
  'object-detection',
  'onnx-community/rfdetr_medium-ONNX',
  { device: 'webgpu' }
);
```

---

## 📄 Original PyTorch Model: `Roboflow/rf-detr-medium`

| Property | Value |
|---|---|
| **Model** | [hf.co/Roboflow/rf-detr-medium](https://hf.co/Roboflow/rf-detr-medium) |
| **Task** | object-detection |
| **Library** | transformers (PyTorch) |
| **Tags** | safetensors, rf_detr, vision, dataset:coco |
| **License** | Apache-2.0 |
| **ArXiv** | [2511.09554](https://arxiv.org/abs/2511.09554) |
| **Created** | 11 May 2026 |

---

## 📖 Transformers.js WebGPU Documentation

From the official docs ([Running models on WebGPU](https://huggingface.co/docs/transformers.js/guides/webgpu)):

### Key Points

1. **What is WebGPU?** A new web standard for accelerated graphics and compute, succeeding WebGL. It enables GPU-accelerated ML inference directly in the browser.

2. **Browser Support:** ~70% globally as of October 2024. May require feature flags in some browsers:
   - **Firefox:** Enable `dom.webgpu.enabled`
   - **Safari:** Enable the `WebGPU` feature flag
   - **Older Chromium:** Enable `enable-unsafe-webgpu`

3. **How it works:** Thanks to collaboration with [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), enabling WebGPU is as simple as setting `device: 'webgpu'` when loading any model via `pipeline()`.

4. **Quantization options** for resource-constrained browsers:
   - `"fp32"` — default for WebGPU
   - `"fp16"` — half precision
   - `"q8"` — 8-bit quantization (default for WASM fallback)
   - `"q4"` — 4-bit quantization

   Example with quantization:
   ```js
   const pipe = await pipeline('object-detection',
     'onnx-community/rfdetr_medium-ONNX',
     { device: 'webgpu', dtype: 'fp16' }
   );
   ```

5. **Bug reporting:** Due to WebGPU's experimental nature, issues can be reported on [GitHub](https://github.com/huggingface/transformers.js/issues).

---

## 🔗 Summary of All Related Resources

| Resource | Link | Type |
|---|---|---|
| ONNX Model (for browser) | [hf.co/onnx-community/rfdetr_medium-ONNX](https://hf.co/onnx-community/rfdetr_medium-ONNX) | Model |
| Original PyTorch Model | [hf.co/Roboflow/rf-detr-medium](https://hf.co/Roboflow/rf-detr-medium) | Model |
| Transformers.js WebGPU Docs | [huggingface.co/docs/transformers.js/guides/webgpu](https://huggingface.co/docs/transformers.js/guides/webgpu) | Docs |
| CoreML Variant | [hf.co/pzoltowski/rfdetr-medium-fp16-coreml](https://hf.co/pzoltowski/rfdetr-medium-fp16-coreml) | Model |
| Segmentation Variant | [hf.co/Roboflow/rf-detr-seg-medium](https://hf.co/Roboflow/rf-detr-seg-medium) | Model |
