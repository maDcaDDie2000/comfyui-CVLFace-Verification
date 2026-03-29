# comfyui-CVLFace-Verification

Server-side **ComfyUI** custom nodes for **face verification** with **CVLFace ViT KP-RPE**. Detection and dense landmarks use **InsightFace buffalo_l**; embeddings load from a fixed path under **ComfyUI/models/**.

## Local models only

This pack **does not download** weights, **does not call** the Hugging Face Hub/API, and **does not use** HF tokens. You copy files into the paths below.

Paths resolve from ComfyUI **`folder_paths.models_dir`** (including **`extra_model_paths.yaml`** overrides of the models directory).

## Required directory layout

```
ComfyUI/models/cvlface/vit_kprpe_webface12m/
  config.json
  wrapper.py
  models/
  pretrained_model/
```

Also place every other file from the published **CVLFace AdaFace ViT base KP-RPE WebFace12M** checkpoint in that same directory, including all weight files from that checkpoint.

```
ComfyUI/models/insightface/models/buffalo_l/
  det_10g.onnx
  2d106det.onnx
  1k3d68.onnx
  genderage.onnx
  w600k_r50.onnx
```

Copy the official **buffalo_l** pack from the InsightFace model zoo so this directory contains these five files and matches that pack.

The loader always reads CVLFace from **`models/cvlface/vit_kprpe_webface12m/`** and InsightFace from **`models/insightface/models/buffalo_l/`**. Those names are fixed in code; there is no folder name setting in the UI.

## Install

1. Clone this repository to **`ComfyUI/custom_nodes/comfyui-cvlface-verification`**.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Populate **`models/cvlface/vit_kprpe_webface12m/`** and **`models/insightface/models/buffalo_l/`** as above.

`transformers` loads the CVLFace tree with **`from_pretrained(local_dir, local_files_only=True)`** only.

## Nodes

| Node | Role |
|------|------|
| **CVLFace Loader (KP-RPE)** | Loads KP-RPE from **`models/cvlface/vit_kprpe_webface12m/`**; outputs `FACE_EMBEDDER`. |
| **Face Align (InsightFace)** | `IMAGE` → aligned `IMAGE`, landmarks preview, `FACE_META`. |
| **Face Reference Profile** | Reference `IMAGE`(s) → `FACE_PROFILE`. |
| **Face Compare KP-RPE** | Target `IMAGE` + profile → scores, aggregate, match, previews. |

## License / use

Treat **InsightFace** pretrained ONNX packs per their **non-commercial research** terms where applicable. **CVLFace** follows upstream and dataset licenses. Use **locally** and in compliance with those terms.
