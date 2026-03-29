# comfyui-CVLFace-Verification

Server-side **ComfyUI** custom nodes for **face verification** with **CVLFace ViT KP-RPE**. Detection and dense landmarks use **InsightFace buffalo_l**; embeddings load from a fixed path under **ComfyUI/models/**.

## Local models only

This pack **does not download** weights, **does not call** the Hugging Face Hub/API, and **does not use** HF tokens. You copy files into the paths below.

Paths resolve from ComfyUI **`folder_paths.models_dir`** (including **`extra_model_paths.yaml`** overrides of the models directory).

### Hugging Face (manual download source for CVLFace)

This pack expects the **WebFace12M** KP-RPE checkpoint tree. Download or clone the snapshot from Hugging Face and copy it into **`models/cvlface/vit_kprpe_webface12m/`** (see layout below).

- **Required checkpoint (matches the fixed folder name):** [minchul/cvlface_adaface_vit_base_kprpe_webface12m](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m)
- **Same architecture, WebFace4M training:** [minchul/cvlface_adaface_vit_base_kprpe_webface4m](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface4m) (not used by this pack unless you retarget code and folder names)
- **Author / related models:** [huggingface.co/minchul](https://huggingface.co/minchul)

**buffalo_l** ONNX files are **not** distributed through these Hugging Face model cards; obtain them from the **InsightFace** model zoo (see layout below).

## Required directory layout

```
ComfyUI/models/cvlface/vit_kprpe_webface12m/
  config.json
  wrapper.py
  models/
  pretrained_model/          # optional if you use root safetensors
  model.safetensors          # typical Hugging Face root weights (supported)
```

Also copy the rest of the published **CVLFace AdaFace ViT base KP-RPE WebFace12M** tree (source: [cvlface_adaface_vit_base_kprpe_webface12m](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m)). Weights may be **`model.safetensors`** at this folder root, or **`pretrained_model/model.pt`** / **`pretrained_model/model.safetensors`**; the loader resolves the path the upstream wrapper expects.

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

If no **CVLFace** nodes show in the Add Node menu, check the ComfyUI console for **`[comfyui-CVLFace-Verification] Failed to import cvlface_nodes`** (usually a missing `pip install -r requirements.txt` dependency).

If loading KP-RPE fails with **`cannot import name 'get_model' from 'models'`**, another custom node (for example **comfyui-rmbg**) has a top-level **`models`** package that was shadowing the checkpoint. This pack isolates the checkpoint on `sys.path` during load and clears that import afterward; update to the latest pack code if you still see the error.

On load you may see **`Failed to import cuda/cpp RPEIndexFunction`** followed by **`setup.py install`** noise: that comes from upstream CVLFace trying to compile optional RPE ops; the model can still run with the slower pure-PyTorch RPE path. **`Tensor.item() cannot be called on meta tensors`** during **CVLFace Loader** is addressed by forcing **`torch.linspace`** onto **CPU** for the duration of **`from_pretrained`** (CVLFace’s `vit.py` uses `linspace(...).item()` for drop-path rates; Transformers/accelerate can still run that under a meta context). The loader also sets **`low_cpu_mem_usage=False`**, **`device_map=None`** when supported, and avoids restoring a **`meta`** default device after load.

## Nodes

In the graph editor, use **Add Node** (double-click / right-click): all pack nodes are grouped under **`CVLFace`**, matching the **`models/cvlface/`** layout.

| Node | Role |
|------|------|
| **CVLFace Loader (KP-RPE)** | Loads KP-RPE from **`models/cvlface/vit_kprpe_webface12m/`**; outputs `FACE_EMBEDDER`. |
| **Face Align (InsightFace)** | `IMAGE` → aligned `IMAGE`, landmarks preview, `FACE_META`. |
| **Face Reference Profile** | Reference `IMAGE`(s) → `FACE_PROFILE`. |
| **Face Compare KP-RPE** | Target `IMAGE` + profile → scores, aggregate, match, previews. |

## License / use

Treat **InsightFace** pretrained ONNX packs per their **non-commercial research** terms where applicable. **CVLFace** follows upstream and dataset licenses. Use **locally** and in compliance with those terms.
