# Chart → Interpretation (free, local)

This Streamlit app converts chart images into short, plain-English interpretations using a **hybrid** approach:

1. **Visual Language Model (VLM)** — optional: Qwen2-VL 2B (free, open-source).  
2. **OCR + tiny local LLM** — EasyOCR + TinyLlama 1.1B for CPU-friendly fallback.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the VLM cannot load on your hardware, choose **"OCR + LLM only"** in the sidebar.

## Tips
- For best OCR, upload images with at least ~600px width.
- You can swap models: try `llava-hf/llava-onevision-qwen2-0.5b-ov` or `InternVL2-1B` if you have GPU.
