import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import io
import re

# --- Optional OCR + small LLM imports (all CPU-friendly) ---
import easyocr
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Chart → Interpretation", layout="wide")
st.title("Chart → Interpretation (OCR + VLM hybrid, free + local)")


@st.cache_resource
def get_ocr_reader(lang=("en",)):
    return easyocr.Reader(lang, gpu=False)

@st.cache_resource
def get_tiny_llm():
    # Small, fully local instruction-tuned LLM that runs on CPU.
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return tok, model

def run_tiny_llm(prompt, max_new_tokens=300):
    tok, model = get_tiny_llm()
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)

# ---- Optional VLM (Qwen2-VL 2B) ----
@st.cache_resource
def get_vlm():
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        return processor, model
    except Exception as e:
        st.info("VLM not loaded (OK). We'll rely on OCR+LLM. Error: {}".format(str(e)))
        return None, None

def ask_vlm(image, question):
    processor, model = get_vlm()
    if model is None:
        return None
    try:
        msgs = [
            {"role": "user", "content": [{"type":"image", "image": image}, {"type":"text","text": question}]},
        ]
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        inputs = inputs.to("cpu")
        with torch.no_grad():
            output_ids = model.generate(inputs=inputs, max_new_tokens=256)
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "assistant" in text:
            text = text.split("assistant")[-1].strip()
        return text.strip()
    except Exception as e:
        st.warning(f"VLM failed: {e}")
        return None

def basic_preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_extract(img: Image.Image):
    reader = get_ocr_reader()
    res = reader.readtext(np.array(img))
    lines = []
    for bbox, text, conf in res:
        if conf > 0.3 and text.strip():
            lines.append(text.strip())
    raw_text = "\n".join(lines)
    return raw_text

def heuristic_title_subtitle(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ""
    subtitle = ""
    if lines:
        title = lines[0]
        if len(lines) > 1:
            subtitle = lines[1]
    return title, subtitle

TEMPLATE_PROMPT = """
You are an analyst. Write a short, plain-English interpretation of a chart based on the extracted texts and any visual cues.

Rules:
- 3–6 sentences max.
- Focus on the *message*, not the visual details.
- If there is a histogram or bar chart with values concentrated near zero, say that most estimates are very small.
- If confidence is low, say so.

Extracted text:
{ocr_text}

Observed visual cues:
{visual_notes}

Now write the interpretation.
"""

st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Analysis mode", ["Auto (VLM -> OCR)", "OCR + LLM only"])
st.sidebar.write("Tip: If VLM isn't available on your hardware, choose 'OCR + LLM only'.")

uploaded = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    img = Image.open(uploaded)
    img = basic_preprocess(img)
    st.image(img, caption="Input chart", use_column_width=True)

    visual_notes = ""
    interpretation = None

    if mode == "Auto (VLM -> OCR)":
        vlm_q = "Summarize the main finding of this chart in 3-5 sentences in plain English. Focus on what the data shows, not design."
        vlm_ans = ask_vlm(img, vlm_q)
        if vlm_ans and len(vlm_ans.split()) > 12:
            interpretation = vlm_ans
        else:
            st.info("Falling back to OCR + small LLM...")

    if interpretation is None:
        text = ocr_extract(img)
        title, subtitle = heuristic_title_subtitle(text)
        if title:
            visual_notes += f"Detected title: '{title}'. "
        if subtitle:
            visual_notes += f"Detected subtitle: '{subtitle}'. "

        import cv2
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        edges = cv2.Canny(g, 100, 200)
        v_count = np.sum(edges[:, ::8] > 0)
        h_count = np.sum(edges[::8, :] > 0)
        if v_count > h_count * 1.2:
            visual_notes += "Likely a bar chart or histogram with many vertical edges. "
        if text.strip() == "":
            text = "(no text found)"

        prompt = TEMPLATE_PROMPT.format(ocr_text=text[:1500], visual_notes=visual_notes)
        interpretation = run_tiny_llm(prompt)
        if "Interpretation" in interpretation:
            interpretation = interpretation.split("Interpretation")[-1].strip()
        sents = re.split(r"(?<=[.!?])\s+", interpretation)
        interpretation = " ".join(sents[:6]).strip()

    st.subheader("Generated interpretation")
    st.write(interpretation)

    st.download_button("Download interpretation (.txt)", data=interpretation, file_name="interpretation.txt")

st.markdown("""
---
**Notes**

- This demo runs fully on CPU by default. If your hardware has a GPU, add `bitsandbytes` to speed up.
- Optional VLM: Qwen2-VL-2B-Instruct (free). If it fails to load, the app falls back to OCR + TinyLlama.
- OCR uses EasyOCR, which is robust for chart titles and legends. For production, consider PaddleOCR or DocTR.
""")
