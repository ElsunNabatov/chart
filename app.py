import streamlit as st
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="ChartVLM (Streamlit wrapper)", layout="wide")
st.title("ChartVLM-powered Chart → Interpretation")

REPO_URL = "https://github.com/UniModal4Reasoning/ChartVLM.git"
REPO_DIR = Path("ChartVLM")

@st.cache_resource
def clone_repo():
    if not REPO_DIR.exists():
        st.write("Cloning ChartVLM...")
        result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL], capture_output=True, text=True)
        if result.returncode != 0:
            st.error("Failed to clone ChartVLM repo. Error:\n" + result.stderr)
            raise RuntimeError(result.stderr)
    return str(REPO_DIR.resolve())

def build_base_command(task:str, image_path:str, ckpt_dir:str, extra_args:str=""):
    """We call ChartVLM/inference.py via subprocess.
    Adjust flags according to the repo README if names change.
    """
    cmd = [
        sys.executable, "inference.py",
        "--task", task,
        "--image_path", image_path,
        "--ckpt_dir", ckpt_dir,
    ]
    if extra_args:
        cmd += extra_args.split()
    return cmd

def run_inference(task, image_path, ckpt_dir, extra_args):
    repo = clone_repo()
    with st.status("Running ChartVLM inference...", expanded=True) as status:
        cwd = str(REPO_DIR)
        cmd = build_base_command(task, image_path, ckpt_dir, extra_args)
        st.write("Command:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        st.write(proc.stdout)
        if proc.returncode != 0:
            st.error(proc.stderr)
            raise RuntimeError("Inference failed")
        status.update(label="Done", state="complete")
        # naive parse: look for last non-empty line as result
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        return lines[-1] if lines else proc.stdout

st.sidebar.header("Model checkpoints")
st.sidebar.write("Download ChartVLM-base/large from their HF page and paste the path.")
ckpt_dir = st.sidebar.text_input("Checkpoint directory (on disk)", value="/mount/chartvlm_ckpts")

st.sidebar.header("Task")
task = st.sidebar.selectbox("Choose task", ["summarize", "describe", "qa"])
extra_args = st.sidebar.text_input("Extra CLI args (advanced)", value="")

uploaded = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input chart", use_column_width=True)

    # save temp image for CLI
    tmp_dir = Path("tmp_inputs"); tmp_dir.mkdir(exist_ok=True)
    tmp_img_path = tmp_dir / uploaded.name
    img.save(tmp_img_path)

    if st.button("Run ChartVLM"):
        if not ckpt_dir or not os.path.exists(ckpt_dir):
            st.error("Checkpoint directory not found. Please download the pretrained weights and set the path in the sidebar.")
        else:
            try:
                result = run_inference(task, str(tmp_img_path), ckpt_dir, extra_args)
                st.subheader("Model Output")
                st.write(result)
            except Exception as e:
                st.exception(e)

st.markdown("""
---

### Setup notes
- This wrapper **clones the ChartVLM repo** and shells out to `inference.py`.
- You must **download ChartVLM checkpoints** (base/large) from their Hugging Face page and point `Checkpoint directory` to that folder.
- If CLI flags differ in your version of ChartVLM, use **Extra CLI args** to pass them through.
- Recommended runtime: **Python 3.10 or 3.11**.

""")
