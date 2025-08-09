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

def ensure_ckpts(ckpt_mode, local_dir, hf_repo_id):
    """Return a directory path that contains the checkpoints.
    - ckpt_mode: 'local' or 'download'
    - local_dir: path provided by user
    - hf_repo_id: e.g., 'Org/ChartVLM-base' (you can use any HF repo id)
    """
    if ckpt_mode == "local":
        if not local_dir or not os.path.isdir(local_dir):
            raise FileNotFoundError("Checkpoint directory not found: " + str(local_dir))
        return local_dir
    else:
        # download via huggingface_hub
        from huggingface_hub import snapshot_download
        cache_dir = Path(".hf_ckpts"); cache_dir.mkdir(exist_ok=True)
        st.write(f"Downloading weights from Hugging Face repo: {hf_repo_id} ...")
        path = snapshot_download(repo_id=hf_repo_id, local_dir=cache_dir, local_dir_use_symlinks=False, resume_download=True)
        return path

def build_base_command(task:str, image_path:str, ckpt_dir:str, extra_args:str=""):
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
    clone_repo()
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
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        return lines[-1] if lines else proc.stdout

st.sidebar.header("Model weights")
ckpt_mode = st.sidebar.radio("How to provide checkpoints?", ["Local path", "Download from Hugging Face"], index=0)

local_dir = ""
hf_repo_id = ""

if ckpt_mode == "Local path":
    local_dir = st.sidebar.text_input("Checkpoint directory (absolute or relative path)", value="")
else:
    hf_repo_id = st.sidebar.text_input("Hugging Face repo id (e.g., Org/ChartVLM-base)", value="")

st.sidebar.header("Task")
task = st.sidebar.selectbox("Choose task", ["summarize", "describe", "qa"])
extra_args = st.sidebar.text_input("Extra CLI args (advanced)", value="")

uploaded = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input chart", use_column_width=True)

    tmp_dir = Path("tmp_inputs"); tmp_dir.mkdir(exist_ok=True)
    tmp_img_path = tmp_dir / uploaded.name
    img.save(tmp_img_path)

    if st.button("Run ChartVLM"):
        try:
            ckpt_dir = ensure_ckpts("local" if ckpt_mode=="Local path" else "download", local_dir, hf_repo_id)
        except Exception as e:
            st.error(f"Checkpoint setup failed: {e}")
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
- **Option A (Local path):** Put the downloaded ChartVLM weights in a folder and paste that path in the sidebar.
- **Option B (Download):** Enter a Hugging Face repo id (e.g., `Org/ChartVLM-base`). The app will download into `.hf_ckpts/`.
- This wrapper clones the ChartVLM repo and invokes `inference.py`. If CLI flags differ, pass them via **Extra CLI args**.

""")
