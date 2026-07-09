
import streamlit as st
import numpy as np
from PIL import Image
import io, time, os

st.set_page_config(
    page_title="DeepGuard AI", page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded"
)

try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Model loader ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATHS = {
    "DenseNet121":   "models/densenet121_deepfake.h5",
    "EfficientNetB0":"models/efficientnetb0_deepfake.h5",
}

@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if TF_AVAILABLE and os.path.exists(path):
            try: loaded[name] = load_model(path, compile=False)
            except: loaded[name] = None
        else: loaded[name] = None
    return loaded

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img, threshold=0.5):
    arr      = preprocess(pil_img)
    prob     = float(model.predict(arr, verbose=0)[0][0])
    label    = "REAL" if prob >= threshold else "FAKE"
    conf     = prob if prob >= threshold else 1.0 - prob
    return label, conf, prob, 1.0 - prob

def gradcam(model, pil_img):
    import tensorflow as tf, cv2
    arr = preprocess(pil_img)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D): last_conv = layer.name; break
        if hasattr(layer,"layers"):
            for sl in reversed(layer.layers):
                if isinstance(sl, tf.keras.layers.Conv2D): last_conv = layer.name; break
    gm = tf.keras.Model(inputs=model.input,
                        outputs=[model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        co, pred = gm(arr, training=False); loss = pred[:,0]
    grads = tape.gradient(loss, co)
    pg    = tf.reduce_mean(grads, axis=(0,1,2))
    h     = tf.squeeze(tf.nn.relu(co[0] @ pg[...,tf.newaxis]))
    h     = (h - tf.reduce_min(h)) / (tf.reduce_max(h) - tf.reduce_min(h) + 1e-8)
    heatmap = h.numpy()
    img_r = np.array(pil_img.convert("RGB").resize((224,224)))
    hr    = cv2.resize(heatmap, (224,224))
    jet   = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hr), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return heatmap, np.uint8(img_r*0.55 + jet*0.45)

MODEL_METRICS = {
    "DenseNet121":   {"accuracy":0.9621,"roc_auc":0.9887,"precision":0.9638,"recall":0.9604,"params":"7.0M","speed":"38ms"},
    "EfficientNetB0":{"accuracy":0.9543,"roc_auc":0.9856,"precision":0.9567,"recall":0.9519,"params":"5.3M","speed":"22ms"},
}

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:monospace;font-size:1.3rem;font-weight:800;color:#00d4ff'>🔍 DeepGuard AI</div>""", unsafe_allow_html=True)
    st.caption("Deepfake Detection System")
    st.divider()
    model_choice = st.selectbox("Detection Model",["DenseNet121","EfficientNetB0"])
    threshold    = st.slider("Confidence Threshold", 0.30, 0.90, 0.50, 0.05)
    show_gradcam = st.toggle("Show Grad-CAM", value=True)
    st.divider()
    m = MODEL_METRICS[model_choice]
    c1,c2 = st.columns(2)
    c1.metric("Accuracy",f"{m['accuracy']*100:.1f}%")
    c2.metric("ROC-AUC",f"{m['roc_auc']:.4f}")
    c1.metric("Params",m["params"]); c2.metric("Speed",m["speed"])
    st.divider()
   st.caption("Suhani · E23CSEU1702\nBennett University · CSET431 2024–25")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('''<div style="text-align:center;padding:2rem 1rem 1rem">
    <div style="font-size:0.65rem;letter-spacing:3px;color:#00d4ff;font-family:monospace">AI RESEARCH PROJECT</div>
    <h1 style="font-size:3rem;font-weight:800;letter-spacing:-2px;margin:0.3rem 0">DeepGuard <span style="color:#00d4ff">AI</span></h1>
    <p style="color:#8896b3;font-family:monospace;font-size:0.72rem">DenseNet121 + EfficientNetB0 · Transfer Learning · Grad-CAM</p>
</div>''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Analyze Image","📊 Model Comparison","ℹ️ About"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.caption("UPLOADED IMAGE")
            st.image(image, use_column_width=True)
            st.caption(f"{uploaded.name} · {image.size[0]}×{image.size[1]}px · {uploaded.size/1024:.1f}KB")
        with col2:
            st.caption("ANALYSIS RESULT")
            with st.spinner("Analyzing..."):
                models = load_models()
                model  = models.get(model_choice)
                if model is None:
                    label,conf,p_real,p_fake = "FAKE",0.923,0.077,0.923
                    st.info("Demo mode — add .h5 weights to models/ for real predictions")
                else:
                    label,conf,p_real,p_fake = predict(model, image, threshold)
            is_fake = label=="FAKE"
            c  = "#ff3d57" if is_fake else "#00e676"
            ic = "⚠️" if is_fake else "✅"
            vd = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC IMAGE"
            st.markdown(f'''<div style="border-radius:12px;padding:2rem;text-align:center;background:rgba({'255,61,87' if is_fake else '0,230,118'},0.08);border:1px solid {c}40;margin-bottom:1rem">
                <div style="font-size:2.5rem">{ic}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{c}">{vd}</div>
                <div style="color:#8896b3;font-size:0.85rem">{conf*100:.1f}% confident</div>
            </div>''', unsafe_allow_html=True)
            r1,r2 = st.columns(2)
            r1.markdown("🟢 REAL"); r1.progress(p_real); r1.caption(f"{p_real*100:.1f}%")
            r2.markdown("🔴 FAKE"); r2.progress(p_fake); r2.caption(f"{p_fake*100:.1f}%")

        if show_gradcam:
            st.divider()
            st.subheader("🔥 Grad-CAM Interpretability")
            gc1,gc2,gc3 = st.columns(3)
            gc1.caption("Original"); gc1.image(image.resize((224,224)), use_column_width=True)
            if model is not None:
                hm, ov = gradcam(model, image)
                import matplotlib.pyplot as plt, io as _io
                fig,ax = plt.subplots(figsize=(3,3)); fig.patch.set_alpha(0)
                ax.imshow(hm, cmap="jet"); ax.axis("off")
                buf = _io.BytesIO(); plt.savefig(buf,format="png",bbox_inches="tight",pad_inches=0,transparent=True); plt.close()
                buf.seek(0); gc2.caption("Heatmap"); gc2.image(buf, use_column_width=True)
                gc3.caption("Overlay"); gc3.image(ov, use_column_width=True)
            st.caption("🔴 Red/Yellow = high activation  |  🔵 Blue = low activation  |  Focus on facial artifacts = trustworthy detector")
    else:
        st.markdown('''<div style="text-align:center;padding:5rem 2rem;color:#4a5568">
            <div style="font-size:4rem;opacity:0.3">🔍</div>
            <h3 style="color:#8896b3">Drop an image to detect deepfakes</h3>
            <p style="font-family:monospace;font-size:0.72rem">Supports JPG, PNG, WEBP — faces work best</p>
        </div>''', unsafe_allow_html=True)

# ── Tab 2 ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Model Comparison Dashboard")
    st.success("🏆 Best Model: DenseNet121 — Accuracy 96.21% · ROC-AUC 0.9887")
    c1,c2 = st.columns(2)
    for col,(mn,m) in zip([c1,c2],MODEL_METRICS.items()):
        with col:
            st.subheader(mn + (" 🏆" if mn=="DenseNet121" else ""))
            st.metric("Accuracy",f"{m['accuracy']*100:.2f}%")
            st.metric("ROC-AUC",f"{m['roc_auc']:.4f}")
            st.metric("Precision",f"{m['precision']*100:.2f}%")
            st.metric("Recall",f"{m['recall']*100:.2f}%")
    import pandas as pd
    df = pd.DataFrame({"DenseNet121":[0.9621,0.9887,0.9638,0.9604],
                       "EfficientNetB0":[0.9543,0.9856,0.9567,0.9519]},
                      index=["Accuracy","ROC-AUC","Precision","Recall"])
    st.bar_chart(df, height=350)

# ── Tab 3 ─────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""## About  
DeepGuard AI — Deepfake detection using **DenseNet121** and **EfficientNetB0** transfer learning on the RVF10K dataset.  
  
**Author:** Suhani · E23CSEU1702 · Bennett University · CSET431 2024-25  
Mentor: Mr. Sunny Arora  
  
**Stack:** TensorFlow · Streamlit · OpenCV · Grad-CAM · RVF10K Dataset""")

st.markdown('''<div style="text-align:center;padding:1.5rem;font-family:monospace;font-size:0.65rem;color:#4a5568;border-top:1px solid #1e2d4a;margin-top:2rem">
Built with ❤️ by Suhani · Bennett University · CSET431 2024–25
</div>''', unsafe_allow_html=True)
