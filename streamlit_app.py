import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="English → Moroccan Darija Translator",
    page_icon="🇲🇦",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🇲🇦 English → Moroccan Darija Translator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Translate English into natural spoken Moroccan Darija</p>",
    unsafe_allow_html=True
)
st.divider()

# -----------------------------
# INPUT AREA
# -----------------------------
english_text = st.text_area(
    "✍️ Enter English text",
    placeholder="Type something like: How are you doing today?",
    height=140
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "mistralai/Mistral-7B"  # Free, open-source
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # automatically uses GPU if available
        torch_dtype=torch.float16  # speeds up inference
    )
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# TRANSLATION BUTTON
# -----------------------------
if st.button("🔄 Translate", use_container_width=True):
    if english_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Translating to Darija..."):
            prompt = f"""Translate the following English text into natural spoken Moroccan Darija (no formal Arabic):
English: {english_text}
Darija:"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.markdown(
                f"""
                <div style="
                    background-color:#111;
                    padding:15px;
                    border-radius:10px;
                    font-size:18px;
                    color:#00ffcc;
                ">
                {translation}
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit and Mistral-7B</p>",
    unsafe_allow_html=True
)
