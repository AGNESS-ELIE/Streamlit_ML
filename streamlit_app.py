import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="English → Darija Translator", layout="centered")

st.title("🇲🇦 English → Moroccan Darija Translator")

english_text = st.text_area("Enter English text", height=120)

model_name = "mistralai/Mistral-7B"  # Free, open-source
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        prompt = f"Translate the following English text into natural spoken Moroccan Darija:\n{english_text}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown(f"**Darija Translation:** {translation}")
