import streamlit as st

st.title("English → Moroccan Darija Translator")

english_text = st.text_area(
    "Enter English text",
    placeholder="Type something in English..."
)

if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        st.write("🔄 Translating...")
        st.success("Darija translation will appear here")
