import streamlit as st
from retrieval import answer_question, extract_relevant_years, texts

st.set_page_config(page_title="Disaster QA in Africa", layout="wide")
st.title("🌍 Disaster QA System (Africa, 2015–2025)")
st.markdown("Ask any question about natural disasters in African countries between 2015 and 2025.")

question = st.text_input("Enter your question:", value="What disasters occurred in 2023 in East Africa?") #user input
submit = st.button("Get Answer")

if submit and question.strip():
    with st.spinner("Retrieving relevant context and generating an answer..."):
        years = extract_relevant_years(question)
        st.markdown(f"**Extracted Years:** `{years if years else 'None'}`")

        answer = answer_question(question)
        
        st.markdown("###Answer")
        st.success(answer)

        with st.expander("🔍 Show Top Context (Preview)"):
            filtered_contexts = [t for t in texts if any(str(y) in t for y in years)] if years else texts[:5]
            st.text("\n\n".join(filtered_contexts[:3]))  # only show first 3 chunks