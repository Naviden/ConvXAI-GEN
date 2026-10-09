"""Publication demo. Run: streamlit run interface.py."""
import io
import os
from pathlib import Path
import streamlit as st
from convxai.adapters import DATASETS, MODELS
from convxai.locales import LANGUAGES, text
from convxai.service import Service, chart_png

UI = {
 "en": ("Language", "Dataset", "Reset conversation", "Ask for an explanation", "Listen", "Audio unavailable", "Checked evidence", "Template mode (no LLM calls)"),
 "it": ("Lingua", "Dataset", "Reimposta conversazione", "Chiedi una spiegazione", "Ascolta", "Audio non disponibile", "Evidenza verificata", "Modalità template (senza chiamate LLM)"),
 "es": ("Idioma", "Conjunto de datos", "Reiniciar conversación", "Pide una explicación", "Escuchar", "Audio no disponible", "Evidencia verificada", "Modo plantilla (sin llamadas al LLM)"),
 "fr": ("Langue", "Jeu de données", "Réinitialiser la conversation", "Demandez une explication", "Écouter", "Audio indisponible", "Données vérifiées", "Mode modèle de texte (sans appel LLM)"),
 "de": ("Sprache", "Datensatz", "Gespräch zurücksetzen", "Fragen Sie nach einer Erklärung", "Anhören", "Audio nicht verfügbar", "Geprüfte Evidenz", "Vorlagenmodus (ohne LLM-Aufrufe)"),
}

def main():
    st.set_page_config(page_title="ConvXAI-GEN", layout="wide")
    prior_lang = st.session_state.get("language", "en")
    labels = UI[prior_lang]
    lang = st.sidebar.selectbox(labels[0], list(LANGUAGES), format_func=LANGUAGES.get, key="language")
    labels = UI[lang]
    dataset = st.sidebar.selectbox(labels[1], DATASETS)
    use_llm = os.getenv("CONVXAI_USE_LLM") == "1"
    context = (dataset, lang, use_llm)
    reset = st.sidebar.button(labels[2])
    if reset or st.session_state.get("context") != context:
        st.session_state.context = context
        st.session_state.service = Service(Path(__file__).parent, use_llm=use_llm)
        st.session_state.messages = []
    st.title("ConvXAI-GEN")
    if not use_llm: st.caption(labels[7])
    st.info(text(lang, "help"))
    st.caption("global SHAP Random Forest · local SHAP SVM instance 0 · compare with Random Forest")
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.text(entry["content"])
            if entry.get("chart"): st.image(entry["chart"])
            if entry.get("comparison_chart"): st.image(entry["comparison_chart"])
    question = st.chat_input(labels[3], max_chars=8000)
    if question:
        service = st.session_state.service
        result = service.process(question, dataset, lang)
        entry = {"role": "assistant", "content": result["response"]}
        if result.get("evidence"):
            entry["chart"] = chart_png(result["evidence"], lang)
        if result.get("comparison_evidence"):
            entry["comparison_chart"] = chart_png(result["comparison_evidence"], lang)
        st.session_state.messages.extend([{"role": "user", "content": question}, entry])
        st.session_state.messages = st.session_state.messages[-40:]
        st.rerun()
    if st.session_state.messages and st.button(labels[4]):
        # Explicit click sends the latest response to the external gTTS service.
        try:
            from gtts import gTTS
            audio = io.BytesIO()
            gTTS(st.session_state.messages[-1]["content"][:4000], lang=lang).write_to_fp(audio)
            st.audio(audio.getvalue(), format="audio/mp3")
        except Exception:
            st.warning(labels[5])

if __name__ == "__main__":
    main()
