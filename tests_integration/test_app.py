"""Optional UI tests; do not call external LLM/TTS services."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class AppTests(unittest.TestCase):
    def test_five_language_chat_and_plots(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {'CONVXAI_USE_LLM':'0'}):
            app=AppTest.from_file(str(ROOT/'interface.py'),default_timeout=30).run()
            self.assertFalse(app.exception)
            for lang in ('en','it','es','fr','de'):
                app.sidebar.selectbox[0].set_value(lang).run()
                app.chat_input[0].set_value('local SHAP Random Forest instance 0').run()
                self.assertFalse(app.exception)
                self.assertIn('random_forest',app.chat_message[-1].text[0].value)

    def test_chart_png(self):
        from convxai.service import Service, chart_png
        r=Service(ROOT).process('local SHAP Random Forest instance 0')
        self.assertTrue(chart_png(r['evidence']).startswith(b'\x89PNG'))
