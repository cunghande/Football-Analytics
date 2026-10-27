import pytest
from pathlib import Path


def test_dashboard_initial_render_has_upload_and_reports():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert len(app.exception) == 0
    assert len(app.tabs) == 2
    assert len(app.get("file_uploader")) == 1
    assert len(app.button) == 1
