"""Headless tests of the Streamlit UI using Streamlit's AppTest harness."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 180
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_loads_without_errors(app):
    assert not app.exception
    assert not app.error
    assert app.title[0].value == "Day Trading Forecaster"
    assert [t.label for t in app.tabs] == ["Data & EDA", "ARIMA", "SARIMA", "LSTM", "Compare"]


def test_compare_tab_prompts_before_any_run(app):
    assert any("Run at least one model" in i.value for i in app.info)


def test_run_arima_shows_metrics(app):
    app.button[0].click().run()  # "Run ARIMA"
    assert not app.exception and not app.error
    labels = [m.label for m in app.metric]
    assert {"MAE", "RMSE", "R²", "Direction accuracy"} <= set(labels)
    assert "ARIMA" in app.session_state["results"]


def test_train_lstm_with_one_epoch(app):
    lstm_button = next(b for b in app.button if b.label == "Train LSTM")
    app.number_input(key="lstm_epochs").set_value(1)
    lstm_button.click().run()
    assert not app.exception and not app.error
    assert "LSTM" in app.session_state["results"]
