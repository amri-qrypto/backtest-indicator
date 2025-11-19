import pandas as pd

from src.pipelines.single_asset import _plot_equity_curve


def test_plot_equity_curve_saves_image(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"equity_curve": [1.0, 1.1, 1.05, 1.2]}, index=index)

    output_path = tmp_path / "equity.png"
    _plot_equity_curve(df, output_path)

    assert output_path.exists(), "plot image should be written to disk"
    assert output_path.stat().st_size > 0, "plot image should not be empty"
