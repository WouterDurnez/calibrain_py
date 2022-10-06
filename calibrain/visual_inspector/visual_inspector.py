"""
  ___      _ _ _             _
 / __|__ _| (_) |__ _ _ __ _(_)_ _
| (__/ _` | | | '_ \ '_/ _` | | ' \
 \___\__,_|_|_|_.__/_| \__,_|_|_||_|

- Coded on_col Wouter Durnez & Jonas De Bruyne
"""
"""
Visual inspector: use this tool to visually check the data quality each participant
"""

import dash_bootstrap_components as dbc
from dash import Dash, html

from utils.helper import hi

if __name__ == "__main__":
    # Opening message
    hi("CaliBrain -- Visual Inspector")

    # path_to_data = '../data/test_202206021426'
    # data = calibrain.CalibrainData(dir=path_to_data)

    #############
    # Dashboard #
    #############

    # Initialize dashboard app
    app = Dash(
        __name__,
        meta_tags=[
            {
                "name": "viewport",
                "content": "width=device-width, initial-scale=1",
            }
        ],
        external_stylesheets=[dbc.themes.YETI],
    )

    app.title = "CaliBrain - visual inspector"

    app.layout = html.Div(
        id="app-container",
        children=[],
        style={
            "height": "100vh",
            "width": "100vw",
            "position": "relative",
            "display": "block",
        },
    )

    app.run_server(debug=False)
