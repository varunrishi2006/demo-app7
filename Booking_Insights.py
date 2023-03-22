import dash
from dash import dcc, html, Input, Output, State, dash_table
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import datetime
from datetime import datetime as dt
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1" }],
)
# app.title = "Forward Load Analytics Dashboard"

columns = ['Route', 'Flight No.', 'Departure Time', 'Days-to-Departure', 'Forecast', 'Current Booked', 'Last 3 day Bookings']

df = pd.read_csv("C:/Users/varun/forward_load_analytics.csv")
df_product = pd.read_csv("C:/Users/varun/Product.csv")
df_journey = pd.read_csv("C:/Users/varun/Journey.csv")
df_channel = pd.read_csv("C:/Users/varun/Channel.csv")
df_inventory = pd.read_csv("C:/Users/varun/Inventory.csv")
df_booking_curve = pd.read_csv("C:/Users/varun/sample_data.csv")


# Code for initial preprocessing of booking curve dataframe starts here
df_booking_curve['Total Rev (Historical)'] = df_booking_curve['Total Rev (Historical)'].astype(int)
df_booking_curve['Avg Fare (Historical)'] = df_booking_curve['Avg Fare (Historical)'].astype(int)
df_booking_curve['Booked Historical'] = df_booking_curve['Booked Historical'].astype(int)


category_order = CategoricalDtype([
    '0-7',
    '8-15',
    '16-30',
    '31-60',
    '61-90',
    '91-120',
    '>120',
], ordered=True)

df_booking_curve.loc[df_booking_curve['NDO'].between(0, 7, 'both'), 'NDO Range'] = '0-7'
df_booking_curve.loc[df_booking_curve['NDO'].between(8, 15, 'both'), 'NDO Range'] = '8-15'
df_booking_curve.loc[df_booking_curve['NDO'].between(16, 30, 'both'), 'NDO Range'] = '16-30'
df_booking_curve.loc[df_booking_curve['NDO'].between(31, 60, 'both'), 'NDO Range'] = '31-60'
df_booking_curve.loc[df_booking_curve['NDO'].between(61, 90, 'both'), 'NDO Range'] = '61-90'
df_booking_curve.loc[df_booking_curve['NDO'].between(91, 120, 'both'), 'NDO Range'] = '91-120'
df_booking_curve.loc[df_booking_curve['NDO'].between(120, 365, 'both'), 'NDO Range'] = '>120'

df.loc[df['NDO'].between(0, 7, 'both'), 'NDO Range'] = '0-7'
df.loc[df['NDO'].between(8, 15, 'both'), 'NDO Range'] = '8-15'
df.loc[df['NDO'].between(16, 30, 'both'), 'NDO Range'] = '16-30'
df.loc[df['NDO'].between(31, 60, 'both'), 'NDO Range'] = '31-60'
df.loc[df['NDO'].between(61, 90, 'both'), 'NDO Range'] = '61-90'
df.loc[df['NDO'].between(91, 120, 'both'), 'NDO Range'] = '91-120'
df.loc[df['NDO'].between(120, 365, 'both'), 'NDO Range'] = '>120'

# Date
# Format checkin Time
# df["Departure Time"] = df["Departure Time"].apply(
#     lambda x: dt.strptime(x, "%Y-%m-%d %I:%M:%S %p")
# )  # String -> Datetime

# Insert weekday and hour of checkin time
df['Departure Time'] = pd.to_datetime(df['Departure Time'])
df["Days of Wk"] = df["Check-In Hour"] = df["Departure Time"]
df["Days of Wk"] = df["Days of Wk"].apply(
    lambda x: dt.strftime(x, "%A")
)  # Datetime -> weekday string

df["Check-In Hour"] = df["Check-In Hour"].apply(
    lambda x: dt.strftime(x, "%I %p")
)  # Datetime -> int(hour) + AM/PM

day_list = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

all_markets = df['Route'].unique().tolist()
all_products = df['Product'].unique().tolist()
all_journeys = df['Journey Type'].unique().tolist()


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Forward Booked Load Insights"),
            html.H3("Welcome to the Forward Booked Load Insights Dashboard"),
            html.Div(
                id="intro",
                children="Explore insights related to forward booked load by Route, Days-to-departure (DTD), Days-of-Week and Time-of-days. Compare performance with the reference historical date range."

            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P(),
            html.P("Select Departure Range"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=min(df['Departure Time']).date(),
                end_date=dt(2023, 9, 10),
                min_date_allowed=min(df['Departure Time']).date(),
                max_date_allowed=max(df['Departure Time']).date(),
                initial_visible_month=dt(2023, 3, 1),
                className="dcc_control",
            ),
            html.Br(),
            html.P(),
            html.P("Select Desired Forecast Range"),
            dcc.RangeSlider(
                id="forecast-range-slider",
                min=20,
                max=120,
                step=10,
                value=[50, 100],
                allowCross=False,
                className="dcc_control",
            ),
            html.Br(),
            html.P("Select Reference Date Range"),
            dcc.DatePickerRange(
                id="date-ref-select",
                start_date=dt(2022, 1, 1),
                end_date=dt(2022, 3, 31),
                min_date_allowed=dt(2021, 1, 1),
                max_date_allowed=dt(2023, 3, 16),
                initial_visible_month=dt(2022, 3, 1),
                className="dcc_control",
            ),


            # html.Br(),
            # html.P(),
            # html.P("Select Departure Type"),
            # dcc.Dropdown(
            #     id="journey-select",
            #     options=[
            #         {'label': 'Departures Within Desired Forecast Range', 'value': 'within_range'},
            #         {'label': 'Over-booked Departures', 'value': 'over-booked'},
            #         {'label': 'Under-booked Departures', 'value': 'under-booked'}
            #     ],
            #     value='within_range',
            #     className="dcc_control",
            # ),
            html.Br(),
            html.P(),
            html.P("Select Route"),
            dcc.Dropdown(
                id="market-select",
                options=[{"label": i, "value": i} for i in all_markets],
                value=all_markets[:],
                multi=True,
                className="dcc_control",
            ),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
                style={"display": "none"}
            ),
        ],
    )


def generate_filtered_dataframe(start, end, market):
    filtered_df_init = df[(df["Route"].isin(market)) &
                          (df['Departure Time'] >= start) &
                          (df['Departure Time'] <= end)]
    return filtered_df_init


def generate_departure_volume_heatmap(start, end, forecast_range, hm_click, market, clickData, clickndodata, reset):
    """
    :param: start: start date from departure date selection.
    :param: end: end date from departure date selection.
    :param: journey: journey type from selection.
    :param: hm_click: clickData from heatmap.
    :param: market: market name from selection.
    :param: reset (boolean): reset heatmap graph if True.

    :return: Oversell/Undersell annotated heatmap.
    """
    print(start)
    print(end)
    journey = ""
    print(f'value of ndo data is {clickndodata}')

    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        filtered_df_init = df[(df["Route"] == market) &
                              (df['Departure Time'] >= start) &
                              (df['Departure Time'] <= end)]
    else:
        filtered_df_init = generate_filtered_dataframe(start, end, market)

    if clickndodata is not None:
        print('Are we entering in this loop ?')
        journey = clickndodata['points'][0]['customdata'][0]
        dtd_range = clickndodata['points'][0]['x']
        print(f'value of journey is {journey}')
        print(f'value of dtd range is {dtd_range}')
        filtered_df_init = filtered_df_init[filtered_df_init['NDO Range'] == dtd_range]
        print(f'value of filtered_df_init after NDO click is {filtered_df_init["NDO Range"]}')

    if journey == "On-track":
        filtered_df = filtered_df_init[
            (filtered_df_init['Forecast'] >= forecast_range[0]) &
            (filtered_df_init['Forecast'] <= forecast_range[1])
            ]
        print(f'value of journey is {journey}')
    elif journey == "High Forecast":
        filtered_df = filtered_df_init[(filtered_df_init['Forecast'] >= forecast_range[1])]
        print(f'value of journey is {journey}')
        print(filtered_df)
    elif journey == "Low Forecast":
        filtered_df = filtered_df_init[(filtered_df_init['Forecast'] <= forecast_range[0])]
        print(f'value of journey is {journey}')
        print(filtered_df)
    else:
        print(f'No value for journey')
        filtered_df = filtered_df_init.copy()

    filtered_df = filtered_df.sort_values("Departure Time").set_index("Departure Time")
    print(f'Shape of filtered_df is {filtered_df.shape}')

    x_axis = [datetime.time(i).strftime("%I %p") for i in range(24)]  # 24hr time list
    y_axis = day_list

    hour_of_day = ""
    weekday = ""
    shapes = []

    if hm_click is not None:
        print(hm_click)
        hour_of_day = hm_click["points"][0]["x"]
        weekday = hm_click["points"][0]["y"]

        # Add shapes
        x0 = x_axis.index(hour_of_day) / 24
        x1 = x0 + 1 / 24
        y0 = y_axis.index(weekday) / 7
        y1 = y0 + 1 / 7

        shapes = [
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#ff6347"),
            )
        ]

    # Get z value : sum(number of records) based on x, y,

    z = np.zeros((7, 24))
    annotations = []

    for ind_y, day in enumerate(y_axis):
        filtered_day = filtered_df[filtered_df["Days of Wk"] == day]
        for ind_x, x_val in enumerate(x_axis):
            sum_of_record = filtered_day[filtered_day["Check-In Hour"] == x_val][
                "Departures"
            ].sum()
            # print(f' value of sum of records is {sum_of_record}')
            z[ind_y][ind_x] = sum_of_record

            annotation_dict = dict(
                showarrow=False,
                text="<b>" + str(sum_of_record) + "<b>",
                xref="x",
                yref="y",
                x=x_val,
                y=day,
                font=dict(family="sans-serif"),
            )
            # Highlight annotation text by self-click
            if x_val == hour_of_day and day == weekday:
                if not reset:
                    annotation_dict.update(size=15, font=dict(color="#ff6347"))

            annotations.append(annotation_dict)

    # Heatmap
    hovertemplate = "<b> %{y}  %{x} <br><br> %{z} Departures"

    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#caf3ff"], [1, "#97e3ff"], [2, "#579dca"]],
        )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            # children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        html.Div(
            [
                html.Div(
                    id="left-column",
                    className="pretty_container four columns",
                    children=[description_card(), generate_control_card()]
                ),
                # Right column
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H5(id="departureText"), html.P("Total Departures")],
                                    id="departures",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H5(id="overBookedText"), html.P("Yield Critical Departures")],
                                    id="overbooked_departures",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H5(id="underBookedText"), html.P("Load Critical Departures")],
                                    id="underbooked_departures",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H5(id="obDTDText"), html.P("Yield Critical DTD Range")],
                                    id="obDTD",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H5(id="ubDTDText"), html.P("Load Critical DTD Range")],
                                    id="ubDTD",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H5(id="MarketText"), html.P("Critical Route")],
                                    id="obMarket",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        # Total Departure Heatmap
                        html.Div(
                            id="departure_volume_card",
                            children=
                            [
                                html.B("Revenue Opportunities (Yield/Load Critical) by Route and Departures"),
                                dcc.Graph(id="mkt_booking_trend")
                            ],
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.B("Critical Departures Distribution by Days-to-Departure"),
                        dcc.Graph(id="dtd_booking_trend")
                    ],
                    className="pretty_container five columns"
                ),
                html.Div(
                    [
                        html.B("Critical Departures Distribution by Days/Time-of-Week"),
                        # html.Hr(),
                        dcc.Graph(id="departure_volume_hm"),
                    ],
                    className="pretty_container seven columns"
                )
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    id='output-data-upload',
                    children=[],
                    className="pretty_container twelve columns")
            ],
            className="row flex-display"
        ),
        html.Br(),
        html.P(
            '-- Use Booking Curve to compare the current performance with the historical/reference date range defined in the selection Panel'),

        html.Div(
            [
                html.Div([
                    html.Label('Select a Dimension :', style={'font-weight': 'bold'}),
                    dcc.RadioItems(id='data-radio-dimension',
                                   options={
                                       'load': 'Booking Comparison',
                                       'revenue': 'Revenue Comparison',
                                       'fare': 'Fare Comparison'
                                   },
                                   value='load',
                                   inputStyle={'margin-right': '10px'},
                                   labelStyle={'display': 'inline-block', 'padding': '0.5rem 0.5rem'}
                                   )
                ], className="twelve columns"
                ),
            ], className="row flex-display"
        ),

        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='fig-booking-curve')
                    ],
                    className="twelve columns",
                ),
            ],
            className="row flex-display"
        ),


        html.Div(
            [
                html.Div(
                    [
                        html.B("Product Performance(Current vs Reference Period)"),
                        dcc.Graph(id="prod_trend")
                    ],
                    className="pretty_container six columns"
                ),
                html.Div(
                    [
                        html.B("Journey Performance(Current vs Reference Period)"),
                        dcc.Graph(id="journey_trend")
                    ],
                    className="pretty_container six columns"
                ),
            ],
            className="row flex-display",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.B("Channel Performance(Current vs Reference Period)"),
                        dcc.Graph(id="channel_trend")
                    ],
                    className="pretty_container six columns"
                ),
                html.Div(
                    [
                        html.B("Inventory Performance(Current vs Reference Period)"),
                        dcc.Graph(id="inventory_trend")
                    ],
                    className="pretty_container six columns"
                ),
            ],
            className="row flex-display",
        ),
    ]
)


@app.callback(
    [
        Output("departureText", "children"),
        Output("overBookedText", "children"),
        Output("underBookedText", "children"),
        Output("obDTDText", "children"),
        Output("ubDTDText", "children"),
        Output("MarketText", "children"),
    ],
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("forecast-range-slider", "value"),
        Input("market-select", "value"),
    ]
)
def update_departure_summary(start, end, forecast_range, market):
    ob_ub_list = list()
    filtered_df_init = df[(df['Departure Time'] >= start) &
                          (df['Departure Time'] <= end) &
                          (df["Route"].isin(market))]

    # filtered_df_reg = filtered_df_init[(filtered_df_init['Forecast'] >= forecast_range[0]) &
    #                                    (filtered_df_init['Forecast'] <= forecast_range[1])]
    filtered_df_ob = filtered_df_init[filtered_df_init['Forecast'] >= forecast_range[1]]
    filtered_df_ub = filtered_df_init[filtered_df_init['Forecast'] <= forecast_range[0]]
    df_imp_mkt = filtered_df_init[(filtered_df_init['Forecast'] <= forecast_range[0]) |
                                  (filtered_df_init['Forecast'] >= forecast_range[1])]

    total_departures = filtered_df_init['Departures'].sum()
    print(total_departures)

    ob_departures = filtered_df_ob['Departures'].sum()
    print(ob_departures)
    ob_departures_perc = round((ob_departures / total_departures) * 100)
    print(ob_departures_perc)
    ob_df = filtered_df_ob.groupby('NDO Range')['Departures'].sum().sort_values(ascending=False).reset_index()
    ob_ndo_range = ob_df['NDO Range'][0]
    print(ob_ndo_range)
    ob_ndo_perc = round((ob_df['Departures'][0] / ob_departures) * 100)
    print(ob_ndo_perc)


    ub_departures = filtered_df_ub['Departures'].sum()
    print(ub_departures)
    ub_departures_perc = round((ub_departures / total_departures) * 100)
    print(ub_departures_perc)
    ub_df = filtered_df_ub.groupby('NDO Range')['Departures'].sum().sort_values(ascending=False).reset_index()
    ub_ndo_range = ub_df['NDO Range'][0]
    print(ub_ndo_range)
    ub_ndo_perc = round((ub_df['Departures'][0] / ub_departures) * 100)
    print(ub_ndo_perc)

    df_imp_mkt_init = df_imp_mkt.groupby('Route')['Departures'].sum().sort_values(ascending=False).reset_index()
    imp_mkt = df_imp_mkt_init['Route'][0]
    imp_mkt_perc = round(df_imp_mkt_init['Departures'][0] / total_departures * 100)

    ob_dep_detail = str(ob_departures) + " " + str("(") + str(ob_departures_perc) + str("%)")
    ob_ub_list.append(ob_dep_detail)
    ub_dep_detail = str(ub_departures) + " " + str("(") + str(ub_departures_perc) + str("%)")
    ob_ub_list.append(ub_dep_detail)
    ob_ndo_detail = str(ob_ndo_range) + " " + str("(") + str(ob_ndo_perc) + str("%)")
    ob_ub_list.append(ob_ndo_detail)
    ub_ndo_detail = str(ub_ndo_range) + " " + str("(") + str(ub_ndo_perc) + str("%)")
    ob_ub_list.append(ub_ndo_detail)

    imp_mkt_detail = str(imp_mkt) + str("(") + str(imp_mkt_perc) + str("%)")
    ob_ub_list.append(imp_mkt_detail)

    return str(total_departures), ob_ub_list[0], ob_ub_list[1], ob_ub_list[2], \
        ob_ub_list[3], ob_ub_list[4]


@app.callback(
    Output("dtd_booking_trend", "figure"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("forecast-range-slider", "value"),
        Input("market-select", "value"),
        Input("mkt_booking_trend", "clickData")
    ]
)
def update_dtd_booking_trend(start, end, forecast_range, market, clickData):
    print(f"Checking the value of click data from Market graph {clickData}")

    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        filtered_df_init = df[(df["Route"] == market) &
                              (df['Departure Time'] >= start) &
                              (df['Departure Time'] <= end)]
    else:
        filtered_df_init = generate_filtered_dataframe(start, end, market)

    filtered_df_init['Departure_Status'] = filtered_df_init['Forecast'].apply(
        lambda x: 'Yield Critical Departures' if x > forecast_range[1] else (
            'Load Critical Departures' if x < forecast_range[0] else 'On-Track Departures'))

    filtered_df_1 = filtered_df_init.groupby(['NDO Range', 'Departure_Status'])['Departures'].sum().reset_index()
    filtered_df_2 = filtered_df_1.groupby('NDO Range')['Departures'].sum().reset_index()
    df_res = pd.merge(filtered_df_1, filtered_df_2, left_on='NDO Range', right_on='NDO Range')
    perc_dep_ndo = df_res['Departures_x'] / df_res['Departures_y']
    df_res_1 = pd.concat([df_res, perc_dep_ndo], axis=1).rename(columns={0: "Departures(%)"})
    df_res_1['Departures(%)'] = round(df_res_1['Departures(%)'] * 100, 1)
    df_res_1.rename(columns={'Departure_Status': 'Departure Type', 'NDO Range':'Days-to-Departure Range'}, inplace=True)

    fig = px.bar(df_res_1, x='Days-to-Departure Range', y='Departures(%)', color='Departure Type', barmode='stack', text='Departures(%)', custom_data=['Departure Type'], template='ggplot2')
    return fig


@app.callback(
    Output("mkt_booking_trend", "figure"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("forecast-range-slider", "value"),
        Input("market-select", "value")
    ]
)
def update_mkt_booking_trend(start, end, forecast_range, market):
    filtered_df_init = generate_filtered_dataframe(start, end, market)

    filtered_df_init['Departure_Status'] = filtered_df_init['Forecast'].apply(
        lambda x: 'High Forecast' if x > forecast_range[1] else (
            'Low Forecast' if x < forecast_range[0] else 'On-track'))

    filtered_df_init = pd.concat([filtered_df_init,
                                  pd.get_dummies(filtered_df_init['Departure_Status'])], axis=1)

    df_1 = filtered_df_init.groupby('Route').agg(
        {'High Forecast': 'sum', 'Low Forecast': sum, 'On-track': sum, 'Departures': 'sum'})

    df_1.reset_index(inplace=True)
    df_1['Yield Critical Departures (%)'] = round(df_1['High Forecast'] / df_1['Departures'] * 100, 1)
    df_1['Load Critical Departures (%)'] = round(df_1['Low Forecast'] / df_1['Departures'] * 100, 1)
    df_1['On-Track Departures (%)'] = round(df_1['On-track'] / df_1['Departures'] * 100, 1)

    fig = px.scatter(df_1, y="Yield Critical Departures (%)", x="Load Critical Departures (%)", size="Departures", color="Route", custom_data=["Route"], template='ggplot2')
    return fig


@app.callback(
    Output('output-data-upload', 'children'),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("forecast-range-slider", "value"),
        Input("market-select", "value"),
        Input("mkt_booking_trend", "clickData"),
        Input("dtd_booking_trend", "clickData"),
        Input('output-data-upload', 'children')
    ]
)
def update_flight_data_output(start, end, forecast_range, market, clickData, clickndodata, children):
    journey = ""
    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        filtered_df_init = df[(df["Route"] == market) &
                              (df['Departure Time'] >= start) &
                              (df['Departure Time'] <= end)]
    else:
        filtered_df_init = generate_filtered_dataframe(start, end, market)

    if clickndodata is not None:
        journey == clickndodata['points'][0]['customdata'][0]

    if journey == "On-track":
        filtered_df = filtered_df_init[
            (filtered_df_init['Forecast'] >= forecast_range[0]) &
            (filtered_df_init['Forecast'] <= forecast_range[1])
            ]
        print(f'value of journey is {journey}')
    elif journey == "High Forecast":
        filtered_df = filtered_df_init[(filtered_df_init['Forecast'] >= forecast_range[1])]
        print(f'value of journey is {journey}')
        print(filtered_df)
    elif journey == "Low Forecast":
        filtered_df = filtered_df_init[(filtered_df_init['Forecast'] <= forecast_range[0])]
        print(f'value of journey is {journey}')
        print(filtered_df)
    else:
        filtered_df = filtered_df_init.copy()

    filtered_df['Forecast'] = round(filtered_df['Forecast'], 0)

    filtered_df.rename(columns={'NDO': 'Days-to-Departure'}, inplace=True)
    children = (html.Div([
        html.B("Departure Details"),

        dash_table.DataTable(
            filtered_df[:100].to_dict('records'),
            [{'name': i, 'id': i, 'selectable': True} for i in columns],
            filter_action='native',
            editable=True,
            sort_action='native',
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            page_current=0,
            page_size=20,
            style_cell={  # ensure adequate header width when text is shorter than cell's text
                'minWidth': 95, 'maxWidth': 95, 'width': 95, 'textAlign': 'center', 'padding': '5px',
                'font_family': 'sans-serif', 'backgroundColor': 'rgba(0,0,0,0)'
            },
            style_table={'height': '500px', 'overflowY': 'auto'},
            style_data={  # overflow cells' content into multiple lines
                'whiteSpace': 'normal',
                'height': 'auto',
                'template': 'ggplot2',
                'font_family': 'sans-serif'
            },
            style_header={
                'template': 'ggplot2',
                'fontWeight': 'bold',
                'font_family': 'sans-serif'
            },
            export_format='csv',
            style_as_list_view=True,
        )
    ]))

    return children


@app.callback(
    Output("departure_volume_hm", "figure"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("forecast-range-slider", "value"),
        # Input("journey-select", "value"),
        Input("departure_volume_hm", "clickData"),
        Input("market-select", "value"),
        Input("mkt_booking_trend", "clickData"),
        Input("dtd_booking_trend", "clickData"),
        # Input("reset-btn", "n_clicks"),
    ],
)
def update_heatmap(start, end, forecast_range, hm_click, market, clickData, clickndodata):
    start = start + " 00:00:00"
    end = end + " 00:00:00"

    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context
    print(f' Check the value of ctx as {ctx}')
    print(f'Value extracted from clicked data for NDO is {clickndodata}')

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return generate_departure_volume_heatmap(
        start, end, forecast_range, hm_click, market, clickData, clickndodata, reset
    )


@app.callback(
    Output('fig-booking-curve', 'figure'),
    Input("mkt_booking_trend", "clickData"),
    Input('data-radio-dimension', 'value')
)
def updated_booking_curve(clickData, dimension):
    y1 = ""
    y2 = ""
    name1 = ""
    name2 = ""

    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        df_curve = df_booking_curve[df_booking_curve['Route'] == market]
    else:
        df_curve = df_booking_curve.copy()

    if dimension == 'load':
        y1 = 'Booked Historical'
        y2 = 'Current'
        name1 = 'Flown Load'
        name2 = 'Booked Load'
    elif dimension == 'revenue':
        y1 = 'Total Rev (Historical)'
        y2 = 'Current Revenue'
        name1 = 'Historical Revenue'
        name2 = 'Current Revenue'
    else:
        y1 = 'Avg Fare (Historical)'
        y2 = 'Current Fare'
        name1 = 'Historical Avg. Fare'
        name2 = 'Current Avg. Fare'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_curve['NDO'], y=df_curve[y1],
                             mode='lines', name=name1))
    fig.add_trace(go.Scatter(x=df_curve['NDO'], y=round(df_curve[y2]),
                             mode='lines', name=name2))
    fig.update_layout(title={'text': 'Booking Curve by NDO', 'font': {'size': 20}},
                      title_font_family="Arial",
                      title_font_color="black",
                      xaxis_title='Days-to-Departure',
                      template='ggplot2')

    return fig


@app.callback(
    Output("prod_trend", "figure"),
    Input("mkt_booking_trend", "clickData")
)
def product_performance(clickData):
    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        df_product_init = df_product[df_product['Route'] == market]
        df_product_final = df_product_init.groupby(['Product', 'NDO', 'Period'])['Load Factor'].sum().reset_index()
    else:
        df_product_final = df_product.groupby(['Product', 'NDO', 'Period'])['Load Factor'].sum().reset_index()

    figure = px.area(df_product_final, x='NDO', y='Load Factor', color='Product', template='ggplot2', facet_row='Period', category_orders={'Period': ['Historical', 'Current']})
    figure.update_layout(legend=dict(
        orientation="h",
        # entrywidth=90,
        font=dict(size=12),
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.8))
    return figure


@app.callback(
    Output("journey_trend", "figure"),
    Input("mkt_booking_trend", "clickData")
)
def journey_performance(clickData):
    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        df_journey_init = df_journey[df_journey['Route'] == market]
        df_journey_final = df_journey_init.groupby(['Journey', 'NDO', 'Period'])['Load Factor'].sum().reset_index()
    else:
        df_journey_final = df_journey.groupby(['Journey', 'NDO', 'Period'])['Load Factor'].sum().reset_index()

    figure = px.area(df_journey_final, x='NDO', y='Load Factor', color='Journey', template='ggplot2', facet_row='Period', category_orders={'Period': ['Historical', 'Current']})
    figure.update_layout(legend=dict(
        orientation="h",
        # entrywidth=90,
        font=dict(size=12),
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.55))
    return figure


@app.callback(
    Output("channel_trend", "figure"),
    Input("mkt_booking_trend", "clickData")
)
def channel_performance(clickData):
    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        df_channel_init = df_channel[df_channel['Route'] == market]
        df_channel_final = df_channel_init.groupby(['Channel', 'NDO', 'Period'])['Load Factor'].sum().reset_index()
    else:
        df_channel_final = df_channel.groupby(['Channel', 'NDO', 'Period'])['Load Factor'].sum().reset_index()

    print(f'Shape of df_channel_final is {df_channel_final.shape}')
    figure = px.area(df_channel_final, x='NDO', y='Load Factor', color='Channel', template='ggplot2',
                     facet_row='Period', category_orders={'Period': ['Historical', 'Current']})
    figure.update_layout(legend=dict(
        orientation="h",
        # entrywidth=90,
        font=dict(size=12),
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.1))
    return figure


@app.callback(
    Output("inventory_trend", "figure"),
    Input("mkt_booking_trend", "clickData")
)
def journey_performance(clickData):
    print(f'check the value of clickData {clickData}')
    if clickData is not None:
        market = clickData['points'][0]['customdata'][0]
        df_inventory_init = df_inventory[df_inventory['Route'] == market]
        df_inventory_final = df_inventory_init.groupby(['Inventory Class', 'NDO', 'Period'])[
            'Load Factor'].sum().reset_index()
    else:
        df_inventory_final = df_inventory.groupby(['Inventory Class', 'NDO', 'Period'])[
            'Load Factor'].sum().reset_index()

    print(f'Shape of df_product_final is {df_inventory_final.shape}')
    figure = px.area(df_inventory_final, x='NDO', y='Load Factor', color='Inventory Class', template='ggplot2',
                     facet_row='Period',
                     category_orders={'Period': ['Historical', 'Current'],
                                      'Inventory Class': ['R', 'Q', 'P', 'L', 'J', 'T', 'V', 'Y']})
    figure.update_layout(legend=dict(
        orientation="h",
        # entrywidth=90,
        font=dict(size=12),
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.92))
    return figure


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
