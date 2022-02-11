#!/usr/bin/env python
# coding: utf-8

# In[5]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction, State
from dash import dcc
import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import dash_table
import pathlib
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import plotly.graph_objects as go
import warnings
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import timedelta
import datetime
from os import path
import pandas as pd
import numpy as np
import scipy
import sys
from PIL import Image
import os
import base64
import io
import json
import sys


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

application = app.server
app.title = "Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True


# In[6]:


from sou.functions_check import horizon_list


# In[7]:


data_analysis=pd.read_excel("data_analysis.xlsx")
download_data=pd.read_excel("download_data.xlsx")
filtered_df2=pd.read_excel("filtered_df2.xlsx")
filtered_df1=pd.read_excel("filtered_df1.xlsx")
filtered_df=pd.read_excel("filtered_df.xlsx")


# In[8]:


df = pd.DataFrame(download_data)
plant_savings=209580
currency_value="INR"


# In[9]:


bar_data=data_analysis.iloc[1:,]
plant_data=data_analysis.iloc[:1,]
filtered_df=filtered_df[filtered_df==0.0000].replace(np.nan,1)


# In[10]:


import dash_bootstrap_components as dbc
cards = [
    dbc.Card(
        [
            html.H4(f"{plant_savings:.2f} {currency_value}", className="card-title"),
            html.P("Total Savings", className="card-text"),
        ],
        body=True,
        color="light",
    ),
]


# In[11]:


def upload_data_card():
    """
    :return: A Div containing an Upload button
    """
    return html.Div(
        id='upload-data-card',
        children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([html.Button('Upload data')]),
                multiple=False
            ),
        ]
    )


# In[12]:


tab_style = {
    'borderTop': '2px solid #ffaf2a',
}


# In[13]:


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("SH"),
            html.H3("Welcome to the Dashboard"),
            html.Div(
                id="intro",
                children="Explore the most optimal calendar",
            ),
        ],
    )


# In[14]:


input_list = ['INR','EUR','USD','CHF']
Input_type = ["Cost/plant/event","Cost/MWh/event","Cost/kWh/event"]

def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Upload Data"),
            html.Div(
                id="banner-logo",
                children=[
                    upload_data_card()]),
            html.Br(),
            html.P("file"),
            dcc.Dropdown(
                id="clinic-select",
                options=[{"label": i, "value": i} for i in input_list],
                value=input_list[0],
                persistence=True,
                persisted_props=['value'],
                persistence_type='session',
            ),  
            html.Br(),
            html.Div(
                        [
                            #html.I("Try typing in input 1 & 2, and observe how debounce is impacting the callbacks. Press Enter and/or Tab key in Input 2 to cancel the delay"),
                            html.P("cost"),
                            dcc.Input(id="Electricity cost", type="number", placeholder="", style={'marginRight':'10px'}),
                            html.Br(),
                            html.P("kk"),
                            dcc.Input(id="Currency", type="number", placeholder="",),
                            html.Div(id="output"),
                        ]
                    ),
            html.Br(),
            html.P("Time Horizon"),
            dcc.Dropdown(
                id="admit-select",
                #options=[{"label": i, "value": i} for i in horizon_list],
                options = horizon_list,
                value=horizon_list[-1],
                multi=False,
                persistence=True,
                persisted_props=['value'],
                persistence_type='session',
            ),
            html.Br(),
            dbc.Button("Submit", color="primary"),
            html.Br(),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
        ],
    )


# In[15]:


def generate_modal():
    return html.Div(
        id="markdown",
        className="modal",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                        ###### What is this Dashboard about?

                        Explore the most optimal cleaning dates for your site. Click on the 'Dynamic Schedule' to visualize detailed calendar

                        ###### Significance




                    """
                            )
                        ),
                    ),
                ],
            )
        ),
    )


# In[16]:


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.A(
                html.Img(src=app.get_asset_url("SH_logo.png")),
                      href="https://smarthelio.com/",
                        ),
                     html.A(
                         html.Button(
                        id="learn-more-button", children="LEARN MORE", n_clicks=0
                    ), 
                      ) 
                     
                     ],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
                dcc.Tabs(
                    id="stitching-tabs",
                    value="canvas-tab",
                    style={
                            'height': '5%'
                        },
                    children=[
                        dcc.Tab(
                            label="Dynamic Schedule", value="schedule-tab",selected_style=tab_style,
                            children=[html.Div(
                                    id="overview_volume_card",
                                    children=[
                                    html.B("Calender"),
                                    html.Hr(),
                                    dcc.Graph(id="patient_volume_hm"),
                                    ],
                                    ),
                                    html.Hr(), 
                                    html.Div(
                                        [
                                            html.Button("Download Calendar", id="btn_csv"),
                                            dcc.Download(id="download-dataframe-csv"),
                                        ]
                                    ),
                                     ],
                        ),
                        dcc.Tab(
                            label="Impact of Cleaning", value="result-tab", selected_style=tab_style,
                            children=[
                                   html.Hr(),
                                   dbc.Row([dbc.Col(card) for card in cards]),
                                   html.Div(
                                   id="patient_volume_card",
                                   children=[
                                   html.B("Cost Analysis (Plant level)"),
                                   html.Hr(),
                                   dcc.Graph(id="plant-bar"),   
                                   ],
                                   ),
                                   html.Div(
                                   id="wait_time_card",
                                   children=[
                                   html.B("Inverter/MPPT level analysis"),
                                   html.Hr(),
                                   dcc.Graph(id="count-bar"),   
                                   ],
                                   ),                                
                                ]),
                    ],
                    className="tabs",
                ),
                html.Div(
                    id="tabs-content-example",
                    className="canvas",
                    style={"text-align": "left", "margin": "auto"},
                ),
                html.Div(
                dcc.Store(id='plot-data', storage_type='session')
                ),
        generate_modal(),
      dcc.Store(id='intermediate-value', storage_type='session')  
    ],
)


# In[17]:


# HeatMap for the Schedule

def generate_patient_volume_heatmap(hm_click, admit_type, reset):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    :param: clinic: clinic from selection.
    :param: hm_click: clickData from heatmap.
    :param: admit_type: admission type from selection.
    :param: reset (boolean): reset heatmap graph if True.

    :return: Patient volume annotated heatmap.
    """
    y_axis = filtered_df.index
    x_axis = filtered_df.columns
    annotations = []
    

    hour_of_day = ""
    weekday = ""
    shapes = []

    if hm_click is not None:
        hour_of_day = hm_click["points"][0]["x"]
        weekday = hm_click["points"][0]["y"]

        # Add shapes
        x0 = x_axis.index(hour_of_day) / len(x_axis)
        x1 = x0 + 1 / len(x_axis)
        y0 = y_axis.index(weekday) / len(y_axis)
        y1 = y0 + 1 / len(y_axis)

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

    z = np.zeros((len(y_axis), len(x_axis)))
    annotations = []

    for ind_y, day in enumerate(y_axis):
        filtered_day = filtered_df[filtered_df.index == day]
        for ind_x, x_val in enumerate(x_axis):
            sum_of_record = filtered_day[x_val].sum()
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
    hovertemplate = "<b> %{y}  %{x} <br><br> <br>Cumulative Soiling Loss (DCS): %{customdata[0]:.1f} in kWh <br>Cumulative Soiling Loss (No DCS) :%{customdata[1]:.3f} in kWh"

    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap", 
            name="",
            customdata = np.dstack((filtered_df1, filtered_df2)),
            hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#ff6d00"], [1, "#fde295"]], #6d0101 #ebff8b
            zmin=0,
            zmax=1,
            linecolor="black",
            #showgrid = True,
            edgecolor='blue',
        )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "h"},
        font=dict(family="Open Sans"),
        shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=1,
            tickfont=dict(family="sans-serif"),
            #tickmode = 'linear',
            tickcolor="#ffffff",
            #showline=True,
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" ", tickmode = 'linear',
        ),
        colorbar = dict(colorbar_orientation="h"),
        hovermode="closest",
        #showlegend=True,
        #showgrid=True,
    )
    return {"data": data, "layout": layout}


@app.callback(
    Output("patient_volume_hm", "figure"),
    [
        Input("patient_volume_hm", "clickData"),
        Input("admit-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],#prevent_initial_call=True,
)
def update_heatmap(hm_click, admit_type, reset_click):
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True
    return generate_patient_volume_heatmap(hm_click, admit_type, reset)


# In[18]:


@app.callback(
    Output("count-bar", "figure"),
    [Input("admit-select", "value")],
    #prevent_initial_call=True,
)
def update_figure1(horizon):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        name='With Cleaning',
        x=bar_data["Plant/Inverter"],
        y=bar_data['Energy_wl'],
        marker_color= "saddlebrown", 
        opacity=0.6,
    ), secondary_y=False
    )

    fig.add_trace(go.Bar(
        name='No cleaning',
        x=bar_data["Plant/Inverter"],
        y=bar_data['Energy_nl'],
        marker_color= "#005A9C", #""tan",
        opacity=0.6,
    ), secondary_y=False
    )

    fig.update_layout(barmode='stack')

    fig.update_yaxes(title_text="Values", secondary_y=False)
    fig.update_layout(
        hovermode="x",
        barmode="group",
        #title="Cleaning Events",
        font={"color": "darkslategray"},
        paper_bgcolor="white",
        plot_bgcolor= "#f8f5f0",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        yaxis=dict(tick0=1),
        xaxis=dict(tickangle=290)
    )
    return fig


# In[19]:


@app.callback(
    Output("plant-bar", "figure"),
    [Input("admit-select", "value")],
    #prevent_initial_call=True,
)
def update_figure1(horizon):
    fig = make_subplots()
    fig.add_trace(go.Bar(
        name='With Cleaning',
        x=plant_data["Plant/Inverter"],
        y=plant_data['Energy_wl']*energy_cost,
        marker_color= "saddlebrown", 
        opacity=0.6,
    ), 
    )

    fig.add_trace(go.Bar(
        name='No cleaning',
        x=plant_data["Plant/Inverter"],
        y=plant_data['Energy_nl']*energy_cost,
        marker_color= "#005A9C", #""tan",
        opacity=0.6,
    ), 
    )
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(
        hovermode="x",
        barmode="group",
        #title="Cleaning Events",
        font={"color": "darkslategray"},
        paper_bgcolor="white",
        plot_bgcolor= "#f8f5f0",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        yaxis=dict(tick0=1),
        xaxis=dict(tickangle=290)
    )
    return fig


# In[20]:


# ======= Callbacks for modal popup =======
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "learn-more-button":
            return {"display": "block"}

    return {"display": "none"}


# In[21]:



@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "cleaning_schedule.csv")


# In[ ]:


# Run the server
if __name__ == "__main__":
    application.run(debug=False, port=8080)

