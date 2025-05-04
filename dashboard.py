# dashboard.py
import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output

# Load data
df = pd.read_csv('data/output.csv')

def create_dashboard(flask_app):
    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname='/dashapp/',
    )

    dash_app.layout = html.Div([
        html.H2("Healthcare Dashboard"),

        html.Label("Filter by Treatment Group:"),
        dcc.Dropdown(
            id='treatment-dropdown',
            options=[{'label': str(i), 'value': i} for i in sorted(df['Treatment_encoded'].dropna().unique())],
            value=df['Treatment_encoded'].dropna().unique()[0]
        ),

        dcc.Graph(id='line-chart'),
        dcc.Graph(id='bar-chart'),
        dcc.Graph(id='pie-chart')
    ])

    @dash_app.callback(
        Output('line-chart', 'figure'),
        Output('bar-chart', 'figure'),
        Output('pie-chart', 'figure'),
        Input('treatment-dropdown', 'value')
    )
    def update_charts(treatment):
        filtered = df[df['Treatment_encoded'] == treatment]

        line_fig = px.line(
            filtered.reset_index(),
            x='index',
            y='Ill_days_total',
            title='Ill Days Over Observations'
        )

        bar_fig = px.bar(
            filtered.groupby('GenderFA')['Ill_days_total'].mean().reset_index(),
            x='GenderFA',
            y='Ill_days_total',
            title='Avg Illness Days by Gender'
        )

        pie_fig = px.pie(
            filtered,
            names='any_healthcare_visit',
            title='Proportion of Healthcare Visits'
        )

        return line_fig, bar_fig, pie_fig

    return dash_app

