import numpy as np 

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from ipywidgets import interact

# init_notebook_mode(connected=True)

import ipywidgets as widgets


dt = 0.01
x = np.arange(0, 10, 0.01)
y = np.sin(x)

def update_circle(t):
    data = [go.Scatter(
        x=[t],
        y=[np.sin(t)],
    ),
    go.Scatter(
        x=x,
        y=y,
    )]
    iplot(data, show_link=False)


def update_plot(x):
    data = [go.Bar(
                x=['1', '2'],
                y=[x, 1-x]
    )]
    iplot(data, show_link=False)

t = widgets.FloatSlider(min=0, max=10, value=0., step=dt)
widgets.interactive(update_circle, t=t)