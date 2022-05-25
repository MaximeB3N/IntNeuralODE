import numpy as np

import matplotlib.pyplot as plt

from bokeh.layouts import layout
from bokeh.models import Div, RangeSlider, Spinner, Slider
from bokeh.plotting import figure, show

dt = 0.01
x = np.arange(0, 10, 0.01)
y = np.sin(x)
# plt.plot(x, y)
# plt.show()

# create plot with circle glyphs
p = figure(x_range=(0, 10), width=1000, height=500)
points = p.circle(x=x, y=y, size=1, fill_color="#FF0000")
current_point = p.circle(size=3, fill_color="#0000FF")

# set up textarea (div)
div = Div(
    text="""
          <p>Select the circle's size using this control element:</p>
          """,
    width=200,
    height=30,
)

# set up spinner
spinner = Spinner(
    title="Circle size",
    low=0.1,
    high=5,
    step=0.1,
    value=points.glyph.size,
    width=200,
)
spinner.js_link("value", points.glyph, "size")

# set up RangeSlider
range_slider = RangeSlider(
    title="Adjust x-axis range",
    start=0,
    end=10,
    step=dt,
    value=(p.x_range.start, p.x_range.end),
)

range_slider.js_link("value", p.x_range, "start", attr_selector=0)
range_slider.js_link("value", p.x_range, "end", attr_selector=1)


# moving point using a slider
slider = Slider(
    title="Move point",
    start=0,
    end=10,
    step=dt,
    value=0,
)
slider.js_link("value", current_point.glyph, "x")


# create layout
layout = layout(
    [
        [div, spinner],
        [range_slider, slider],
        [p],
    ]
)

# show result
show(layout)