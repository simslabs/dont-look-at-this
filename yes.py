import plotly.express as px
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

np.random.seed(123)
N = 50
d = np.random.exponential(10, size=(N,))
event_observed = np.random.binomial(1, 0.5, size=(N,))

kmf = KaplanMeierFitter()
kmf.fit(d, event_observed=event_observed, label="Not CR")

d = np.random.exponential(15, size=(N,))
event_observed = np.random.binomial(1, 0.7, size=(N,))

kmf_2 = KaplanMeierFitter()
kmf_2.fit(d, event_observed=event_observed, label="CR")

# Calculate the survival function for each model
t = np.linspace(0, 30, 100)
y = kmf.survival_function_
y2 = kmf_2.survival_function_

# Combine the calculated survival functions into a single data frame
df = pd.DataFrame({'timeline': t, 'Not CR': y['Not CR'].values, 'CR': y2['CR'].values})

# Use Plotly Express to create the figure
fig = px.line(df, x="timeline", y="value", color='variable', color_discrete_sequence=['#636efa', '#ef553b'],
             line_group='variable', line_dash='solid', line_shape='linear',
             render_mode='svg', hover_name='variable')

fig.update_layout(legend=dict(title=dict(text='variable'), tracegroupgap=0), margin=dict(t=60), template='plotly_white')

fig.show()
