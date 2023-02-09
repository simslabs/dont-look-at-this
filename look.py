import plotly.express as px
import plotly.graph_objs as go
from lifelines import KaplanMeierFitter

# Create example data for sample 1
durations1 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
event_observed1 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

# Create example data for sample 2
durations2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
event_observed2 = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]

# Fit the models
kmf1 = KaplanMeierFitter()
kmf1.fit(durations1, event_observed1, label='Sample 1')

kmf2 = KaplanMeierFitter()
kmf2.fit(durations2, event_observed2, label='Sample 2')

# Plot the complete response curves
fig = go.Figure()

kmf1_data = kmf1.survival_function_
kmf2_data = kmf2.survival_function_

fig.add_trace(go.Scatter(x=kmf1_data.index, y=kmf1_data['Sample 1'], name='Sample 1'))
fig.add_trace(go.Scatter(x=kmf2_data.index, y=kmf2_data['Sample 2'], name='Sample 2'))

fig.update_layout(title='Progression Free Survival Comparison', xaxis_title='Time (days)', yaxis_title='Probability of Survival')

fig.show()

from lifelines.statistics import logrank_test

results = logrank_test(durations1, durations2, event_observed1, event_observed2)
print(results.print_summary())

'''
This will print a summary of the log-rank test, including the test statistic, the p-value, and the conclusion. A small p-value (typically less than 0.05) indicates that there is a statistically significant difference between the survival rates of the two groups.

Note that the log-rank test is only valid if the censoring is non-informative, meaning that the censored observations are missing at random. If the censoring is informative, you should use other methods, such as regression models, to account for the dependence between the censoring and survival times.
'''

'''
Yes, that's correct! In survival analysis, censoring is considered informative when the probability of being censored is dependent on the time to the event of interest. For example, if patients are more likely to be censored (i.e., drop out of a study) if they have a poor prognosis, then the censoring is informative.

Informative censoring can introduce bias into the estimates of the survival distribution and make it difficult to compare survival rates between groups. This is because the distribution of the censoring times is not random and may be related to the time to event.

For example, if a study is only following patients for a certain amount of time, and patients who have not had an event by that time are censored, then the censoring is considered non-informative. However, if patients are dropped from the study for other reasons (e.g., they move away, they refuse to continue participating, etc.), the censoring is considered informative.

In general, it is desirable to have non-informative censoring in survival analysis so that the results are not biased. However, it can be difficult to determine whether censoring is informative or not, and sometimes additional information (e.g., from additional sources or through sensitivity analyses) is needed to make this determination.
'''

from lifelines.statistics import weighted_logrank_test

weights1 = ... # calculate the IPW weights for group 1
weights2 = ... # calculate the IPW weights for group 2

results = weighted_logrank_test(durations1, durations2, event_observed1, event_observed2, weights1, weights2)
print(results.print_summary())

from lifelines import CoxPHFitter
import pandas as pd

# create a data frame with the durations, censoring indicators, class identifier, and other covariates
data = pd.DataFrame({'duration': durations, 'event_observed': event_observed, 'class_id': class_id, 'covariate1': covariate1, 'covariate2': covariate2})

# fit the Cox proportional hazards regression model
cph = CoxPHFitter()
cph.fit(data, 'duration', event_col='event_observed')

# calculate the IPW weights
weights = 1 / cph.predict_survival_function(data).loc[:, 0].values

import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# create an array of durations for each subject
durations = np.array([2.4, 3.1, 5])

# create arrays of durations and event_observed for subjects with PFS >= 3
durations_3 = durations[durations >= 3]
event_observed_3 = np.ones(len(durations_3))

# create arrays of durations and event_observed for subjects with PFS < 3
durations_lt3 = durations[durations < 3]
event_observed_lt3 = np.zeros(len(durations_lt3))

# fit the Kaplan-Meier model for subjects with PFS >= 3
kmf_3 = KaplanMeierFitter()
kmf_3.fit(durations_3, event_observed_3)

# fit the Kaplan-Meier model for subjects with PFS < 3
kmf_lt3 = KaplanMeierFitter()
kmf_lt3.fit(durations_lt3, event_observed_lt3)

# plot the survival curves
plt.step(kmf_3.timeline, kmf_3.survival_function_, where="post", label="PFS >= 3")
plt.step(kmf_lt3.timeline, kmf_lt3.survival_function_, where="post", label="PFS < 3")
plt.legend()


import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# create an array of durations for each subject
durations = np.array([2.4, 3.1, 5])

# create arrays of durations and event_observed for subjects with PFS >= 3
durations_3 = durations[durations >= 3]
event_observed_3 = np.ones(len(durations_3))

# create arrays of durations and event_observed for subjects with PFS < 3
durations_lt3 = durations[durations < 3]
event_observed_lt3 = np.zeros(len(durations_lt3))

# fit the Kaplan-Meier model for subjects with PFS >= 3
kmf_3 = KaplanMeierFitter()
kmf_3.fit(durations_3, event_observed_3)

# fit the Kaplan-Meier model for subjects with PFS < 3
kmf_lt3 = KaplanMeierFitter()
kmf_lt3.fit(durations_lt3, event_observed_lt3)

# plot the survival curves
plt.step(kmf_3.timeline, kmf_3.survival_function_, where="post", label="PFS >= 3")
plt.step(kmf_lt3.timeline, kmf_lt3.survival_function_, where="post", label="PFS < 3")
plt.legend()
plt.xlabel("Time (years)")
plt.ylabel("Probability of Survival")
plt.title("Kaplan-Meier Survival Curves")
plt.show()

import plotly.express as px
import pandas as pd

# Create example data
pfs_greater_3 = [3.5, 4.2, 4.6, 5.1, 5.5]
pfs_less_3 = [2.1, 2.4, 2.7, 2.9]

# Create a dataframe from the data
df = pd.DataFrame({'PFS': pfs_greater_3 + pfs_less_3,
                   'Subgroup': ['Greater 3 years'] * len(pfs_greater_3) + ['Less 3 years'] * len(pfs_less_3)})

# Plot the Kaplan-Meier curve
fig = px.line(df, x="PFS", y="survival_prob", color="Subgroup",
              title="Kaplan-Meier Curve", log_y=False,
              labels={"PFS": "Progression-free survival (years)",
                      "survival_prob": "Survival Probability"})

# Set the background to white
fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Show the plot
fig.show()
