import streamlit as st
#from st_aggrid import AgGrid
import pandas as pd
import numpy as np
import seaborn as sns
import simpy
import random
import plotly.express as px
import time

st.title('AI Ventures Studio Simulator®')

# Sidebar top-level variables
with st.sidebar:
    st.header('Studio Inputs')
    SIM_TIME = st.number_input('Simulation Time (Days)', value=200,min_value=100, key='sim_time')
    st.write('Resources')
    NUM_VALIDATORS = st.number_input('Num. Builders', value=3, min_value=1, key='num_builders')
    NUM_BUILDERS = NUM_VALIDATORS
    VALIDATOR_PRODUCTIVITY = st.number_input('Builder Productivity', value=2, min_value=1, key='bldr_productivity')
    BUILDER_PRODUCTIVITY = VALIDATOR_PRODUCTIVITY
    st.write('Advanced')
    with st.expander('Expand:'):
        P = st.number_input('Projects per Simulation', value=100,min_value=100, key='big_p')
        P = int(P)
        DURATION_WEIBULL_SHAPE = st.slider('Weibull shape',value=1.29,min_value=0.0,max_value=5.0,step=0.01)

# Stage input variables in main body
st.subheader('Input Variables')
with st.expander('Expand to change inputs:'):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.write('1: Validating')
    col2.write('2: Recruiting')
    col3.write('3: Building')
    col4.write('4: Preseed')
    col5.write('5: Seed')
    s1d = col1.number_input('Duration', value=30, key='stage1_duration')
    s2d = col2.number_input('', value=60, key='stage2_duration')
    s3d = col3.number_input('', value=90, key='stage3_duration')
    s4d = col4.number_input('', value=30, key='stage4_duration')
    s5d = col5.number_input('', value=200, key='stage5_duration')
    s1c = col1.number_input('Conversion', value=0.50, key='stage1_conversion')
    s2c = col2.number_input('', value=0.95, key='stage2_conversion')
    s3c = col3.number_input('', value=0.90, key='stage3_conversion')
    s4c = col4.number_input('', value=1.0, key='stage4_conversion')
    s5c = col5.number_input('', value=0.7, key='stage5_conversion')
    DURATION_MEAN = [int(s1d), int(s2d), int(s3d), int(s4d), int(s5d)]
    CONVERSION_MEAN = [s1c, s2c, s3c, s4c, s5c]

sim_button = st.button('Run Simulation!')
# Setup and start the simulation
# BASE CASE

# Global variables
T_INTER = 7              # Add a new project to backlog every 7 days.
SIM_TIME = int(SIM_TIME)           # Simulation time in days

def run_sim():
    st.write('Starting run...')
    st.write('*'*20)
    st.write('Starting with project backlog of %d projects.' % P)
    st.write('Simulating over %d days...' % SIM_TIME)
    start = time.time()
    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, NUM_VALIDATORS*VALIDATOR_PRODUCTIVITY, NUM_BUILDERS*BUILDER_PRODUCTIVITY, T_INTER, P, 1))
    # Execute!
    env.run(until=SIM_TIME)
    end = time.time()
    st.write('Run 1 completed in %.4f sec' % (end-start))
    data = pd.read_csv('results/studiosim_run_1.csv', index_col=0)

    # Metrics
    total = data['ID'].nunique()
    vals = data[(data['Stage'] == 'Validating') & (data['Outcome'] == 1)].shape[0]
    buis = data[(data['Stage'] == 'Building') & (data['Outcome'] == 1)].shape[0]
    dvals = data[(data['Stage'] == 'Validating')]['Duration'].mean()
    dbuis = data[(data['Stage'] == 'Building')]['Duration'].mean()

    st.metric(label="Total Projects", value='%d projects' % total)
    col6, col7, col8 = st.columns(3)
    col6.metric(label="Converted Validations", value=('%d projects' % vals))
    col6.metric(label="Converted Builds", value=('%d startups' % buis))
    col7.metric(label="Validation Conversion", value=('%.1f%%' % round(vals/total*100,3)))
    col7.metric(label="Build Conversion", value=('%.1f%%' % round(buis/total*100,3)))
    col8.metric(label="Validation Duration", value=('%.1f days' % round(dvals,3)))
    col8.metric(label="Build Duration", value=('%.1f days' % round(dbuis,3)))

    # Charts
    fig = px.timeline(data, title='Gantt Chart', x_start="Start", x_end="End", y="ID",color='Stage')
    fig.layout.xaxis.type = 'linear'
    fig.data[0].x = data['Duration'].tolist()
    fig['layout']['yaxis']['autorange'] = "reversed"
    for d in fig.data:
        filt = data['Stage'] == d.name
        d.x = data[filt]['Duration'].tolist()
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    #st.subheader('Raw Data')
    #AgGrid(data, enable_enterprise_modules=True, allow_unsafe_jscode=True)


def generate_projects(p):
    # Creates matrix of shape (P, 5) filled with projected durations by stage
    # Instead of random durations, we project duration by stage using a Weibull distribution with shape = 1.29
    durations = (np.random.weibull(DURATION_WEIBULL_SHAPE, p*5).reshape(p,5)*DURATION_MEAN).astype(int)
    # Builds are much easier to control the duration for. Instead of a weibull, we'll use a uniform distribution with a median of 90.
    durations[:, 2] = np.random.randint(low=70, high=110, size=p)
    # Sometimes, stage 1 durations are initialized as a decimal, which is truncated to a 0. Change those to a floor of 1.
    durations[:, 0] = np.where(durations[:,0] == 0, 1, durations[:,0])
    # Creates matrix of shape (P, 5) filled with random decimals
    # If decimal is less than expected conversion rate, that counts as a conversion (1)
    original_conversions = (np.random.rand(1, p*5).reshape(p,5)<=CONVERSION_MEAN).astype(int)
    # If a project fails in stage i, it must be marked as failed in stages i through n.
    # Iterates through conversion matrix, modifies future stages as failed if ANY preceding stage failed.
    ccopy = original_conversions.copy()
    for i in range(1, 5):
        ccopy[:, i] = original_conversions[:, 0:i+1].all(axis=1)

    conversions = ccopy
    active_stages = np.concatenate((np.ones([conversions.shape[0],1]), conversions.copy()),axis=1).astype(int)
    return (durations, conversions, active_stages)

# Discrete Event Simulator setup
class Studio(object):
    def __init__(self, env, num_validators, num_builders, run_number):
        self.env = env
        self.outfile = 'results/studiosim_run_%d.csv' % run_number
        self.validator = simpy.FilterStore(env, num_validators)
        self.completed_tasks = []
        self.tasks_df = pd.DataFrame(columns = ['ID', 'Stage', 'Builder', 'Start', 'End', 'Duration', 'Outcome'])
        for i in range(1, VALIDATOR_PRODUCTIVITY+1):
            self.validator.put({'id': 'RC%d' % i})
            self.validator.put({'id': 'BZ%d' % i})
            self.validator.put({'id': 'KB%d' % i})

        self.builder = simpy.FilterStore(env, num_builders)
        for i in range(1, BUILDER_PRODUCTIVITY+1):
            self.builder.put({'id': 'RC%d' % i})
            self.builder.put({'id': 'BZ%d' % i})
            self.builder.put({'id': 'KB%d' % i})

    def validate(self, project, duration):
        #print('%s enters validation at %d' % (project, env.now))
        yield self.env.timeout(duration)
        #print("%s validated at time %d" % (project, env.now))

    def recruit(self, project, duration):
        #print('%s enters FIR recruiting at %d' %(project, env.now))
        yield self.env.timeout(duration)
        #print("%s FIR recruited at time %d" % (project, env.now))

    def build(self, project, duration):
        #print('%s enters build at %d' % (project, env.now))
        yield self.env.timeout(duration)
        #print("%s built at time %d" % (project, env.now))


def project(env, name, studio, stage_durations, stage_outcomes):
    """The car process (each car has a ``name``) arrives at the carwash
    (``cw``) and requests a cleaning machine.

    It then starts the washing process, waits for it to finish and
    leaves to never come back ...

    """
    #print('%s enters backlog at time %d' % (name, env.now))

    # Backlog to Validation
    # Assign builder to project and save ID (this comes in handy later)
    val = yield studio.validator.get()
    builder_id = val['id'][:2]
    start = env.now
    duration = stage_durations[0]
    outcome = stage_outcomes[1]
    yield env.process(studio.validate(name, duration))
    end = env.now
    studio.validator.put(val)

    # Add this project to the completed validations list
    p_id = int(name[10:])
    studio.completed_tasks.append(pd.Series([name, 'Validating', builder_id, start, end, duration, outcome], index=['ID', 'Stage', 'Builder', 'Start', 'End', 'Duration', 'Outcome']))

    if outcome == 0:
        return
    # Validation to Recruiting
    start = env.now
    duration = stage_durations[1]
    outcome = stage_outcomes[2]
    yield env.process(studio.recruit(name, stage_durations[1]))
    end = env.now

    studio.completed_tasks.append(pd.Series([name, 'Recruiting', builder_id, start, end, duration, outcome], index=['ID', 'Stage', 'Builder', 'Start', 'End', 'Duration', 'Outcome']))

    if outcome == 0:
        return
    # Recruiting to Building
    # The selected Builder resource id MUST match the Validator resource id
    bldr = yield studio.builder.get(lambda b: b['id'][:2] == builder_id)
    start = env.now
    duration = stage_durations[2]
    outcome = stage_outcomes[3]
    yield env.process(studio.build(name, duration))
    end = env.now
    studio.builder.put(bldr)
    #print('%s exits build at %d.' % (name, env.now))

    studio.completed_tasks.append(pd.Series([name, 'Building', builder_id, start, end, duration, outcome], index=['ID', 'Stage', 'Builder', 'Start', 'End', 'Duration', 'Outcome']))
    studio.tasks_df = pd.DataFrame(studio.completed_tasks)
    studio.tasks_df.to_csv(studio.outfile)


def setup(env, num_validators, num_builders, t_inter, p, run_number):
    """Create a carwash, a number of initial cars and keep creating cars
    approx. every ``t_inter`` minutes."""
    # Create the Studio
    studio = Studio(env, num_validators, num_builders, run_number)
    id_index = 0

    durations, conversions, active_stages = generate_projects(p)
    # Create 4 initial cars
    #for i in range(int(p/5)):
    #    # pull ith row from actual_durations array
    #    stage_durations = durations[i]
    #    stage_outcomes = active_stages[i]
    #    env.process(project(env, 'Project ID%d' % i, studio, stage_durations, stage_outcomes))
    #    print(i)

    while True:
        if id_index > int(p/5):
            yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
            print('%d, Picking up project from backlog' % id_index)
            # FIFO - get first item in array, then remove so we don't duplicate
            stage_durations = durations[0]
            stage_outcomes = active_stages[0]

            # Removes project from backlog
            durations = np.delete(durations, 0, 0)
            active_stages = np.delete(active_stages, 0, 0)

            env.process(project(env, 'Project ID%d' % id_index, studio, stage_durations, stage_outcomes))
            id_index+=1
        else:
            stage_durations = durations[0]
            stage_outcomes = active_stages[0]

            # Removes project from backlog
            durations = np.delete(durations, 0, 0)
            active_stages = np.delete(active_stages, 0, 0)

            env.process(project(env, 'Project ID%d' % id_index, studio, stage_durations, stage_outcomes))
            id_index+=1


    # Add 3 new projects to the backlog every week
    """while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        print(i, durations.shape)
        i += 1
        if i < len(durations):
            print(durations.shape)
            stage_durations = durations[i]
            stage_outcomes = active_stages[i]
        else:
            # Add 3 more projects
            inc_durations, inc_conversions, inc_active_stages = generate_projects(3)

            # Convert to lists for appendation
            dlist = durations.tolist()
            idlist = inc_durations.tolist()
            print(durations.shape, len(dlist))
            durations = np.array(dlist.append(idlist))

            alist = active_stages.tolist()
            ialist = inc_active_stages.tolist()
            active_stages = np.array(alist.append(ialist))

            stage_durations = durations[i] # beep boop
            stage_outcomes = active_stages[i]

        env.process(project(env, 'Project ID%d' % i, studio, stage_durations, stage_outcomes))"""

# Main streamlit run
if sim_button:
    run_sim()
