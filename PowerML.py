#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ugwumadu Chinonso, Jose Tabares, and Anup Pandey"
__affiliation__ = "A-1 Group, Los Alamos National Laboratory, NM, USA."
__credits__ = ["Ugwumadu Chinonso, Jose Tabares, and Anup Pandey"]
__version__ = "0.0.1"
__email__ = "cugwumadu@lanl.gov"

print("######################################## S T A R T  O F  T H E  A L G O R I T M ##########################################################")
print()

print('Greetings! I am PowerModel-AI, born from the research publication "PowerModel-AI: A First On-the-fly Machine-Learning Predictor for AC Power Flow Solutions." \n\
Feel free to use my capabilities for non-profit endeavors, but please remember to cite my paper in your work.\nThank you, and enjoy exploring my features!\n')


#****************************************************************** Introduction ********************************************************************#
# This script contains the algorithm for PowerModel-AI. It is associated to our manuscript titled: PowerModel-AI: A First On-the-fly Machine-Learning 
# Predictor for AC Power Flow Solutions. 

# While this script is totally free for academic (non-profit) use, We would appreciate citation of our paper when you use any part of this script for 
# your own work. Thank You and Enjoy!
#***************************************************************************************************************************************************#

welcome = 'Greetings! I am PowerModel-AI, born from the research publication "PowerModel-AI: A First On-the-fly Machine-Learning Predictor for AC Power Flow Solutions." \n \n\
Feel free to use my capabilities for non-profit endeavors, but please remember to cite my paper in your work.\n \nThank you, and enjoy exploring my features!\n'

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import julia

from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import json

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import random as ran
import numpy as np
import pandas as pd
import time
#----------------------------------------------------------- End: Library Imports ---------------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#----------------------------------------------------------- Set-UP Stage 1 ---------------------------------------------------------------#

## Page Set up
st.set_page_config(
    page_title=f"PowerModel-AI",
    page_icon="⚡",
    layout="wide"
)

#----------------- Parameters ---------------------------------#
logo = "./etc/Setup Files/PM_LOGO.png"
lanl_logo = "./etc/Setup Files/LANL_LOGO.png"
domain_file = "/Training Domain.txt"
gen_power_limit_file = "/Gen Power Limits.txt"
julia_params = "./etc/Setup Files/Julia Parameters.txt"
retrain_folder = "./etc/on-the-fly"
#--------------------------------------------------------------#

welcome = '\n\n\n\n # Hello there! 👋 \
\n \n \n ###### I am :blue[PowerModel-:flag-ai:] :robot_face:, conceived from the research publication titled: ***PowerModel-:flag-ai::  A First On-the-fly Machine-Learning Predictor for AC Power Flow Solutions.***\
\n \n ###### Feel free to use my capabilities for :blue-background[non-profit endeavors] :100:, but please remember to :violet[cite my parents] :point_up_2: in your work.\
\n \n ###### Thank you, and enjoy exploring my features! :tada:\n'



#----------------- Functions ----------------------------------# 

def stream_data():
    for word in welcome.split(" "):
        yield word + " "
        time.sleep(0.08)



def readTrainingDomain(txt_path):
    """
    Reads the new training domain limit from a text file.

    The function expects the text file to contain domain limits, where the first line is ignored, and the second line
    contains the new training domain limit as a numeric value.

    Parameters
    ----------
    txt_path : str
        Path to the text file containing the training domain limits.

    Returns
    -------
    float
        The new training domain limit as a float value.

    Notes
    -----
    The text file format should be similar to:

    ```
    base_training_domain_limit: <base_value>
    new_training_domian_limit:  <new_value>
    ```

    Example
    -------
    Assuming `domain_limits.txt` contains:
    
    ```
    base_training_domain_limit:    100
    new_training_domian_limit:    150
    ```
    Calling `readTrainingDomain('domain_limits.txt')` will return:
    
    ```
    150.0
    ```
    """
    with open(txt_path, "r") as f:
       a = f.readline()
       line = f.readline()
       upper_domain_st = np.array(line.rstrip('\t').split()[1]).astype(np.float64)
    return float(upper_domain_st)
#*******************************************************************************************#

def writeTrainingDomain(txt_path, new_domain, base_training_domain_limit):
    """
    Writes training domain limits to a text file.

    The file will contain two key-value pairs:
    - The base training domain limit.
    - The new training domain limit.

    Parameters
    ----------
    txt_path : str
        Path to the output text file where the domain limits will be written.
    new_domain : str
        The new training domain limit to be written in the file.
    base_training_domain_limit : str
        The base training domain limit to be written in the file.

    Returns
    -------
    None

    Notes
    -----
    The output file will have the following format:
    
    ```
    base_training_domain_limit: <base_training_domain_limit_value>
    new_training_domian_limit: <new_domain_value>
    ```

    Example
    -------
    If `base_training_domain_limit = "100"`, `new_domain = "150"`, and `txt_path = "domain_limits.txt"`, 
    the file `domain_limits.txt` will contain:
    
    ```
    base_training_domain_limit:    100
    new_training_domian_limit:    150
    ```
    """
    with open(txt_path, "w") as o:
        o.write("base_training_domain_limit:" + "\t" + base_training_domain_limit + '\n')
        o.write("new_training_domian_limit:" + "\t" + new_domain)
#*******************************************************************************************#

def readGenPowerLimit(txt_path):
    """
    Reads generator power limits from a text file.

    The text file is expected to have two lines:
    - The first line contains space-separated variable names (e.g., generator IDs or labels).
    - The second line contains corresponding numeric power limit values.

    Parameters
    ----------
    txt_path : str
        Path to the text file containing generator power limit data.

    Returns
    -------
    tuple
        power_limit_var : list of str
            A list of variable names representing generator identifiers.
        power_limit_val : numpy.ndarray
            A NumPy array of generator power limits as float values.

    Notes
    -----
    The text file format should follow this structure:
    
    - First line: variable names (strings).
    - Second line: space-separated numeric values (floats) for each generator.

    Example
    -------
    Assuming `power_limits.txt` contains the following:
    
    ```
    Gen1 Gen2 Gen3
    100.0 200.0 300.0
    ```
    Calling `readGenPowerLimit('power_limits.txt')` will return:
    
    ```
    (['Gen1', 'Gen2', 'Gen3'], array([100.0, 200.0, 300.0]))
    ```
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()
        power_limit_var = lines[0].rstrip('\n').split()
        power_limit_val = np.array(lines[1].rstrip('\n').split()).astype(np.float64)
    return power_limit_var, power_limit_val
#*******************************************************************************************#

def writeGenPowerLimit(txt_path, gen_power_limit_dict):
    """
    Writes generator power limits to a text file.

    The file will contain two lines:
    - The first line consists of variable names (generator IDs or labels) separated by tabs.
    - The second line contains the corresponding power limit values, also separated by tabs.

    Parameters
    ----------
    txt_path : str
        Path to the output text file where the generator power limits will be written.
    gen_power_limit_dict : dict
        A dictionary where keys are generator variable names (str) and values are the corresponding power limits (float or int).

    Returns
    -------
    None

    Notes
    -----
    The output file will have the following format:
    
    ```
    Gen1    Gen2    Gen3
    100     200     300
    ```

    Example
    -------
    If `gen_power_limit_dict = {"Gen1": 100, "Gen2": 200, "Gen3": 300}` and `txt_path = "gen_limits.txt"`, 
    the file `gen_limits.txt` will contain:

    ```
    Gen1            Gen2            Gen3
    100             200             300
    ```
    """
    with open(txt_path, 'w') as o:
        line_one = [var_name + "\t\t" for var_name in gen_power_limit_dict.keys()]
        o.write("".join(line_one) + '\n')
        line_two = [str(val) + "\t\t\t" for val in gen_power_limit_dict.values()]
        o.write("".join(line_two) + '\n')
#*******************************************************************************************#

def writeJuliaPath(txt_path, julia_path):
    """
    Writes the specified Julia installation path to a text file.

    Parameters
    ----------
    txt_path : str
        Path to the output text file where the Julia path will be written.
    julia_path : str
        The Julia installation path to be written in the file.

    Returns
    -------
    None

    Example
    -------
    If `julia_path = "/usr/local/bin/julia"` and `txt_path = "julia_path.txt"`, 
    the file `julia_path.txt` will contain:

    ```
    /usr/local/bin/julia
    ```
    """
    with open(txt_path, "w") as o:
        o.write(julia_path)
#*******************************************************************************************#

def cleanSlate():
    if "training_domain" in st.session_state:
        del st.session_state["training_domain"]


def getUpperBound():
    if "training_domain_UB" in st.session_state:
        st.session_state["training_domain_UB"] = readTrainingDomain(retrain_folder+ "//" + selected_dirname + domain_file)

#*******************************************************************************************#

def disableCaseSelect(b):
    """
    Disables or enables case selection and removes the training domain from session state if it exists.

    Args:
        b (bool): Boolean indicating whether to disable (True) or enable (False) case selection.

    Returns:
        None

    Side Effects:
        - Sets `st.session_state["disableCaseSelect"]` to the value of `b`.
        - Deletes `st.session_state["training_domain"]` if it is present in session state.

    Example:
        Disable case selection and clear training domain:

        >>> disableCaseSelect(True)

        Enable case selection:

        >>> disableCaseSelect(False)
    """
    st.session_state["disableCaseSelect"] = b
    cleanSlate()
#*******************************************************************************************#

def disableRetrain(b):
    """
    Disables or enables the retraining option in session state.

    Args:
        b (bool): Boolean indicating whether to disable (True) or enable (False) retraining.

    Returns:
        None

    Side Effects:
        - Sets `st.session_state["disableRetrain"]` to the value of `b`.

    Example:
        Disable retraining:

        >>> disableRetrain(True)

        Enable retraining:

        >>> disableRetrain(False)
    """
    st.session_state["disableRetrain"] = b
#*******************************************************************************************#

def retainBuses():
    """
    Retains the current state of `load_changes` in the session state.

    If `load_changes` exists in `st.session_state`, it will reassign the value of 
    `st.session_state.load_changes` to itself, ensuring the value is retained.

    Returns:
        None

    Side Effects:
        - Reassigns `st.session_state.load_changes` to itself if it exists.

    Example:
        To retain the value of `load_changes`:

        >>> retain()
    """
    if "load_changes" in st.session_state:
        st.session_state.load_changes = st.session_state.load_changes
    getUpperBound() #NEW

def runPowerModelJL():
    st.session_state.run = True


#----------------------------------------------------------- End: Set-UP Stage 1 ----------------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#----------------------------------------------------------- Set-UP Stage 2 ---------------------------------------------------------------#

## More set-up initializations
st.markdown("## PowerModel-AI")
st.sidebar.image(logo, width=200)
st.session_state.getButton = 0
num_buses = 10
dir_path='./CaseModels'
dir_names = os.listdir(dir_path)

# The power grid is selected using this select box
selected_dirname = st.sidebar.selectbox('Select Bus Case:', dir_names, index = None,  placeholder="Select Power Grid",key="selected_directory", on_change=disableCaseSelect, args=(False,), help="Select bus case to analyze.")

if selected_dirname:
    # Read the base model training domain
    with open(retrain_folder+ "//" + selected_dirname + domain_file, "r") as f:
        line1 = f.readline()
        base_training_domain_limit = float(line1.rstrip('\t').split()[1])

    # Get the name for the current workifn directory.
    num_buses = int(selected_dirname[3:].replace("-Bus", "").replace("K","000"))
    st.session_state.dir_name_st = os.path.join(dir_path, selected_dirname) #file_selector()
    dir_name = st.session_state.dir_name_st

    # Get the upper training limit in file.
    st.session_state.training_domain_limit = readTrainingDomain(retrain_folder+ "//" + selected_dirname + domain_file)
    training_domain_limit = st.session_state.training_domain_limit

    # this initialize the training domains to be used later
    if "training_domain" not in  st.session_state:
        st.session_state.training_domain = 2.5 # the max to which we have trained the model
        training_domain = st.session_state.training_domain

        st.session_state.training_domain_UB = training_domain_limit
        training_domain_UB = st.session_state.training_domain_UB

        st.session_state.training_domain_LB = 0.01
        training_domain_LB  = st.session_state.training_domain_LB

        #*** Debug ***#
        print(f"A: After training domain was added to session state for: {selected_dirname}")
        print(f"A: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain_limit}")
        #*** Debug ***#

    if "training_domain" in  st.session_state:
                 
        training_domain = st.session_state.training_domain
        training_domain_LB  = st.session_state.training_domain_LB
        training_domain_UB = np.round(st.session_state.training_domain_UB,2)

        #*** Debug ***#
        print(f"C: While training domain exist in session state for: {selected_dirname}")
        print(f"C: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain_limit}")
        #*** Debug ***#

#----------------------------------------------------------- End: Set-UP Stage 2 ----------------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#----------------------------------------------------------- Set-UP Stage 3 ---------------------------------------------------------------#


## File Name Initialization for the Model and Associated Parameters
st.session_state.gen_model_name_MW = '/_genModel_MW.pth'
gen_model_name_MW = st.session_state.gen_model_name_MW 
st.session_state.gen_model_name_Mvar = '/_genModel_Mvar.pth'
gen_model_name_Mvar = st.session_state.gen_model_name_Mvar 
st.session_state.gen_features = "/_genFeatures.npz"
gen_features = st.session_state.gen_features
st.session_state.gen_targets_MW = "/_genTargets_MW.npz"
gen_targets_MW= st.session_state.gen_targets_MW
st.session_state.gen_targets_Mvar = "/_genTargets_Mvar.npz"
gen_targets_Mvar= st.session_state.gen_targets_Mvar

st.session_state.base_model_name_vpu = '/_baseModel_vpu.pth'
base_model_name_vpu = st.session_state.base_model_name_vpu 
st.session_state.base_model_name_vang = '/_baseModel_vang.pth'
base_model_name_vang = st.session_state.base_model_name_vang
st.session_state.base_features_vpu = "/_baseFeatures_vpu.npz"
base_features_vpu= st.session_state.base_features_vpu
st.session_state.base_features_vang = "/_baseFeatures_vang.npz"
base_features_vang= st.session_state.base_features_vang
st.session_state.base_targets_vpu = "/_baseTargets_vpu.npz"
base_targets_vpu = st.session_state.base_targets_vpu
st.session_state.base_targets_vang = "/_baseTargets_vang.npz"
base_targets_vang = st.session_state.base_targets_vang

st.session_state.updated_model_name_vpu  = '/_updatedModel_vpu.pth'
updated_model_name_vpu  = st.session_state.updated_model_name_vpu 
st.session_state.updated_model_name_vang  = '/_updatedModel_vang.pth'
updated_model_name_vang   = st.session_state.updated_model_name_vang  
st.session_state.retrain_bus_features_vpu = "/_updatedFeatures_vpu.npz"
retrain_bus_features_vpu = st.session_state.retrain_bus_features_vpu
st.session_state.retrain_bus_features_vang = "/_updatedFeatures_vang.npz"
retrain_bus_features_vang = st.session_state.retrain_bus_features_vang
st.session_state.retrain_bus_targets_vpu = "/_updatedTargets_vpu.npz"
retrain_bus_targets_vpu = st.session_state.retrain_bus_targets_vpu
st.session_state.retrain_bus_targets_vang = "/_updatedTargets_vang.npz"
retrain_bus_targets_vang = st.session_state.retrain_bus_targets_vang

st.session_state.updated_gen_model_name_MW = '/_updatedGenModel_MW.pth'
updated_gen_model_name_MW = st.session_state.updated_gen_model_name_MW
st.session_state.updated_gen_model_name_Mvar = '/_updatedGenModel_Mvar.pth'
updated_gen_model_name_Mvar = st.session_state.updated_gen_model_name_Mvar
st.session_state.retrain_gen_features = "/_updatedGenFeatures.npz"
retrain_gen_features = st.session_state.retrain_gen_features
st.session_state.retrain_gen_targets_MW = "/_updatedGenTargets_MW.npz"
retrain_gen_targets_MW = st.session_state.retrain_gen_targets_MW
st.session_state.retrain_gen_targets_Mvar = "/_updatedGenTargets_Mvar.npz"
retrain_gen_targets_Mvar = st.session_state.retrain_gen_targets_Mvar
#------------------ END OF Model and Feature File Names ------------------#


## Used to filter the bus nodes whose power demands are being updated 
st.session_state.change_list = []
change_list = st.session_state.change_list


## More Functions
#----------------------- Maps in Tab 2 -----------------------------------------# 

def plot_map(state, plot_container, buses, branches, map_title, result_type):
    """
    Plots a map displaying bus locations and branch connections using Plotly.

    The map visualizes bus data and connections between them based on the specified result type,
    which determines the color coding for the buses.

    Parameters
    ----------
    state : object
        The current application state (not used directly in the function).
    plot_container : object
        The Streamlit container where the plot will be displayed.
    buses : DataFrame
        A DataFrame containing bus data with columns for latitude, longitude, size, and bus numbers.
    branches : DataFrame
        A DataFrame containing branch data with columns for latitude, longitude, and nominal voltages.
    map_title : str
        The title of the map to be displayed.
    result_type : str
        The type of result to determine the color scheme for the bus markers. 
        Acceptable values are "vpu" for voltage per unit or any other value for voltage angle.

    Returns
    -------
    None

    Example
    -------
    To plot a map with bus and branch data:

    >>> plot_map(state, plot_container, buses, branches, "Network Map", "vpu")
    """
    if result_type == "vpu": 
        v_color='vpu'
    else:
        v_color = 'vangle'
    fig = px.scatter_mapbox(buses, 
        lat='Latitude:1', 
        lon='Longitude:1',
        color = v_color,
        color_continuous_scale= px.colors.qualitative.Dark2_r, 
        size='size',
        size_max=12, 
        zoom=9,
        hover_name = "BusNum",             
    )
    branch_data = {
        'lats' :    [],
        'lons' :    [],
        'kvs' :     [],
    }
    for index, row in branches.iterrows():
        branch_data['lats'].extend([row['Latitude'], row['Latitude:1'], None])
        branch_data['lons'].extend([row['Longitude'], row['Longitude:1'], None])
        branch_data['kvs'].extend([row['BusNomVolt'], row['BusNomVolt'], None])
    fig2 = px.line_mapbox(
        lat=branch_data['lats'],
        lon=branch_data['lons'],
    )
    fig2.update_traces(line=dict(color="gray", width=2))
    fig.add_traces(fig2.data)
    fig.update_layout(mapbox_style="carto-positron",
        title=map_title,
        width=800, height=800,
                )
    plot_container.plotly_chart(fig, theme="streamlit", use_container_width=True)


def get_name(self, row, state, id):
    """
    Generates a name for the specified row based on the type of object it represents.

    This method constructs a name string depending on the object's type (Buses, ACLines, Transformers, 
    Generators, or Loads) and adds the generated name to the row.

    Parameters
    ----------
    self : object
        The instance of the class this method belongs to.
    row : pandas.Series
        A row of data containing information about the object.
    state : object
        The current application state (not used directly in this method).
    id : object
        An identifier (not used directly in this method).

    Returns
    -------
    pandas.Series
        The updated row with the generated name added as a new key-value pair.

    Example
    -------
    To get a name for a bus row:

    >>> row = pd.Series({"Bus": 1})
    >>> updated_row = get_name(instance, row, state, id)
    >>> print(updated_row['name'])
    "1"
    
    To get a name for a transformer row:

    >>> row = pd.Series({"Bus From": 1, "To Bus": 2, "Ckt": "1"})
    >>> updated_row = get_name(instance, row, state, id)
    >>> print(updated_row['name'])
    "(1,2,1)"
    """
    if self.name == "Buses":
        name = str(row["Bus"])
    elif self.name == "ACLines" or self.name == "Transformers":
        name = "({},{},{})".format(row["Bus From"], row["To Bus"], row['Ckt'])
    elif self.name == "Generators" or self.name == "Loads":
        name = "({},{})".format(row['Bus'], row['ID'])
    row['name'] = name
    return row


def update_voltage(row, v):
    """
    Updates the voltage values in a row based on the provided voltage data.

    This function modifies the given row by setting the voltage per unit (vpu) and voltage angle (vangle)
    values from the provided voltage dictionary or array `v`, using the bus number from the row.

    Parameters
    ----------
    row : pandas.Series
        A row of data containing information about the bus, including its bus number.
    v : dict or array-like
        A collection of voltage values, where the index/key corresponds to the bus number.

    Returns
    -------
    pandas.Series
        The updated row with the voltage values set.

    Example
    -------
    To update the voltage for a specific row:

    >>> row = pd.Series({"BusNum": 1})
    >>> v = {1: 0.95, 2: 0.98}
    >>> updated_row = update_voltage(row, v)
    >>> print(updated_row["vpu"], updated_row["vangle"])
    0.95 0.95
    """
    row["vpu"] = v[row["BusNum"]]
    row["vangle"] = v[row["BusNum"]]
    return row 
#----------------------------------------------------------- E N D     M A P S ------------------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#------------------------------------------------- Wrapper Class for PowerModels.jl -------------------------------------------------------#

class mld:
    """
    Class for generating a Julia file for power system modeling and analysis.

    This class provides methods to create and write to a Julia file, setting up the necessary imports,
    parsing data from a specified model file, and executing power flow analysis.

    Attributes
    ----------
    jlFile : str
        The name of the Julia file to be created.
    modelFile : str
        The name of the model file to be parsed.
    samples : int
        The number of samples for Power Demand Multiplier (to be set externally).
    LB : float
        Lower bound for Power demand change percentage (to be set externally).
    UB : float
        Upper bound for Power demand change percentage (to be set externally).
    jsonName : str
        The name of the JSON file where results will be saved.

    Parameters
    ----------
    jl_file_name : str
        The name of the Julia file to be created.
    model_file : str
        The name of the model file to be parsed.
    """
    
    def __init__(self, jl_file_name, model_file):
        """
        Initializes the mld class with the provided Julia and model file names.

        Parameters
        ----------
        jl_file_name : str
            The name of the Julia file to be created.
        model_file : str
            The name of the model file to be parsed.
        """
        self.jlFile = jl_file_name
        self.modelFile = model_file

    def createJLFile(self):
        """
        Creates and opens a new Julia file for writing.

        Returns
        -------
        None
        """
        self.io = open(self.jlFile, "w+")

    def writeJLHeader(self):
        """
        Writes the header information and necessary imports to the Julia file.

        Returns
        -------
        None
        """
        if not hasattr(self, "io"):
            self.createJLFile()
        self.io.write(f"using PowerModels\n")
        self.io.write(f"import InfrastructureModels\n")
        self.io.write(f"import Memento\n")
        self.io.write(f'Memento.setlevel!(Memento.getlogger(InfrastructureModels), "error")\n')
        self.io.write(f'PowerModels.logger_config!("error")\n')
        self.io.write(f'import Ipopt\n')
        self.io.write(f'import Random\n')
        self.io.write(f'using StatsBase\n')
        self.io.write(f'using Distributions\n')
        self.io.write(f'import JuMP\n')
        self.io.write(f'import Random\n')
        self.io.write(f'using JSON\n\n')
        self.io.write(f'start_time = time()\n')

    def writeJLParse(self):
        """
        Parses the model file and extracts generator information into the Julia file.

        Returns
        -------
        None
        """
        file = self.modelFile
        file = file.replace('\\', "\\\\")
        self.io.write(f'data = PowerModels.parse_file("{file}")\n')
        # Shut off shunts (Future Plan)
        self.io.write(f'buses = []\n') 
        self.io.write('for (i, gen) in data["gen"]\n')
        self.io.write('\tif !(gen["gen_bus"] in buses)\n')
        self.io.write('\t\tappend!(buses,gen["gen_bus"])\n')
        self.io.write('\tend\n')
        self.io.write('end\n')
        self.io.write('gen_dict = Dict{String, Any}()\n')
        self.io.write('genfuel = Dict{String, Any}()\n')
        self.io.write('gentype = Dict{String, Any}()\n')
        self.io.write('lookup_buses = Dict{Int, Any}()\n')
        self.io.write('counter = 1\n')
        self.io.write('for (i, gen) in data["gen"]\n')
        self.io.write('\tif gen["gen_bus"] in keys(lookup_buses) && gen["gen_status"] == 1\n')
        self.io.write('\t\tindx = lookup_buses[gen["gen_bus"]]\n')
        self.io.write('\t\tgen_dict[indx]["pg"] += gen["pg"]\n')
        self.io.write('\t\tgen_dict[indx]["qg"] += gen["qg"]\n')
        self.io.write('\t\tgen_dict[indx]["pmax"] += gen["pmax"]\n')
        self.io.write('\t\tgen_dict[indx]["pmin"] += gen["pmin"]\n')
        self.io.write('\t\tgen_dict[indx]["qmax"] += gen["qmax"]\n')
        self.io.write('\t\tgen_dict[indx]["qmin"] += gen["qmin"]\n')
        self.io.write('\telseif gen["gen_status"] == 1\n')
        self.io.write('\t\tgen["index"] = counter\n')
        self.io.write('\t\tgen_dict["$(counter)"] = gen\n')
        self.io.write('\t\tgenfuel["$(counter)"] = data["genfuel"][i]\n')
        self.io.write('\t\tgentype["$(counter)"] = data["gentype"][i]\n')
        self.io.write('\t\tlookup_buses[gen["gen_bus"]] = "$(counter)"\n')
        self.io.write('\t\tglobal counter += 1\n')
        self.io.write('\tend\n')
        self.io.write('end\n')
        self.io.write('data["gen"] = gen_dict\n')
        self.io.write('data["genfuel"] = genfuel\n')
        self.io.write('data["gentype"] = gentype\n')

    def writeJLGetBaseInfo(self):
        """
        Writes the base information retrieval section to the Julia file.

        Returns
        -------
        None
        """
        self.io.write('nlp_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-6, "print_level"=>0)\n')
        self.io.write('results = Dict{String, Any}()\n')
        self.io.write('results["system"] = data\n')
        self.io.write('result = PowerModels.solve_ac_pf(data, nlp_solver)\n')
        self.io.write('results["base pf"] = data\n')
        self.io.write('results["pf"] = Dict{Int, Any}()\n')
        self.io.write('results["NOpf"] = Dict{Int, Any}()\n')

    def writeJLLoadChangePF(self):
        """
        Writes the Power demand change power flow analysis section to the Julia file.

        Returns
        -------
        None
        """
        self.io.write(f'samples = {self.samples}\n')
        self.io.write('for i = 1:samples\n')
        self.io.write('\tdata_ = deepcopy(data)\n')
        self.io.write('\tl = length(keys(data_["load"]))\n')
        self.io.write('\tn = rand(1:l)\n')
        self.io.write('\tm = sample(1:l, n, replace=false)\n')
        self.io.write('\tdelta = Dict{Int, Any}()\n')
        self.io.write('\tfor (j, k) in enumerate(m)\n')
        self.io.write(f'\t\tpct = rand(Uniform({self.LB},{self.UB}))\n')
        self.io.write('\t\tpd = (pct) * data_["load"]["$(k)"]["pd"]\n')
        self.io.write('\t\tqd = (pct) * data_["load"]["$(k)"]["qd"]\n')
        self.io.write('\t\tdata_["load"]["$(k)"]["pd"] = pd\n')
        self.io.write('\t\tdata_["load"]["$(k)"]["qd"] = qd\n')
        self.io.write('\t\tdelta[k] = pct\n')
        self.io.write('\tend\n')
        self.io.write('\tresult = PowerModels.solve_ac_pf(data_, nlp_solver)\n')
        self.io.write('\tresult["load"] = data_["load"]\n')
        self.io.write('\tresult["delta"] = delta\n')
        self.io.write('\tif result["termination_status"] == LOCALLY_SOLVED\n')
        self.io.write('\t\tresults["pf"][i] = result\n') 
        self.io.write('\telse\n') 
        self.io.write('\t\tresults["NOpf"][i] = result\n')
        self.io.write('\tend\n')
        self.io.write('end\n')

    def writeJLJSON(self):
        """
        Writes the results to a JSON file at the end of the analysis.

        Returns
        -------
        None
        """
        filename = self.jsonName.replace("\\", "/")
        self.io.write(f'open("{filename}","w") do f\n')
        self.io.write('\tJSON.print(f, results)\n')
        self.io.write('end\n')
#------------------------------------------- END: CLASS FOR JULIA TO RUN POWER MODEL ------------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#------------------------------------------ COLLECT POWERMODEL.jl DATA FOR ML PIPELINE ----------------------------------------------------#

def collectGridData(dir_path_, base_case): 
    if __name__ == '__main__':
        filename = ""
        holder = 1
                              
        for file in os.listdir(dir_path_):
            if '.json' in file:
                filename = os.path.join(dir_path_, file)
                f = open(filename)
                data_ = json.load(f)

                base_system = data_["base pf"]
                base_gen = base_system['gen']
                base_buses_wt_load = base_system['load']
                base_bus_dict = base_system["bus"]

                NOpf_data = data_["NOpf"]
                pf_data = data_["pf"]
                pf_iterations = np.array(list(pf_data.keys()))

                feasible_solution_report = f"{len(pf_data.keys())} out of {len(NOpf_data.keys()) + len(pf_data.keys())} events have feaseable solutions"

                base_gen_MW = []
                base_gen_MVar = []
                base_load_MW = []
                base_load_MVar = []
                
                buses_ = []
                buses_wt_gen_ = []
                buses_wt_load_ = []
                
                for the_bus_ in range(1,  num_buses + 1):
                    buses_.append(int(the_bus_)) 
                    
                for the_bus_ in range(1, len(base_gen.keys()) + 1):
                    buses_wt_gen_.append(int(base_gen[str(the_bus_)]['gen_bus']))
                    base_gen_MW.append(base_gen[str(the_bus_)]['pg'])
                    base_gen_MVar.append(base_gen[str(the_bus_)]['qg'])


                for the_bus_ in range(1, len(base_buses_wt_load.keys()) + 1):
                    buses_wt_load_.append(int(base_buses_wt_load[str(the_bus_)]['load_bus']))
                    base_load_MW.append(base_buses_wt_load[str(the_bus_)]['pd'])
                    base_load_MVar.append(base_buses_wt_load[str(the_bus_)]['qd'])


                buses_ = np.array(buses_)
                buses_wt_gen_ = np.array(buses_wt_gen_)
                buses_wt_load_ = np.array(buses_wt_load_)

                
                for run_iter in range(len(pf_iterations)):
                    iter = pf_iterations[run_iter]
                    
                    data_dict = pf_data[str(iter)]["solution"]
                    data_dict_genPower = data_dict['gen']
                                    
                    data_dict_busVPU = data_dict['bus']
                    data_dict_loadedBus = pf_data[str(iter)]["load"]
                    
                    if holder == 1:
                        base_gen = base_system['gen']
                        base_buses_wt_load = base_system['load']
                        base_gen_MW = []
                        base_gen_MVar = []
                        base_load_MW = []
                        base_load_MVar = []
                        
                        buses_ = []
                        buses_wt_gen_ = []
                        buses_wt_load_ = []
                        
                        for the_bus_ in range(1,  len(data_dict_busVPU.keys()) + 1):
                            buses_.append(int(the_bus_))
                            
                        for the_bus_ in range(1, len(base_gen.keys()) + 1):
                            buses_wt_gen_.append(int(base_gen[str(the_bus_)]['gen_bus']))
                            base_gen_MW.append(base_gen[str(the_bus_)]['pg'])
                            base_gen_MVar.append(base_gen[str(the_bus_)]['qg'])

      
                        for the_bus_ in range(1, len(data_dict_loadedBus.keys()) + 1):
                            buses_wt_load_.append(int(data_dict_loadedBus[str(the_bus_)]['load_bus']))
                            base_load_MW.append(base_buses_wt_load[str(the_bus_)]['pd'])
                            base_load_MVar.append(base_buses_wt_load[str(the_bus_)]['qd'])


                        buses_ = np.array(buses_)
                        buses_wt_gen_ = np.array(buses_wt_gen_)
                        buses_wt_load_ = np.array(buses_wt_load_)

                        num_buses_ = len(buses_)
                        bus_vpu_arr = np.empty([len(pf_iterations), num_buses_])
                        bus_vangle_arr = np.empty([len(pf_iterations), num_buses_])
                        load_MW_arr = np.empty([len(pf_iterations),len(buses_wt_load_)])
                        load_Mvar_arr = np.empty([len(pf_iterations),len(buses_wt_load_)])

                        gen_MW_arr = np.zeros([len(pf_iterations), num_buses_])
                        gen_Mvar_arr = np.zeros([len(pf_iterations), num_buses_])

                        base_gen_MW_arr = np.zeros([len(pf_iterations), num_buses_])
                        base_gen_Mvar_arr = np.zeros([len(pf_iterations), num_buses_])

                        sum_generation_MW = np.zeros(len(pf_iterations))
                        sum_generation_Mvar = np.zeros(len(pf_iterations))
                        holder += 1

                    # Get the Power demand data
                    if True:
                          
                        MW_data = [data_dict_loadedBus[str(mw)]['pd'] for mw in range(1, len(data_dict_loadedBus.keys()) + 1)]
                        Mvar_data = [data_dict_loadedBus[str(mw)]['qd'] for mw in range(1, len(data_dict_loadedBus.keys()) + 1)]
    
                        load_MW_arr[run_iter] = np.squeeze(MW_data)  # nodes with load,
                        load_Mvar_arr[run_iter] = np.squeeze(Mvar_data)


                    # Get VPU data
                    if True:
                        vpu_data = [data_dict_busVPU[str(mw)]['vm'] for mw in buses_]
                        vangle_data = [data_dict_busVPU[str(mw)]['va'] for mw in buses_]
                        bus_vpu_arr[run_iter] = np.squeeze(vpu_data) 
                        bus_vangle_arr[run_iter] = np.squeeze(vangle_data)

                    # Get generator data
                    if True: 
                        genMW_data = [data_dict_genPower[str(mw)]['pg'] for mw in range(1, len(data_dict_genPower.keys()) + 1)]
                        genMvar_data = [data_dict_genPower[str(mw)]['qg'] for mw in range(1, len(data_dict_genPower.keys()) + 1)]
                        
                        sum_generation_MW[run_iter] = np.sum(genMW_data)
                        sum_generation_Mvar[run_iter] = np.sum(genMvar_data)

                        gen_MW_arr[run_iter][buses_wt_gen_-1] = np.squeeze(genMW_data)
                        gen_Mvar_arr[run_iter][buses_wt_gen_-1] = np.squeeze(genMvar_data) 

                        base_gen_MW_arr[run_iter][buses_wt_gen_-1] = np.squeeze(base_gen_MW)
                        base_gen_Mvar_arr[run_iter][buses_wt_gen_-1] = np.squeeze(base_gen_MVar)

    if base_case:
        return buses_, buses_wt_load_, buses_wt_gen_, base_gen_MW, base_gen_MVar, base_load_MW, base_load_MVar, base_bus_dict
    else:            
        return  bus_vpu_arr, bus_vangle_arr, load_MW_arr, load_Mvar_arr, gen_MW_arr, gen_Mvar_arr, base_gen_MW_arr, base_gen_Mvar_arr, sum_generation_MW, sum_generation_Mvar, feasible_solution_report
#-------------------------------------- End of COLLECT POWERMODEL.jl DATA FOR ML PIPELINE -------------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#--------------------------------------- Design Input Structure for ML Pipeline -----------------------------------------------------------#

def runModel():
    if training_domain_UB <= base_training_domain_limit:
        tab1.info(f"Using base model. Within base training domian ({base_training_domain_limit:.2f})")
        vpu_model_file = base_model_name_vpu 
        genMW_model_file = gen_model_name_MW
        vpu_feature_file = base_features_vpu
        vpu_target_file = base_targets_vpu
        gen_feature_file = gen_features
        genMW_target_file = gen_targets_MW

        vang_model_file = base_model_name_vang 
        genMvar_model_file = gen_model_name_Mvar
        vang_feature_file = base_features_vang
        vang_target_file = base_targets_vang
        genMvar_target_file = gen_targets_Mvar
    else:
        if os.path.exists(dir_name + retrain_bus_targets_vpu) and training_domain_UB <= training_domain_limit: 
            tab1.info(f"Using updated model. The current training upper limit is {training_domain_limit:.2f}")
            vpu_model_file = updated_model_name_vpu 
            genMW_model_file = updated_gen_model_name_MW
            vpu_feature_file = retrain_bus_features_vpu
            vpu_target_file = retrain_bus_targets_vpu
            gen_feature_file = retrain_gen_features
            genMW_target_file = retrain_gen_targets_MW

            vang_model_file = updated_model_name_vang
            genMvar_model_file = updated_gen_model_name_Mvar
            vang_feature_file = retrain_bus_features_vang
            vang_target_file = retrain_bus_targets_vang
            genMvar_target_file = retrain_gen_targets_Mvar

        elif os.path.exists(dir_name + retrain_bus_targets_vpu) and training_domain_UB > training_domain_limit: 
            tab1.warning(f"Using updated model but above the updated training domain upper limit ({training_domain_limit:.2f})")
            vpu_model_file = updated_model_name_vpu 
            genMW_model_file = updated_gen_model_name_MW
            vpu_feature_file = retrain_bus_features_vpu
            vpu_target_file = retrain_bus_targets_vpu
            gen_feature_file = retrain_gen_features
            genMW_target_file = retrain_gen_targets_MW

            vang_model_file = updated_model_name_vang
            genMvar_model_file = updated_gen_model_name_Mvar
            vang_feature_file = retrain_bus_features_vang
            vang_target_file = retrain_bus_targets_vang
            genMvar_target_file = retrain_gen_targets_Mvar
        else: 
            tab1.warning(f"Using base model but above the base training domain ({base_training_domain_limit:.2f}). Check on-the-🪰 Criteria.")
            vpu_model_file = base_model_name_vpu 
            genMW_model_file = gen_model_name_MW
            vpu_feature_file = base_features_vpu
            vpu_target_file = base_targets_vpu
            gen_feature_file = gen_features
            genMW_target_file = gen_targets_MW

            vang_model_file = base_model_name_vang 
            genMvar_model_file = gen_model_name_Mvar
            vang_feature_file = base_features_vang
            vang_target_file = base_targets_vang
            genMvar_target_file = gen_targets_Mvar


    if os.path.exists(dir_name + vpu_model_file):   

        features = torch.tensor(np.load(dir_name + vpu_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + vpu_target_file)["arr_0"])
            
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.bus_model_vpu = PowerModel_AI(input_size, hidden1_size, hidden2_size, output_size)
        bus_model_vpu = st.session_state.bus_model_vpu
        bus_model_vpu.load_state_dict(torch.load(dir_name + vpu_model_file ))
        bus_model_vpu.eval()
    
    if os.path.exists(dir_name + vang_model_file):   

        features = torch.tensor(np.load(dir_name + vang_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + vang_target_file)["arr_0"])
            
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.bus_model_vang = PowerModel_AI(input_size, hidden1_size, hidden2_size, output_size)
        bus_model_vang = st.session_state.bus_model_vang
        bus_model_vang.load_state_dict(torch.load(dir_name + vang_model_file))
        bus_model_vang.eval()

    if os.path.exists(dir_name + genMW_model_file):   
 
        features = torch.tensor(np.load(dir_name + gen_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + genMW_target_file)["arr_0"])
        
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.gen_model_MW = PowerModel_AI(input_size, hidden1_size, hidden2_size, output_size)
        gen_model_MW = st.session_state.gen_model_MW
        gen_model_MW.load_state_dict(torch.load(dir_name + genMW_model_file))
        gen_model_MW.eval()

    if os.path.exists(dir_name + genMvar_model_file):   

        features = torch.tensor(np.load(dir_name + gen_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + genMvar_target_file)["arr_0"])
        
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.gen_model_Mvar = PowerModel_AI(input_size, hidden1_size, hidden2_size, output_size)
        gen_model_Mvar = st.session_state.gen_model_Mvar
        gen_model_Mvar.load_state_dict(torch.load(dir_name + genMvar_model_file))
        gen_model_Mvar.eval()

    return bus_model_vpu, bus_model_vang, gen_model_MW, gen_model_Mvar


def generateRetrainModels(retrain_json_loc):
    

    # Gen MW Model --------------------------------------------------#
    gen_features_r, gen_targets_MW_r = genFeatureEngineering(retrain_json_loc, "MW") 

    # Save Data for ML processing
    np.savez(dir_name + retrain_gen_features, gen_features_r)
    np.savez(dir_name + retrain_gen_targets_MW, gen_targets_MW_r)

    # Generate Generation Active Power model
    initial_gen_model = [] 
    modelMW_log = st.container()
    with tab1:
        modelMW_log.markdown(":violet[**Training Generation Active Power [MW] Model...**]")
    generateModel(initial_gen_model, gen_features_r, gen_targets_MW_r, "generator MW model")

    # Gen Mvar Model --------------------------------------------------#
    _, gen_targets_Mvar_r = genFeatureEngineering(retrain_json_loc, "Mvar") 

    # Save Data for ML processing
    np.savez(dir_name + retrain_gen_targets_Mvar, gen_targets_Mvar_r)

    # Generate Generation Reactive Power model
    initial_gen_model = []
    modelMvar_log = st.container()
    with tab1:
        modelMvar_log.markdown(":blue[**Training Generation Rective Power [Mvar] Model...**]")
    generateModel(initial_gen_model, gen_features_r, gen_targets_Mvar_r, "generator Mvar model")  
    #*********************************************************************************************#  

    ########################## Bus Model ###############################################

    #-------------------- Bus VPU Model --------------------------------------------------#
    bus_vpu_features_r, bus_vpu_targets_r = busFeatureEngineering(retrain_json_loc, "vpu") 

    # Save Data for ML processing
    np.savez(dir_name + retrain_bus_features_vpu, bus_vpu_features_r)
    np.savez(dir_name + retrain_bus_targets_vpu, bus_vpu_targets_r)

    # Generate Bus Active Power Demand Model
    initial_bus_model = []
    modelvpu_log = st.container()
    with tab1:
        modelvpu_log.markdown(":green[**Training Voltage Maginitude [vpu] Model...**]")
    generateModel(initial_bus_model, bus_vpu_features_r, bus_vpu_targets_r, "bus vpu model")

    #-------------------- Bus VANGLE Model --------------------------------------------------#
    bus_vang_features_r, bus_vang_targets_r = busFeatureEngineering(retrain_json_loc, "vangle") 

    # Save Data for ML processing
    np.savez(dir_name + retrain_bus_features_vang, bus_vang_features_r)
    np.savez(dir_name + retrain_bus_targets_vang, bus_vang_targets_r)

    # Generate Bus Reactive Power Demand Model
    initial_bus_model = []
    modelvang_log = st.container()
    with tab1:
        modelvang_log.markdown(":orange[**Training Voltage Angle [vangle] Model...**]")
    generateModel(initial_bus_model, bus_vang_features_r, bus_vang_targets_r, "bus vangle model")

#------------------------------------------------- End of Design Input Structure for ML Pipeline ------------------------------------------#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#------------------------------------------------- F E A T U R E    E N G I N E E R I N G -------------------------------------------------#

def on_the_fly_update(retrain_json_loc):
    dir_path = os.path.join(os.getcwd(), retrain_json_loc)

    _, _, _, _, _, _, _, _, _, _, feasible_solution_report = collectGridData(dir_path, base_case = 0)

    return feasible_solution_report


def genFeatureEngineering(retrain_json_loc, gen_power_type):

    dir_path = os.path.join(os.getcwd(), retrain_json_loc)

    _, _, load_MW_arr, load_Mvar_arr, gen_MW_arr, gen_Mvar_arr, base_gen_MW_arr,\
         base_gen_Mvar_arr, sum_generation_MW, sum_generation_Mvar, _ = collectGridData(dir_path, base_case = 0)

    
    features = torch.cat((torch.tensor(load_MW_arr), torch.tensor(load_Mvar_arr),\
                            torch.tensor(base_gen_MW_arr), torch.tensor(base_gen_Mvar_arr)),dim =1) #
    print(f"The size of Generator Feature for {st.session_state['samples_for_PMJL']} changes is {features.shape}")
    

    if gen_power_type == "MW":
        targets = torch.tensor(gen_MW_arr)
        #tab1.write(f"Sigma_P_max = {np.max(sum_generation_MW)}    Sigma_P_min = {np.min(sum_generation_MW)}")
        print(f'The size of Generator MW Targets for {st.session_state["samples_for_PMJL"]} changes is {targets.shape}')


    if gen_power_type == "Mvar":
        targets = torch.tensor(gen_Mvar_arr)
        #tab1.write(f"Sigma_Q_max = {np.round(np.max(sum_generation_Mvar),4)}    Sigma_Q_min = {np.round(np.min(sum_generation_Mvar),4)}")
        print(f'The size of Generator Mvar Targets for {st.session_state["samples_for_PMJL"]} changes is {targets.shape}')

        power_limit_dict_training = {"Sigma_P_max":np.round(np.max(sum_generation_MW),4), "Sigma_P_min":np.round(np.min(sum_generation_MW),4),
                                  "Sigma_Q_max":np.round(np.max(sum_generation_Mvar),4), "Sigma_Q_min":np.round(np.min(sum_generation_Mvar),4)}
        
        writeGenPowerLimit(retrain_folder+ "\\" + selected_dirname + gen_power_limit_file, power_limit_dict_training)
        col_confirm.write(power_limit_dict_training)

    return features, targets

def busFeatureEngineering(retrain_json_loc, voltage_type):
    dir_path = os.path.join(os.getcwd(), retrain_json_loc)

    bus_vpu_arr, bus_vangle_arr, load_MW_arr, load_Mvar_arr, gen_MW_arr, gen_Mvar_arr,_, _, _, _, _ = collectGridData(dir_path, base_case = 0)

    num_buses_wt_load = len(buses_wt_load) # Number of nodes with loads
    num_changes = len(bus_vpu_arr)     # Number of features (training data)

    if voltage_type == "vpu":

        features = torch.cat((torch.tensor(load_MW_arr), torch.tensor(load_Mvar_arr),torch.tensor(gen_Mvar_arr)),dim =1)       
        print(f'The size of VPU FEATURES for {st.session_state["samples_for_PMJL"]} changes to {num_buses_wt_load} nodes with load is {features.shape}')

        targets = torch.tensor(bus_vpu_arr) 
        print(f'The size of VPU OUTPUTS for {st.session_state["samples_for_PMJL"]} changes to {num_buses_wt_load} nodes with load is {targets.shape}')

    if voltage_type == "vangle":

        features = torch.cat((torch.tensor(load_MW_arr), torch.tensor(load_Mvar_arr),torch.tensor(gen_MW_arr)),dim =1)  
        print(f'The size of Vangle FEATURES for {st.session_state["samples_for_PMJL"]} changes to {num_buses_wt_load} nodes with load is {features.shape}')     
        
        targets = torch.tensor(bus_vangle_arr) 
        print(f'The size of Vangle OUTPUTS for {st.session_state["samples_for_PMJL"]} changes to {num_buses_wt_load} nodes with load is {targets.shape}')

    return features, targets

###################################### E N D OF F E A T U R E  E N G I N E E R I N G #######################################################
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
########################################### ML PIPLINE FUNCTIONS ###########################################################################

# Define parameters
def defineParameters(feature, target):
    input_size = feature.shape[-1]
    output_size = target.shape[-1]
    hidden1_size = int(input_size /3) 
    hidden2_size = int(hidden1_size/2)
    return input_size, hidden1_size, hidden2_size, output_size

# Define the neural network model with two hidden layers for the voltages
class PowerModel_AI(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(PowerModel_AI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.bias = nn.Parameter(torch.zeros(output_size))  # Adding bias
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out) + self.bias  # Adding bias
        return out

def trainValidateModel(model, num_epochs, train_loader, val_loader, optimizer, criterion, device):

    training_log = st.empty()

    # Train the model
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()    # zero the gradient buffers
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()  # update the weight
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}", end="\r")
        with tab1:
            training_log.write(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {np.format_float_scientific(epoch_loss, precision=2, trim ='.')}, Val. Loss: {np.format_float_scientific(val_loss, precision=2, trim ='.')}")
    print()
    return model, train_losses, val_losses
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#

#-------------------------------------------------- On-the-fly Algorithm ------------------------------------------------------------------#
def generateModel(model, features, targets, model_type):

    # Define parameters
    input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)
    model = PowerModel_AI(input_size, hidden1_size, hidden2_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate data
    num_samples = features.shape[0]
    num_train_samples = int(num_samples * 0.7)  # 70%
    num_test_samples = int(num_samples * 0.15)   # 15%

    train_features = features[:num_train_samples].to(torch.float32)
    train_targets = targets[:num_train_samples].to(torch.float32)
    test_features = features[num_train_samples:num_train_samples+num_test_samples].to(torch.float32)
    test_targets = targets[num_train_samples:num_train_samples+num_test_samples].to(torch.float32)
    val_features = features[num_train_samples+num_test_samples:].to(torch.float32)
    val_targets = targets[num_train_samples+num_test_samples:].to(torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_dataset = TensorDataset(val_features, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model with base data
    num_epochs = 300
    model, train_losses, val_losses = trainValidateModel(model, num_epochs, train_loader, val_loader, optimizer, criterion, device='cpu')

    # Test the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print()
    print(f"Test Loss: {test_loss:.6f}")
    print()

    # Save the final weights to reuse when needed.
    if model_type == "bus vpu model":
        print("*********** CREATING UPDATED VOLTAGE MAGNITUDE MODEL **************************")
        torch.save(model.state_dict(), dir_name + updated_model_name_vpu )
        print(f"Updated VPU model saved as {updated_model_name_vpu }")
        print("*******************************************************************************")
        print()
        col_confirm.write("")
        col_confirm.markdown("###### Voltage Model Architecture: ")
        col_confirm.write(model)

    elif model_type == "bus vangle model":
        print("*********** CREATING UPDATED VOLTAGE ANGLE MODEL **************************")
        torch.save(model.state_dict(), dir_name + updated_model_name_vang )
        print(f"Updated Vangle model saved as {updated_model_name_vang }")
        print("***************************************************************************")
        print()

    elif model_type == "generator MW model":
        print("*********** CREATING UPDATED GENERATOR MW MODEL **************************")
        torch.save(model.state_dict(), dir_name + updated_gen_model_name_MW)
        print(f"Updated Gen. MW model saved as {updated_gen_model_name_MW}")
        print("*******************************************************************************")
        print()
        col_confirm.write("")
        col_confirm.markdown("###### Generation Power Model Architecture: ")
        col_confirm.write(model)

    elif model_type == "generator Mvar model":
        print("*********** CREATING UPDATED GENERATOR Mvar MODEL **************************")
        torch.save(model.state_dict(), dir_name + updated_gen_model_name_Mvar)
        print(f"Updated Gen. Mvar model saved as {updated_gen_model_name_Mvar}")
        print("*******************************************************************************")
        print()

# Function for prediction on unseen power demand changes
def predict(model, new_features_,new_targets_, criterion = nn.MSELoss(), on_the_fly = 0):
    
    # Convert data to appropriate data types
    new_features = new_features_.to(torch.float32)
    new_targets = new_targets_.to(torch.float32)
    
    # Create a data loader for the new data
    new_dataset = TensorDataset(new_features, new_targets)
    new_loader = DataLoader(new_dataset, batch_size=len(new_targets_), shuffle=False)
    
    # Test the model on the new data
    model.eval()
    new_test_loss = 0.0
    all_output = np.zeros_like(new_targets)
    with torch.no_grad():
        for inputs, targets in new_loader:
            outputs = model(inputs)
            new_test_loss += criterion(outputs, targets).item() * inputs.size(0)
            #all_output = outputs.numpy()
    new_test_loss /= len(new_loader.dataset)

    if on_the_fly == 1:
        print(f"New Test Loss: {new_test_loss}")

    return outputs, new_test_loss

############################### END OF ML PIPLINE FUNCTIONS ################################################################################
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
####################################### S T R E A M L I T ##################################################################################


if selected_dirname:
    tab1, tab2, tab3 = st.tabs(["Activity", f"{selected_dirname[3:]} Grid", 'Model Information'])
    tab1.markdown("###### Recent Activity")
    tab2.markdown("###### Grid Information")
    tab3.markdown("###### ML Model Architecture")
    tab3.info(f" Current Training Domain upper limit is: {training_domain_UB:.2f}")


    st.session_state.buses, st.session_state.buses_wt_load, st.session_state.buses_wt_gen,\
        st.session_state.base_gen_MW, st.session_state.base_gen_Mvar, st.session_state.base_load_MW,\
              st.session_state.base_load_Mvar, st.session_state.base_bus_dict = collectGridData(dir_name, base_case=1)
    
    buses = st.session_state.buses; buses_wt_load = st.session_state.buses_wt_load; buses_wt_gen = st.session_state.buses_wt_gen
    base_gen_MW = st.session_state.base_gen_MW; base_gen_Mvar = st.session_state.base_gen_Mvar
    base_load_MW = st.session_state.base_load_MW; base_load_Mvar = st.session_state.base_load_Mvar
    base_bus_dict = st.session_state.base_bus_dict

    st.session_state.log2 = tab2.text_area("# Grid Information", f"Nodes with Load: {str(buses_wt_load)[1:-1]}\n\n\
Nodes with Generators: {str(buses_wt_gen)[1:-1]}", label_visibility="hidden")
    
    ################  P L O T    M A P ###################################################################################
    with tab2:
        col1_p1, col2_p2 = st.columns(2)

        path = dir_name
        buses =  pd.read_csv(os.path.join(path, 'bus.csv'), dtype={'BusNum' : str, 'BusKVVolt' : np.float64, 'BusName' : str, 'Latitude:1' : np.float64, 'Longitude:1' : np.float64})
        branches =  pd.read_csv(os.path.join(path, 'branch.csv'), dtype={'BusNum' : str, 'BusNum:1' : str, 'LineCircuit' : str, 
                                                                    'LineR' : np.float64, 'LineX' : np.float64, 'BranchDeviceType' : str, 
                                                                    'LineStatus' : str, 'Latitude' : np.float64, 'Longitude' : np.float64,
                                                                    'Latitude:1' : np.float64, 'Longitude:1' : np.float64, 'BusNomVolt' : np.float64}
        ) 

         
        loads = pd.read_csv(os.path.join(path, 'load.csv'), dtype={'BusNum' : str, 'LoadID' : str, 'LoadMW' : np.float64, 'LoadMVR' : np.float64, 
                                                                        'LoadStatus' : str, 'Latitude:1' : np.float64, 'Longitude:1' : np.float64,
                                                                        'BusNomVolt' : np.float64}
        )
        
        ############## Base VPU ################################################################################
        vpu_base = {}
        buses["vpu"] = 0.0
        buses["vangle"] = 0.0
        buses["size"] = 10
        for i, bus in base_bus_dict.items():
            vpu_base[str(i)] = bus['vm']
        st.session_state.buses_basevpu_df = buses.apply(update_voltage, axis=1, args=(vpu_base,))
        plot_container = st.container(border=True)
        plot_map(st.session_state, col1_p1, st.session_state.buses_basevpu_df, branches, "Initial Voltage Magnitude", result_type="vpu")

        ############## Base VANGLE ################################################################################
        vangle_base = {}
        for i, bus in base_bus_dict.items():
            vangle_base[str(i)] = bus['va']
        st.session_state.buses_basevangle_df = buses.apply(update_voltage, axis=1, args=(vangle_base,))
        plot_container = st.container(border=True)
        plot_map(st.session_state, col1_p1, st.session_state.buses_basevangle_df, branches, "Initial Voltage Angle", result_type="vangle")
     ################ E N D: P L O T   M A P ###################################################################################
else:
    tab1, tab2, tab3 = st.tabs(["Activity", f"Grid", 'Model Information'])
    tab1.write_stream(stream_data)
    tab1.image(lanl_logo, width=210)

st.sidebar.write('###### Working Directory: `%s`' %selected_dirname)

load_changes = st.sidebar.radio('Node Selection Process:', options=[1, 2,3], help="select the procedure for specifying the node(s) and power demand multiplier(s)",
                                format_func=lambda x: ["Random Nodes", "Specify Nodes", "User-Defined"][x-1], disabled=st.session_state.get("disableCaseSelect", True), on_change=cleanSlate,  horizontal=1)
###########################################################################################################################################################################################################



###################### LOAD BASE ML MODELS #############################################
if selected_dirname:                                                                   #
    bus_model_vpu, bus_model_vang, gen_model_MW, gen_model_Mvar = runModel()           #
################## END OF  LOAD BASE ML MODELS #########################################

## Collect grid information from base case (default system)  
if selected_dirname:
    st.sidebar.number_input("Number of Nodes to Modify:", min_value=1, max_value= len(buses_wt_load), step = 1, key="num_changes",on_change=retainBuses, help=f"Select the number of nodes in the {num_buses}-bus case you wish to modify their power demand")
    num_changes = st.session_state.num_changes

with tab1:

    # Random Node Seleciton Without Repetition
    if "num_changes" in st.session_state and load_changes == 1:
       
        #*** Debug ***#
        print("D: At Random Nodes Button")
        print(f"D: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain}")
        #*** Debug ***#

        random_ints = ran.sample(range(0, len(buses_wt_load)), num_changes) # smampel withouto replacement
        random_buses = np.array(buses_wt_load)[random_ints]
        random_changes = np.random.rand(num_changes) * training_domain
        buses_to_change_load_list = []
        buses_to_change_load_list.append(random_buses)
        
        st.write(f"{num_changes} Randomly Selected Bus(es): {str(random_buses)[1:-1]}")
        change_list.append(random_changes)
        df = pd.DataFrame(change_list, columns=buses_to_change_load_list)
        st.markdown("*Random Changes for Selected Bus(es)*")
        load_change_df = st.data_editor(df, key="load_df")

        tab2.markdown("*Random Changes for Selected Bus(es)*")
        tab2.data_editor(df)

        col_sb1, col_sb2 = st.sidebar.columns(2, gap = "small")
    
    # Custom Node Selecition and User-Defined Changes to Power demand Multiplier Values
    if "num_changes" in st.session_state and (load_changes == 2 or load_changes == 3):

        change_list = []
        buses_to_change_load_list = []
        buses_to_change_load =st.multiselect("Select Node(s) to Modify Power Demand:", buses_wt_load, max_selections=num_changes, key="load_changes", on_change=retainBuses, help ="Select the nodes you wish to update their power demand")
        buses_to_change_load_list.append(np.array(buses_to_change_load))

        if load_changes == 3 and len(np.array(buses_to_change_load)) > 0:
            st.sidebar.number_input("Increase Multiplier Upper Limit:", min_value=training_domain_LB, value = training_domain_UB, step = 0.1, key="training_domain_UB", help=f"Adjust the power demand multiplier upper limit. This may require on-the-🪰 model updating.", on_change=disableRetrain, args=(False,))

        if len(np.array(buses_to_change_load)) > 0:
            col_sb1, col_sb2 = st.sidebar.columns(2, gap = "small")
            st.session_state.getButton = 1      

        st.write(f"Selected Node(s): {str(buses_to_change_load)[1:-1]}")

        if load_changes == 2:

            #*** Debug ***#
            print("E: At Custom Nodes Radio Button")
            print(f"E: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain}")
            #*** Debug ***#

            if len(np.array(buses_to_change_load)) > 0:

                st.sidebar.number_input("Modify Change Rate (default is 2.5):", min_value=training_domain_LB, max_value= training_domain_UB, value = training_domain, step = 0.01, on_change=getUpperBound, key="training_domain", help=f"Adjust the range of Power Demand Multiplier from 0.01 to {training_domain_UB}")
                change_list.append(np.random.rand(len(buses_to_change_load))*training_domain)

                #*** Debug ***#
                print("F: At Custom Nodes Radio Button With Key as training_domain")
                print(f"F: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain}")
                #*** Debug ***#

                df = pd.DataFrame(change_list, columns= np.array(buses_to_change_load))
                st.markdown(f"*Random Power Demand Multiplier at a rate of {training_domain:.2f}*")
                load_change_df = st.data_editor(df, key="load_df")
                col_sb1, col_sb2 = st.sidebar.columns(2, gap = "medium")

                tab2.markdown(f"*Random Power Demand Multiplier at a rate of {training_domain:.2f}*")
                tab2.data_editor(df)          
            
        if load_changes == 3 and len(np.array(buses_to_change_load)) > 0:

            #*** Debug ***#
            print("G: At User-Defined Radio Button")
            print(f"G: {selected_dirname}, {base_training_domain_limit}, {training_domain_UB}, {training_domain}")
            #*** Debug ***#

            change_text = st.text_input(f"Enter Power Demand Multiplier for {num_changes} Buses. Separate each input with a space.")
            change_text_split = change_text.split()
            change_floats = []

            if training_domain_UB and training_domain_UB > training_domain_limit:
                col_sb2.toggle("On-the-🪰", key="on_the_fly", on_change=disableRetrain, args=(False,))

            if len(np.array(buses_to_change_load)) != len(change_text_split):
                change_list = []
                st.warning(f'You have {len(np.array(buses_to_change_load))} nodes yet specified {len(change_text_split)} changes.', icon="⚠️")
                change_list.append(np.random.rand(len(buses_to_change_load))*np.nan)
            else:
                change_list = []
                pos = 0
                
                for i in change_text_split:
                    changeRatio = float(i)
                    if changeRatio > round(training_domain_UB,2):
                        print(f"The training Upper Bound is: {training_domain_UB}")
                        create_sidebar = 1
                        st.warning(f'Do you know what you are doing?\
                                   \n You selected an out-of-bound power demand mulitplier {changeRatio:.2f} for Node {buses_to_change_load[pos]}.\
                                    \n Max. limit is: {training_domain_UB:.2f}\
                                   \n Default power demand multiplier of {training_domain} has been set for Node {buses_to_change_load[pos]} instead.', icon="⚠️")
                        changeRatio = training_domain
                        change_floats.append(changeRatio)

                    elif changeRatio < training_domain_LB:
                        st.warning(f'Do you know what you are doing?\
                                   \n You selected an out-of-bound power demand mulitplier {changeRatio:.2f} for Node {buses_to_change_load[pos]}.\
                                    \n Min. limit is: {training_domain_LB:.2f}\
                                   \n The lowest power demand multiplier of {training_domain_LB} has been set for Node {buses_to_change_load[pos]} instead.', icon="⚠️")
                        changeRatio = training_domain_LB
                        change_floats.append(changeRatio)
                    else:
                        change_floats.append(changeRatio)
                    pos +=1
                change_list.append(np.array(change_floats))
            
            df = pd.DataFrame(data = change_list, columns =np.array(buses_to_change_load))
            load_change_df = st.data_editor(df, key="load_df")

            tab2.markdown(f"*User-defined Power Demand Multiplier*")
            tab2.data_editor(df)
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#---------------------------------------------- PREDICTION FOR GEN POWER AND VOLTAGE ------------------------------------------------------#

if len(change_list) > 0 and (load_changes == 1 or st.session_state.getButton == 1):
    
    col_sb1.button("Predict", key="predict_button")
    predict_button = st.session_state.predict_button

    with tab1:
        col1_t1, col2_t1 = st.columns(2, gap = "medium")
        f_MW = {}
        f_MW_list = []
        f_MVar = {}
        f_MVar_list = []

        base_gen_MW_arr = np.zeros(num_buses)
        base_gen_MW_mask = np.zeros(num_buses)
        base_gen_MW_arr[buses_wt_gen-1] = base_gen_MW
        base_gen_MW_mask[buses_wt_gen-1] = np.array(base_gen_MW, dtype='bool')

        base_gen_Mvar_arr = np.zeros(num_buses)
        base_gen_Mvar_mask = np.zeros(num_buses)
        base_gen_Mvar_arr[buses_wt_gen-1] = base_gen_Mvar
        base_gen_Mvar_mask[buses_wt_gen-1] = np.array(base_gen_Mvar, dtype='bool')
    
    with tab3:
        col1_t3, col2_t3 = st.columns(2, gap = "large")

    if predict_button:
        if training_domain_UB > training_domain_limit:
            tab1.warning(f"First on-the-🪰 criteria failed: Training upper bound ({training_domain_limit:.2f}) is less than current upper limit ({training_domain_UB:.2f})", icon="⚠️")

        if True:
            ## Create Power demand Multipliers Dictionaries
            buses_wt_load_MWdict = {}
            buses_wt_load_MVardict = {}
            buses_to_change_load_dict = {}
        
            for bus_ind, bus in enumerate(buses_wt_load):
                buses_wt_load_MWdict[str(bus)] = base_load_MW[bus_ind]
                buses_wt_load_MVardict[str(bus)] = base_load_Mvar[bus_ind]
            
            for bus_ind,bus in enumerate(buses_to_change_load_list[0]):
                buses_to_change_load_dict[str(bus)] = change_list[0][bus_ind]

            for bus_ind, bus_int  in enumerate(buses_wt_load):
                bus = str(bus_int)
                if bus in buses_to_change_load_dict.keys():
                    f_MW[bus] = buses_wt_load_MWdict[bus] * buses_to_change_load_dict[bus]
                    f_MVar[bus] = buses_wt_load_MVardict[bus] * buses_to_change_load_dict[bus]
                    f_MW_list.append(buses_wt_load_MWdict[bus] * buses_to_change_load_dict[bus])
                    f_MVar_list.append(buses_wt_load_MVardict[bus] * buses_to_change_load_dict[bus])
                else:
                    f_MW[bus] = base_load_MW[bus_ind]
                    f_MVar[bus] = base_load_Mvar[bus_ind]
                    f_MW_list.append(base_load_MW[bus_ind])
                    f_MVar_list.append(base_load_Mvar[bus_ind])

            ###################### PREDICT GENERATOR MW #######################################################################################
            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.tensor(base_gen_MW_arr), torch.tensor(base_gen_Mvar_arr))) 
            f_gen_MW = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_MW_arr)
            t_gen_MW = torch.reshape(t, (1,-1))

            print("******************************************************************************************************************************")
            print("Generator Model Details:")
            print(f"feature shape = {np.shape(f_gen_MW)}       target shape = {np.shape(t_gen_MW)}\n")
    

            gen_MW_pred, test_losses = predict(gen_model_MW, f_gen_MW, t_gen_MW, criterion = nn.MSELoss(), on_the_fly = 0)
            gen_MW_masked = gen_MW_pred*base_gen_MW_mask

            tab1.write(f"Predicted Gen Active Power [MW]. Sum of Gen. Active Power = {np.round(np.sum(gen_MW_masked.numpy()),4)} MW")
    
            tab1.dataframe(pd.DataFrame(gen_MW_masked, index =["MW"], columns=("B %d" % (i+1) for i in range(len(gen_MW_masked[0])))))

            col1_t3.write(f"Input for Gen. Active Power Prediction:       {f_gen_MW.shape}")
            col1_t3.write(f_gen_MW)

            col1_t3.write(f"Predicted Gen Active Power [MW]: {gen_MW_masked.shape}")
            
            col1_t3.write(gen_MW_masked)
            #***********************************************************************************************************************************#

            ###################### PREDICT GENERATOR MVAR #######################################################################################
            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.tensor(base_gen_MW_arr), torch.tensor(base_gen_Mvar_arr))) 
            f_gen_Mvar = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_Mvar_arr)
            t_gen_Mvar = torch.reshape(t, (1,-1))

            gen_Mvar_pred, test_losses = predict(gen_model_Mvar, f_gen_Mvar, t_gen_Mvar, criterion = nn.MSELoss(), on_the_fly = 0)
            gen_Mvar_masked = gen_Mvar_pred*base_gen_Mvar_mask

            tab1.write(f"Predicted Gen Reactive Power [Mvar]. Sum of Gen. Reactive Power = {np.round(np.sum(gen_Mvar_masked.numpy()),4)} Mvar")
    
            tab1.dataframe(pd.DataFrame(gen_Mvar_masked, index =["Mvar"], columns=("B %d" % (i+1) for i in range(len(gen_Mvar_masked[0])))))

            col1_t3.write(f"Input for Gen. Reactive Power Prediction:      {f_gen_Mvar.shape}")
            col1_t3.write(f_gen_Mvar)

            col1_t3.write(f"Predicted Gen Reactive Power [Mvar]: {gen_Mvar_masked.shape}")
            
            col1_t3.write(gen_Mvar_masked)
            #************************************************************************************************************************************#

            ## Checking Generator Power Saturation Criteria for on-the-fly #######################################################################
            P_sum = np.sum(gen_MW_masked.numpy())
            Q_sum = np.sum(gen_Mvar_masked.numpy())
            print("Checking Criteria for on-the-fly Implementation:")
            print(f"P_sum [MW]= {P_sum:.3f}   and   Q_sum [Mvar] = {Q_sum:.3f}\n")
            power_limit_var, power_limit_val = readGenPowerLimit(retrain_folder+ "\\" + selected_dirname + gen_power_limit_file)
            power_limit_dict = dict(zip(power_limit_var, power_limit_val))
            tab2.dataframe(power_limit_dict)

            if not power_limit_dict["Sigma_P_min"] < P_sum <  power_limit_dict["Sigma_P_max"]:
                tab1.error(f"Third on-the-🪰 criteria failed: Predicted generation active power, at {P_sum:.4f} MW, is out-of-bounds (see power limits in Tab 2). Implement on-the-🪰", icon="🚨")

            if not power_limit_dict["Sigma_Q_min"] < Q_sum <  power_limit_dict["Sigma_Q_max"]:
                tab1.error(f"Third on-the-🪰 criteria failed: Predicted generation reactive power, at {Q_sum:.4f} Mvar, is out-of-bounds (see power limits in Tab 2). Implement on-the-🪰", icon="🚨")

            ###################### Predict Volatage Magnitude (VPU) ###############################################################################
            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.reshape(gen_Mvar_masked, (-1,)))) 
            f_bus_vpu = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_Mvar_arr)
            t_bus_vpu = torch.reshape(t, (1,-1))

            print("Voltage Model Details:")
            print(f"feature shape = {np.shape(f_bus_vpu)}       target shape = {np.shape(t_bus_vpu)}\n")

            bus_VPU_pred, test_losses = predict(bus_model_vpu, f_bus_vpu, t_bus_vpu, criterion = nn.MSELoss(), on_the_fly = 0)

            bus_VPU_pred_np = np.round(bus_VPU_pred.numpy(),4)
    
            tab1.write("Predicted Voltage Magnitude [vpu] (See ***Model Information*** tab for more.)")
            tab1.dataframe(pd.DataFrame(bus_VPU_pred_np, index =["VPU"], columns=("B %d" % (i+1) for i in range(len(bus_VPU_pred[0])))))

            col2_t3.write(f"Input for Voltage Magnitude Prediction:      {f_bus_vpu.shape}")
            col2_t3.write(f_bus_vpu)

            col2_t3.write(f"Predicted Voltage Magnitude [vpu]: {bus_VPU_pred.shape}")
            col2_t3.write(bus_VPU_pred)

            ###################### Predict Volatage Angle (VANGLE) ###############################################################################
            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.reshape(gen_MW_masked, (-1,)))) 
            f_bus_vang = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_MW_arr)
            t_bus_vang = torch.reshape(t, (1,-1))
            bus_vang_pred, test_losses = predict(bus_model_vang, f_bus_vang, t_bus_vang, criterion = nn.MSELoss(), on_the_fly = 0)

            bus_vang_pred_np = np.round(bus_vang_pred.numpy(),4)
    
            tab1.write("Predicted Voltage Angle [vangle] (See ***Model Information*** tab for more.)")
            tab1.dataframe(pd.DataFrame(bus_vang_pred_np, index =["Vangle"], columns=("B %d" % (i+1) for i in range(len(bus_vang_pred[0])))))

            col2_t3.write(f"Input for Voltage Angle Prediction:      {f_bus_vang.shape}")
            col2_t3.write(f_bus_vang)

            col2_t3.write(f"Predicted Voltage Angle [vangle]: {bus_vang_pred.shape}")
            col2_t3.write(bus_vang_pred)

        ##################### Checking Power/Energy Conservation Criteria for on-the-🪰 implementation #######################################
        # To Be Added Later (Future Plan)
        #---------------------------------- SUM of POWER PER BUS ----------------------------------------------------------------------------#


        ########################## PLOT Predicted MAP ################################################################
        with tab2:
            ############### VPU PLOT ####################################
            vpu = {}
            pred_VPU_numpy = bus_VPU_pred[0].numpy()
            #st.write(len(pred_VPU_numpy))
            for i, _vpu in enumerate(pred_VPU_numpy): #vpu_ in enumerate(bus_VPU_pred[0].numpy()):
                vpu[str(i+1)] = _vpu
            
            st.session_state.buses_vpu_df_pred = buses.apply(update_voltage, axis=1, args=(vpu,))
            plot_map(st.session_state, col2_p2, st.session_state.buses_vpu_df_pred, branches,  "Predicted Voltage Magnitude", result_type="vpu")

            ############### VANGLE PLOT ####################################
            vangle = {}
            pred_vang_numpy = bus_vang_pred[0].numpy()
            for i, _vangle in enumerate(pred_vang_numpy):
                vangle[str(i+1)] = _vangle
            
            st.session_state.buses_vang_df_pred = buses.apply(update_voltage, axis=1, args=(vangle,))
            plot_map(st.session_state, col2_p2, st.session_state.buses_vang_df_pred, branches, "Predicted Voltage Angle", result_type="vangle")
        ########################## PLOT Predicted MAP ################################################################
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
################################# ON THE FLY IMPLEMENTATION ################################################################################

## Generate new power flow data in region of interest using PowerModels.jl
with tab1:
    if "on_the_fly" in st.session_state and st.session_state.on_the_fly:
        col_confirm, col_retrain = st.columns(2, gap = "small")
        with col_confirm:
            with st.popover("Run on-the-🪰?", help="If Yes, a new dataset for the training domain will be created using PowerModels.jl. Note that this may take a while", disabled=False, use_container_width=False):
                with st.container(border=True):
                    no_col, yes_col = st.columns(2)
                    no_button_ = no_col.button("No", key = "no_button", on_click=disableRetrain, args=(True,), disabled=st.session_state.get("disableRetrain",True))
                    yes_button_ = yes_col.button("Yes", key = "yes_button", on_click=disableRetrain, args=(True,), disabled=st.session_state.get("disableRetrain",True))
                    if yes_button_:
                        col_samples, col_ok = st.columns(2, gap = "small")
                        col_samples.number_input("No. of Samples: ", min_value = 1, max_value=20000, value = 10, step = None, on_change=runPowerModelJL, key = "samples_for_PMJL")     
                        col_ok.button("Continue",key="ok_button", on_click=runPowerModelJL)   
                                            
        with col_retrain:
            if "ok_button" in st.session_state and st.session_state.run == True:

                if True: 
                    julia_script_path = './RunJuliaScript.py'  

                    writeTrainingDomain(retrain_folder+ "\\" + selected_dirname + domain_file, str(np.round(training_domain_UB,2)), str(np.round(base_training_domain_limit,2)))

                    with st.status("Updating PowerModel-AI On-the-🪰...", expanded=False):
                        st.write("Julia script for PowerModels.jl implementation created")
                        loc = retrain_folder + '\\' + selected_dirname
                        for file in os.listdir(loc):
                            if '.m' in file:
                                test_case_m = loc + '\\' + file

                                jl_file = loc + "\juliaScript.jl"
                                writeJuliaPath(julia_params, jl_file)

                                runJulia = mld(jl_file, test_case_m)
                                runJulia.samples = st.session_state["samples_for_PMJL"]
                                runJulia.LB = 0.01
                                runJulia.UB = training_domain_UB
                                runJulia.jsonName = test_case_m[:-2]  + ".json"
                                runJulia.writeJLHeader()
                                runJulia.writeJLParse()
                                runJulia.writeJLGetBaseInfo()
                                runJulia.writeJLLoadChangePF()
                                runJulia.writeJLJSON()
                                runJulia.io.close()
                                st.write("Generating training data via PowerModels.jl")
                                os.system("python " + julia_script_path)

                                st.markdown(":gray-background[***PowerModel-AI Model Updating...***]")
                                st.write(on_the_fly_update(retrain_json_loc = loc))
                            
                                generateRetrainModels(retrain_json_loc = loc)
                                
                    st.success("On-the-🪰 Model Update Complete!")
            
            if no_button_:
                st.info("Continue with prediction. Make sure you are within the training domain")
                    

st.session_state.run = False
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#---------------------------------------------- Visualize ML Models in Tab 3  -------------------------------------------------------------#
if selected_dirname:
    if os.path.exists(dir_name + base_model_name_vpu ):   
        ## Load Existing data
        
        tab3.write("")
        tab3.markdown("###### Voltage Model Architecture: ")
        tab3.write(bus_model_vpu)
    else:
        tab3.error("Please add a .pth file for the ML base model. See Documentation")

    
    if os.path.exists(dir_name + gen_model_name_MW):   
       
        tab3.write("")
        tab3.markdown("###### Generation Power Model Architecture: ")
        tab3.write(gen_model_MW)
    else:
        tab3.error("Please add a .pth file for the ML Generator VPU model. See Documentation")

st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")

st.sidebar.markdown("********")
st.sidebar.image(lanl_logo, width=210)
print()
print("############################################ E N D  O F  T H E  A L G O R I T M ############################################################")
print()
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
#******************************************************************************************************************************************#
############################################ E N D  O F  T H E  A L G O R I T M ############################################################
