#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ugwumadu Chinonso, Jose tabares, and Anup Pandey"
__credits__ = ["Ugwumadu Chinonso, Jose tabares, and Anup Pandey"]
__version__ = "1.0"
__email__ = "cugwumadu@lanl.gov"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import julia

from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import random as ran
import numpy as np
import pandas as pd
import time

############################ INITIALIZATIN FUNCTIONS ###############################################################
st.set_page_config(
    page_title="PoWerML",
    page_icon="⚡",
    layout="wide"
)

#----------------- Parameters ---------------------------------#
logo = ".\etc\Setup Files\LOGO.png"
domain_file = ".\etc\Setup Files\Training Domain.txt"
julia_params = ".\etc\Setup Files\Julia Parameters.txt"
retrain_folder = ".\etc\on-the-fly"


def readTrainingDomain(txt_path):
    with open(txt_path, "r") as f:
       upper_domain_st = f.read()
    return float(upper_domain_st)

def writeTrainingDomain(txt_path, new_domain):
    with open(txt_path, "w") as o:
       o.write(new_domain)

def writeJuliaPath(txt_path, julia_path):
    with open(txt_path, "w") as o:
       o.write(julia_path)

#*************************************************************************** S T A R T ***************************************************************************#
#st.set_page_config(page_title="Carbon Nanotube Initializer", page_icon="https://chinonsougwumadu.com/wp-content/uploads/2024/05/microsoftteams-image-17.jpg")
st.markdown("## PowerModel AI")

st.sidebar.image(logo, width=120)
st.session_state.getButton = 0

st.session_state.training_domain_limit = readTrainingDomain(domain_file)
training_domain_limit = st.session_state.training_domain_limit

######################### Initializations (including helper functions) #########################
if "training_domain" not in  st.session_state:
    st.session_state.training_domain = 2.5 # the max to which we have trained the model
    training_domain = st.session_state.training_domain


    st.session_state.training_domain_UB = training_domain_limit
    training_domain_UB = st.session_state.training_domain_UB

    st.session_state.training_domain_LB = 0.1
    training_domain_LB  = st.session_state.training_domain_LB

if "training_domain" in  st.session_state:
    training_domain = st.session_state.training_domain
    training_domain_UB = st.session_state.training_domain_UB
    training_domain_LB  = st.session_state.training_domain_LB




#------------------ Model and Feature Names ------------------#
st.session_state.gen_model_name = '/_genModel_MAG.pth'
gen_model_name = st.session_state.gen_model_name 
st.session_state.gen_features = "/_genFeatures.npz"
gen_features = st.session_state.gen_features
st.session_state.gen_targets = "/_genTargets_MAG.npz"
gen_targets= st.session_state.gen_targets

st.session_state.base_model_name = '/_BaseModel_MAG.pth'
base_model_name =st.session_state.base_model_name
st.session_state.base_features = "/_BaseFeatures.npz"
base_features= st.session_state.base_features
st.session_state.base_targets = "/_BaseTargets_MAG.npz"
base_targets = st.session_state.base_targets

st.session_state.updated_model_name = '/_updatedModel_MAG.pth'
updated_model_name = st.session_state.updated_model_name
st.session_state.retrain_features = "/_RetrainFeatures.npz"
retrain_features = st.session_state.retrain_features
st.session_state.retrain_targets = "/_RetrainTargets_MAG.npz"
retrain_targets = st.session_state.retrain_targets

st.session_state.updated_gen_model_name = '/_updatedGenModel_MAG.pth'
updated_gen_model_name = st.session_state.updated_gen_model_name
st.session_state.retrain_gen_features = "/_RetrainGenFeatures.npz"
retrain_gen_features = st.session_state.retrain_gen_features
st.session_state.retrain_gen_targets = "/_RetrainGenTargets_MAG.npz"
retrain_gen_targets = st.session_state.retrain_gen_targets

### To disable the generate key until the pore distribution button is clicked on
def disable(b):
    st.session_state["disabled"] = b

# def display_buses():
#     st.write(st.session_state["load_changes"].values())

def retain():
    if "load_changes" in st.session_state:
        st.session_state.load_changes = st.session_state.load_changes



st.session_state.change_list = []
change_list = st.session_state.change_list




num_buses = 10
dir_path='.\CaseModels'
dir_names = os.listdir(dir_path)

selected_dirname = st.sidebar.selectbox('Select a Case-System', dir_names, index = None,  placeholder="Select a case", on_change=disable, args=(False,),)




if selected_dirname:
    num_buses = int(selected_dirname[3:].replace("-Bus", "").replace("K","000"))
    st.session_state.dir_name_st = os.path.join(dir_path, selected_dirname) #file_selector()
    dir_name = st.session_state.dir_name_st



#----------------------- M A P S -----------------------------------------#
def plot_map(state, plot_container, buses, branches):     
    fig = px.scatter_mapbox(buses, 
        lat='Latitude:1', 
        lon='Longitude:1', 
        color='Vpu',
        color_continuous_scale= px.colors.sequential.Hot, 
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
    fig2.update_traces(line=dict(color="dimgray", width=2))
    fig.add_traces(fig2.data)
    fig.update_layout(mapbox_style="open-street-map",
        title="Bus Voltages",
        width=800, height=800,
                )
    plot_container.plotly_chart(fig, theme="streamlit", use_container_width=True)

def get_name(self, row, state, id):
        if self.name == "Buses":
            name = str(row["Bus"])
        elif self.name == "ACLines" or self.name == "Transformers":
            name = "({},{},{})".format(row["Bus From"], row["To Bus"], row['Ckt'])
        elif self.name == "Generators" or self.name == "Loads":
            name = "({},{})".format(row['Bus'], row['ID'])
        row['name'] = name
        return row

def update_voltage(row, v):
    row["Vpu"] = v[row["BusNum"]]
    return row 

#----------------------- E N D M A P S -----------------------------------------#


####---------------- CLASS FOR JULIA TO RUN POWER MODEL ---------------#####
class mld:
    def __init__(self, jl_file_name, model_file):
        self.jlFile = jl_file_name
        self.modelFile = model_file


    def createJLFile(self):
        self.io = open(self.jlFile, "w+")


    def writeJLHeader(self):
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
        file = self.modelFile
        file = file.replace('\\', "\\\\")
        self.io.write(f'data = PowerModels.parse_file("{file}")\n')


    def writeJLGetBaseInfo(self):
        self.io.write('nlp_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-6, "print_level"=>0)\n')
        self.io.write('results = Dict{String, Any}()\n')
        self.io.write('results["system"] = data\n')
        self.io.write('result = PowerModels.solve_ac_pf(data, nlp_solver)\n')
        self.io.write('results["base pf"] = data\n')
        self.io.write('results["pf"] = Dict{Int, Any}()\n')
        self.io.write('results["NOpf"] = Dict{Int, Any}()\n')


    def writeJLLoadChangePF(self):
        self.io.write(f'samples = {self.samples}\n')
        self.io.write('for i = 1:samples\n')
        self.io.write('\tdata_ = deepcopy(data)\n')
        self.io.write('\tl = length(keys(data_["load"]))\n')
        self.io.write('\tn = rand(1:l)\n')
        self.io.write('\tm = sample(1:l, n, replace=false)\n')
        self.io.write('\tdelta = Dict{Int, Any}()\n')
        self.io.write('\tfor (j, k) in enumerate(m)\n')
        self.io.write('\t\tpct = rand(Uniform(0.01,5))\n')
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
        filename = self.jsonName.replace("\\", "/")
        self.io.write(f'open("{filename}","w") do f\n')
        self.io.write('\tJSON.print(f, results)\n')
        self.io.write('end\n')
####---------------- CLASS FOR JULIA TO RUN POWER MODEL ---------------#####


#------------------ Model and Feature Names ------------------#




def collectGridData(dir_path_): 
    
    if __name__ == '__main__':
        filename = ""
        holder = 1
                        
        for file in os.listdir(dir_path_):
            if '.json' in file:
                filename = os.path.join(dir_path_, file)
                f = open(filename)
                data_ = json.load(f)
                base_system = data_["base pf"]
                    
                if holder == 1:
                    base_gen = base_system['gen']
                    base_buses_wt_load = base_system['load']
                    base_vpu = base_system["bus"]
                    
                    base_gen_MW = []
                    base_gen_MVar = []
                    base_load_MW = []
                    base_load_MVar = []
                    
                    buses_ = []
                    buses_wt_gen_ = []
                    buses_wt_load_ = []
                    
                    for the_bus_ in range(1,  num_buses + 1):
                        buses_.append(int(the_bus_))  # same as keys from PM
                        
                    for the_bus_ in range(1, len(base_gen.keys()) + 1): #data_dict_genPower.keys():
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

                return buses_, buses_wt_load_, buses_wt_gen_, base_gen_MW, base_gen_MVar, base_load_MW, base_load_MVar, base_vpu

##########################################################################################################################

########################### ML Functions ################################################################################
# Define parameters
def defineParameters(feature, target):
    input_size = feature.shape[-1]
    output_size = target.shape[-1]
    hidden1_size = int(input_size /3) 
    hidden2_size = int(hidden1_size/2)
    return input_size, hidden1_size, hidden2_size, output_size


##### ML Functions ##########
# Define the neural network model with two hidden layers for the voltages
class TwoHiddenLayerNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(TwoHiddenLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
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

    return model, train_losses, val_losses


def generateModel(model, features, targets, training):

    # Define parameters
    input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

    if training == "base model":
        # Create the model
        model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
    elif training == "generator model":
        model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
    else:
        pass

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate data
    num_samples = features.shape[0]
    num_train_samples = int(num_samples * 0.7)  # 70%
    num_test_samples = int(num_samples * 0.15)   # 15%
    num_val_samples = num_samples - num_train_samples - num_test_samples #15%

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
    num_epochs = 200
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
    if training == "base model":
        torch.save(model.state_dict(), dir_name + base_model_name)
        print("*********** NEW FEATURES AND NEW ML MODEL CREATED **************************")
    elif training == "on-the-fly model":
        torch.save(model.state_dict(), dir_name + updated_model_name)
        print(f"Updated model saved as {updated_model_name}")
        print("*********** UPDATED ML MODEL CREATED **************************")
    elif training == "generator model":
        torch.save(model.state_dict(), dir_name + gen_model_name)
        print(f"Updated model saved as {gen_model_name}")
        print("*********** GENERATOR POWER PREDICTOR MODEL HAS BEEN CREATED **************************")


    return model

######################################### FUNCTIONS FOR New Test #######################################################
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
            all_output = outputs.numpy()
    new_test_loss /= len(new_loader.dataset)

    if on_the_fly == 1:
        print(f"New Test Loss: {new_test_loss}")

        #plot results
        #plotPrediction(all_output, new_targets, new_num_buses)

    return outputs, new_test_loss





# st.session_state.change_list = []
# change_list = st.session_state.change_list




# num_buses = 10
# dir_path='.\CaseModels'
# dir_names = os.listdir(dir_path)

# selected_dirname = st.sidebar.selectbox('Select a Case-System', dir_names, index = None,  placeholder="Select a case", on_change=disable, args=(False,),)

# if selected_dirname:
#     num_buses = int(selected_dirname[3:].replace("-Bus", "").replace("K","000"))
#     st.session_state.dir_name_st = os.path.join(dir_path, selected_dirname) #file_selector()
#     dir_name = st.session_state.dir_name_st

    
if selected_dirname:
    tab1, tab2, tab3 = st.tabs(["Activity", f"{selected_dirname[3:]} Grid", 'Model Information'])
    tab1.markdown("###### Recent Activity")
    tab2.markdown("###### Grid Information")
    tab3.markdown("###### ML Model Architecture\n " + f" Current Training Domain: {training_domain_UB}")

    st.session_state.buses, st.session_state.buses_wt_load, st.session_state.buses_wt_gen,\
        st.session_state.base_gen_MW, st.session_state.base_gen_MVar, st.session_state.base_load_MW, st.session_state.base_load_MVar, st.session_state.base_VPU = collectGridData(dir_name)
    
    buses = st.session_state.buses; buses_wt_load = st.session_state.buses_wt_load; buses_wt_gen = st.session_state.buses_wt_gen
    base_gen_MW = st.session_state.base_gen_MW; base_gen_MVar = st.session_state.base_gen_MVar
    base_load_MW = st.session_state.base_load_MW; base_load_MVar = st.session_state.base_load_MVar
    base_VPU = st.session_state.base_VPU

    st.session_state.log1 = tab1.text_area("Grid Information", f"Buses with Load: {str(buses_wt_load)[1:-1]}\n\n\
Buses with Generators: {str(buses_wt_gen)[1:-1]}")

    st.session_state.log2 = tab2.text_area("# Grid Information", f"Buses with Load: {str(buses_wt_load)[1:-1]}\n\n\
Buses with Generators: {str(buses_wt_gen)[1:-1]}", label_visibility="hidden") #\n {str(np.round(base_load_MW,2))[1:-1]}"
    
    ################  P L O T M A P ###################################################################################
    with tab2:
        #col1_p1, col2_p2 = st.columns(2)

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
        v = {}
        buses["Vpu"] = 0.0
        buses["size"] = 10
        for i, bus in base_VPU.items():
            v[str(i)] = bus['vm']
        buses_df = buses.apply(update_voltage, axis=1, args=(v,))
        plot_container = st.container(border=True)
        plot_map(st.session_state, plot_container, buses_df, branches)
        #plot_map(st.session_state, col2_p2, buses_df, branches)
     ################ E N D  P L O T M A P ###################################################################################

else:
    tab1, tab2, tab3 = st.tabs(["Activity", f"Grid", 'Model Information'])


st.sidebar.write('###### Working Directory: `%s`' %selected_dirname)

#st.sidebar.markdown("## Power Flow (Load Changes)")

load_changes = st.sidebar.radio('Select Load Change Procedure', options=[1, 2,3], 
                                format_func=lambda x: ["Random Buses", "Custom Buses", "User-Defined"][x-1], disabled=st.session_state.get("disabled", True),  horizontal=1)

