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

#from Module.methods import *

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


###################################################
####################################################

############################ INITIALIZATIN FUNCTIONS ###############################################################
st.set_page_config(
    page_title="PoWerML",
    page_icon="⚡",
    layout="wide"
)

#----------------- Parameters ---------------------------------#
logo = "./etc/Setup Files/LOGO.png"
domain_file = "/Training Domain.txt"
julia_params = "./etc/Setup Files/Julia Parameters.txt"
retrain_folder = "./etc/on-the-fly"
base_training_domian_limit = 5.0

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

### To disable the generate key until the pore distribution button is clicked on
def disableCaseSelect(b):
    st.session_state["disableCaseSelect"] = b

def disableRetrain(b):
    st.session_state["disableRetrain"] = b

# def display_buses():
#     st.write(st.session_state["load_changes"].values())

def retain():
    if "load_changes" in st.session_state:
        st.session_state.load_changes = st.session_state.load_changes

#*************************************************************************** S T A R T ***************************************************************************#
#st.set_page_config(page_title="Carbon Nanotube Initializer", page_icon="https://chinonsougwumadu.com/wp-content/uploads/2024/05/microsoftteams-image-17.jpg")
st.markdown("## PowerModels :flag-ai:")

st.sidebar.image(logo, width=120)
st.session_state.getButton = 0




num_buses = 10
dir_path='./CaseModels'
dir_names = os.listdir(dir_path)

selected_dirname = st.sidebar.selectbox('Select a Case-System', dir_names, index = None,  placeholder="Select a case", on_change=disableCaseSelect, args=(False,),)


if selected_dirname:
    num_buses = int(selected_dirname[3:].replace("-Bus", "").replace("K","000"))
    st.session_state.dir_name_st = os.path.join(dir_path, selected_dirname) #file_selector()
    dir_name = st.session_state.dir_name_st

    st.session_state.training_domain_limit = readTrainingDomain(retrain_folder+ "//" + selected_dirname + domain_file)
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

st.session_state.updated_bus_model_name = '/_updatedModel_MAG.pth'
updated_bus_model_name = st.session_state.updated_bus_model_name
st.session_state.retrain_bus_features = "/_RetrainFeatures.npz"
retrain_bus_features = st.session_state.retrain_bus_features
st.session_state.retrain_bus_targets = "/_RetrainTargets_MAG.npz"
retrain_bus_targets = st.session_state.retrain_bus_targets

st.session_state.updated_gen_model_name = '/_updatedGenModel_MAG.pth'
updated_gen_model_name = st.session_state.updated_gen_model_name
st.session_state.retrain_gen_features = "/_RetrainGenFeatures.npz"
retrain_gen_features = st.session_state.retrain_gen_features
st.session_state.retrain_gen_targets = "/_RetrainGenTargets_MAG.npz"
retrain_gen_targets = st.session_state.retrain_gen_targets


st.session_state.change_list = []
change_list = st.session_state.change_list


#----------------------- M A P S -----------------------------------------#
def plot_map(state, plot_container, buses, branches, map_title):     
    fig = px.scatter_mapbox(buses, 
        lat='Latitude:1', 
        lon='Longitude:1', 
        color='Vpu',
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
        filename = self.jsonName.replace("\\", "/")
        self.io.write(f'open("{filename}","w") do f\n')
        self.io.write('\tJSON.print(f, results)\n')
        self.io.write('end\n')
####---------------- CLASS FOR JULIA TO RUN POWER MODEL ---------------#####


#------------------ Model and Feature Names ------------------#

# def collectGridData(dir_path_): 
#     if __name__ == '__main__':
#         filename = ""
#         holder = 1
                        
#         for file in os.listdir(dir_path_):
#             if '.json' in file:
#                 filename = os.path.join(dir_path_, file)
#                 f = open(filename)
#                 data_ = json.load(f)
#                 base_system = data_["base pf"]
                    
#                 if holder == 1:
#                     base_gen = base_system['gen']
#                     base_buses_wt_load = base_system['load']
#                     base_vpu = base_system["bus"]
#                     NOpf_data = data_["NOpf"]
#                     pf_data = data_["pf"]
#                     pf_iterations = np.array(list(pf_data.keys()))
                    
#                     base_gen_MW = []
#                     base_gen_MVar = []
#                     base_load_MW = []
#                     base_load_MVar = []
                    
#                     buses_ = []
#                     buses_wt_gen_ = []
#                     buses_wt_load_ = []
                    
#                     for the_bus_ in range(1,  num_buses + 1):
#                         buses_.append(int(the_bus_))  # same as keys from PM
                        
#                     for the_bus_ in range(1, len(base_gen.keys()) + 1): #data_dict_genPower.keys():
#                         buses_wt_gen_.append(int(base_gen[str(the_bus_)]['gen_bus']))
#                         base_gen_MW.append(base_gen[str(the_bus_)]['pg'])
#                         base_gen_MVar.append(base_gen[str(the_bus_)]['qg'])

    
#                     for the_bus_ in range(1, len(base_buses_wt_load.keys()) + 1):
#                         buses_wt_load_.append(int(base_buses_wt_load[str(the_bus_)]['load_bus']))
#                         base_load_MW.append(base_buses_wt_load[str(the_bus_)]['pd'])
#                         base_load_MVar.append(base_buses_wt_load[str(the_bus_)]['qd'])


#                     buses_ = np.array(buses_)
#                     buses_wt_gen_ = np.array(buses_wt_gen_)
#                     buses_wt_load_ = np.array(buses_wt_load_)

#                 return buses_, buses_wt_load_, buses_wt_gen_, base_gen_MW, base_gen_MVar, base_load_MW, base_load_MVar, base_vpu


def collectGridData(dir_path_, base_case): #, buses_wt_load_, buses_wt_gen_, percent_change_, num_buses_):
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
                base_vpu = base_system["bus"]

                NOpf_data = data_["NOpf"]
                pf_data = data_["pf"]
                pf_iterations = np.array(list(pf_data.keys()))

                print(f"{len(NOpf_data.keys())} out of {len(NOpf_data.keys()) + len(pf_data.keys())} cases did not solve")
                print(len(pf_iterations))

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
                            buses_.append(int(the_bus_))  # same as keys from PM
                            
                        for the_bus_ in range(1, len(base_gen.keys()) + 1): #data_dict_genPower.keys():
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
                        base_gen_MW_arr = np.zeros([len(pf_iterations), num_buses_])
                        holder += 1

                    # Get the load data
                    if True:
                          
                        MW_data = [data_dict_loadedBus[str(mw)]['pd'] for mw in range(1, len(data_dict_loadedBus.keys()) + 1)]
                        Mvar_data = [data_dict_loadedBus[str(mw)]['qd'] for mw in range(1, len(data_dict_loadedBus.keys()) + 1)]
    
                        load_MW_arr[run_iter] = np.squeeze(MW_data)  # buses with load,
                        load_Mvar_arr[run_iter] = np.squeeze(Mvar_data)


                    # Get VPU data
                    if True:
                        vpu_data = [data_dict_busVPU[str(mw)]['vm'] for mw in buses_]
                        vangle_data = [data_dict_busVPU[str(mw)]['va'] for mw in buses_]
                        bus_vpu_arr[run_iter] = np.squeeze(vpu_data) 
                        bus_vangle_arr[run_iter] = np.squeeze(vangle_data)

                    # Get generator daya
                    if True: 
                        genMW_data = [data_dict_genPower[str(mw)]['pg'] for mw in range(1, len(data_dict_genPower.keys()) + 1)]
                        genMvar_data = [data_dict_genPower[str(mw)]['qg'] for mw in range(1, len(data_dict_genPower.keys()) + 1)]
                        
                        gen_MW_arr[run_iter][buses_wt_gen_-1] = np.squeeze(genMW_data) 
                        base_gen_MW_arr[run_iter][buses_wt_gen_-1] = np.squeeze(base_gen_MW)

    if base_case:
        return buses_, buses_wt_load_, buses_wt_gen_, base_gen_MW, base_gen_MVar, base_load_MW, base_load_MVar, base_vpu
    else:            
        return  bus_vpu_arr, bus_vangle_arr, load_MW_arr, load_Mvar_arr, gen_MW_arr, base_gen_MW_arr

##########################################################################################################################

def runModel():
    if training_domain_UB <= base_training_domian_limit:
        tab1.info("Using base model. Within base training domian")
        bus_model_file = base_model_name
        gen_model_file = gen_model_name
        bus_feature_file = base_features
        bus_target_file = base_targets
        gen_feature_file = gen_features
        gen_target_file = gen_targets
    else:
        if os.path.exists(dir_name + retrain_bus_targets): 
            tab1.info(f"Using updated model. Current Training Upper limit is {training_domain_UB}")
            bus_model_file = updated_bus_model_name
            gen_model_file = updated_gen_model_name
            bus_feature_file = retrain_bus_features
            bus_target_file = retrain_bus_targets
            gen_feature_file = retrain_gen_features
            gen_target_file = retrain_gen_targets
        else: 
            tab1.info("Using base model. Outside base training domian")
            bus_model_file = base_model_name
            gen_model_file = gen_model_name
            bus_feature_file = base_features
            bus_target_file = base_targets
            gen_feature_file = gen_features
            gen_target_file = gen_targets



    if os.path.exists(dir_name + bus_model_file):   
        ## Load Existing data
        features = torch.tensor(np.load(dir_name + bus_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + bus_target_file)["arr_0"])
            
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.base_model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
        base_model = st.session_state.base_model
        base_model.load_state_dict(torch.load(dir_name + base_model_name))
        base_model.eval()
    
    if os.path.exists(dir_name + gen_model_file):   
        ## Load Existing data
        features = torch.tensor(np.load(dir_name + gen_feature_file)["arr_0"])
        targets = torch.tensor(np.load(dir_name + gen_target_file)["arr_0"])
        
        # Define parameters
        input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

        # import the model
        st.session_state.gen_model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
        gen_model = st.session_state.gen_model
        gen_model.load_state_dict(torch.load(dir_name + gen_model_name))
        gen_model.eval()

    return base_model, gen_model

# def runModel():
#     if training_domain_UB <= base_training_domian_limit:
#         model_name = 
#     if os.path.exists(dir_name + base_model_name):   
#         ## Load Existing data
#         features = torch.tensor(np.load(dir_name + base_features)["arr_0"])
#         targets = torch.tensor(np.load(dir_name + base_targets)["arr_0"])
            
#         # Define parameters
#         input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

#         # import the model
#         st.session_state.base_model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
#         base_model = st.session_state.base_model
#         base_model.load_state_dict(torch.load(dir_name + base_model_name))
#         base_model.eval()
    
#     if os.path.exists(dir_name + gen_model_name):   
#         ## Load Existing data
#         features = torch.tensor(np.load(dir_name + gen_features)["arr_0"])
#         targets = torch.tensor(np.load(dir_name + gen_targets)["arr_0"])
        
#         # Define parameters
#         input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

#         # import the model
#         st.session_state.gen_model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)
#         gen_model = st.session_state.gen_model
#         gen_model.load_state_dict(torch.load(dir_name + gen_model_name))
#         gen_model.eval()

#     return base_model, gen_model

def generateRetrainModels(retrain_json_loc):
    ########################## Gen Model ###############################################
    gen_features_r, gen_targets_r = genFeatureEngineering(retrain_json_loc) 

    # Save Data for ML processing
    np.savez(dir_name + retrain_gen_features, gen_features_r)
    np.savez(dir_name + retrain_gen_targets, gen_targets_r)

    #generate Gen model
    initial_gen_model = []
    generateModel(initial_gen_model, gen_features_r, gen_targets_r, "generator model")

    ########################## bus Model ###############################################
    bus_features_r, bus_targets_r = busFeatureEngineering(retrain_json_loc) 

    # Save Data for ML processing
    np.savez(dir_name + retrain_bus_features, bus_features_r)
    np.savez(dir_name + retrain_bus_targets, bus_targets_r)

    #generate Gen model
    initial_bus_model = []
    generateModel(initial_bus_model, bus_features_r, bus_targets_r, "bus model")

    #gen_model = generateModel(initial_model, features_r, targets_r, "generator model")
    #return gen_model

###################################### F E A T U R E  E N G I N E E R I N G ################################################

def genFeatureEngineering(retrain_json_loc):
    dir_path = os.path.join(os.getcwd(), retrain_json_loc)

    _, _, load_MW_arr, load_Mvar_arr, gen_MW_arr, base_gen_MW_arr = collectGridData(dir_path, base_case = 0)

    features = torch.cat((torch.tensor(load_MW_arr), torch.tensor(load_Mvar_arr),torch.tensor(base_gen_MW_arr)),dim =1) #
    targets = torch.tensor(gen_MW_arr)
    return features, targets



def busFeatureEngineering(retrain_json_loc):
    dir_path = os.path.join(os.getcwd(), retrain_json_loc)

    bus_vpu_arr, _, load_MW_arr, load_Mvar_arr, gen_MW_arr, _ = collectGridData(dir_path, base_case = 0)

    num_buses_wt_load = len(buses_wt_load) # Number of buses with loads
    num_changes = len(bus_vpu_arr)     # Number of features (training data)

    features = torch.cat((torch.tensor(load_MW_arr), torch.tensor(load_Mvar_arr),torch.tensor(gen_MW_arr)),dim =1) #       
    print(f"The size of FEATURES for {num_changes} changes to {num_buses_wt_load} buses with load is {features.shape}")

    targets = torch.tensor(bus_vpu_arr) 
    print(f"The size of OUTPUT for {num_changes} changes to {num_buses_wt_load} buses with load is {targets.shape}")

    return features, targets

###################################### E N D OF F E A T U R E  E N G I N E E R I N G ################################################

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

#-------------------------------------------------- on-the-fly-learning ------------------------------------------------------------------#
def generateModel(model, features, targets, model_type):

    # Define parameters
    input_size, hidden1_size, hidden2_size, output_size = defineParameters(features, targets)

    model = TwoHiddenLayerNN(input_size, hidden1_size, hidden2_size, output_size)


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
    if model_type == "bus model":
        print("*********** NEW FEATURES AND NEW ML MODEL CREATED **************************")
        torch.save(model.state_dict(), dir_name + updated_bus_model_name)
        print(f"Updated model saved as {updated_bus_model_name}")
        print("*********** UPDATED ML MODEL CREATED **************************")
    elif model_type == "generator model":
        torch.save(model.state_dict(), dir_name + updated_gen_model_name)
        print(f"Updated model saved as {updated_gen_model_name}")
        print("*********** GENERATOR POWER PREDICTOR MODEL HAS BEEN CREATED **************************")

    #return model

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

######################################### E N D O F F U N C T I O N S#################################################################











##################################### S T A R T O F A P P L I C A T I O N ############################################################
    
if selected_dirname:
    tab1, tab2, tab3 = st.tabs(["Activity", f"{selected_dirname[3:]} Grid", 'Model Information'])
    tab1.markdown("###### Recent Activity")
    tab2.markdown("###### Grid Information")
    tab3.markdown("###### ML Model Architecture")
    tab3.info(f" Current Training Domain Upper Limit is: {training_domain_UB}")

    st.session_state.buses, st.session_state.buses_wt_load, st.session_state.buses_wt_gen,\
        st.session_state.base_gen_MW, st.session_state.base_gen_MVar, st.session_state.base_load_MW, st.session_state.base_load_MVar, st.session_state.base_VPU = collectGridData(dir_name, base_case=1)
    
    buses = st.session_state.buses; buses_wt_load = st.session_state.buses_wt_load; buses_wt_gen = st.session_state.buses_wt_gen
    base_gen_MW = st.session_state.base_gen_MW; base_gen_MVar = st.session_state.base_gen_MVar
    base_load_MW = st.session_state.base_load_MW; base_load_MVar = st.session_state.base_load_MVar
    base_VPU = st.session_state.base_VPU

#     st.session_state.log1 = tab1.text_area("Grid Information", f"Buses with Load: {str(buses_wt_load)[1:-1]}\n\n\
# Buses with Generators: {str(buses_wt_gen)[1:-1]}")

    st.session_state.log2 = tab2.text_area("# Grid Information", f"Buses with Load: {str(buses_wt_load)[1:-1]}\n\n\
Buses with Generators: {str(buses_wt_gen)[1:-1]}", label_visibility="hidden") #\n {str(np.round(base_load_MW,2))[1:-1]}"
    
    ################  P L O T M A P ###################################################################################
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
        v = {}
        buses["Vpu"] = 0.0
        buses["size"] = 10
        for i, bus in base_VPU.items():
            v[str(i)] = bus['vm']
        st.session_state.buses_df = buses.apply(update_voltage, axis=1, args=(v,))
        plot_container = st.container(border=True)
        #plot_map(st.session_state, plot_container,buses_df, branches)
        plot_map(st.session_state, col1_p1, st.session_state.buses_df, branches, "Original Bus Voltages")
     ################ E N D  P L O T M A P ###################################################################################

else:
    tab1, tab2, tab3 = st.tabs(["Activity", f"Grid", 'Model Information'])


st.sidebar.write('###### Working Directory: `%s`' %selected_dirname)

load_changes = st.sidebar.radio('Select Load Change Procedure', options=[1, 2,3], 
                                format_func=lambda x: ["Random Buses", "Custom Buses", "User-Defined"][x-1], disabled=st.session_state.get("disableCaseSelect", True),  horizontal=1)

##############################################################################################################################################



##################################################### LOAD BASE ML MODELS #########################################################################
if selected_dirname:
    base_model, gen_model = runModel()
##################################################### END OF  LOAD BASE ML MODELS #####################################################################

## Collect grid information from base case (default system)  
if selected_dirname:
    st.sidebar.number_input("Number of Load Buses to Change:", min_value=1, max_value= len(buses_wt_load), step = 1, key="num_changes",on_change=retain, help=f"Select the number of changes in the load for the {num_buses} bus case")
    num_changes = st.session_state.num_changes

with tab1:
    #Random Buses without repetition
    
    if "num_changes" in st.session_state and load_changes == 1:
        random_ints = ran.sample(range(0, len(buses_wt_load)), num_changes) # smampel withouto replacement
        random_buses = np.array(buses_wt_load)[random_ints]
        random_changes = np.random.rand(num_changes) * training_domain
        buses_to_change_load_list = []
        buses_to_change_load_list.append(random_buses)
        
        #st.write(f"{num_changes} Randomly Selected Bus(es): {np.array(buses_wt_load)[random_ints]}")
        st.write(f"{num_changes} Randomly Selected Bus(es): {str(random_buses)[1:-1]}")
        change_list.append(random_changes)
        df = pd.DataFrame(change_list, columns=buses_to_change_load_list)
        st.markdown("*Random Changes for Selected Bus(es)*")
        load_change_df = st.data_editor(df, key="load_df")

        tab2.markdown("*Random Changes for Selected Bus(es)*")
        tab2.data_editor(df)

        col_sb1, col_sb2 = st.sidebar.columns(2, gap = "small")
    

    #Custom Buses and User-Defined Changes
    if "num_changes" in st.session_state and (load_changes == 2 or load_changes == 3):
        change_list = []
        buses_to_change_load_list = []
        buses_to_change_load =st.multiselect("Select Bus(es) with load to change:", buses_wt_load, max_selections=num_changes, key="load_changes", on_change=retain, help ="Select the load to update")
        buses_to_change_load_list.append(np.array(buses_to_change_load))


        if load_changes == 3 and len(np.array(buses_to_change_load)) > 0:
            st.sidebar.number_input("New Upper Bound for Change Rate:", min_value=training_domain_LB, value = training_domain_UB, step = 0.1, key="training_domain_UB", help=f"Update the load change rate (may require on-th-fly training)", on_change=disableRetrain, args=(False,))

        if len(np.array(buses_to_change_load)) > 0:
            col_sb1, col_sb2 = st.sidebar.columns(2, gap = "small")
            st.session_state.getButton = 1      

        st.write(f"You have selected Bus No.: {str(buses_to_change_load)[1:-1]}")

        if load_changes == 2 and len(np.array(buses_to_change_load)) > 0: 
            st.sidebar.number_input("Modify Change Rate (default is 2.5):", min_value=training_domain_LB, max_value= training_domain_UB + 0.01, value = training_domain, step = 0.1, key="training_domain", help=f"Adjust the range of Load changes from 0.1 to 4.0")
            #num_changes = st.session_state.num_changes
            change_list.append(np.random.rand(len(buses_to_change_load))*training_domain)
            df = pd.DataFrame(change_list, columns= np.array(buses_to_change_load))
            st.markdown(f"*Random Load Changes at a rate of {training_domain:.2f}*")
            load_change_df = st.data_editor(df, key="load_df")
            col_sb1, col_sb2 = st.sidebar.columns(2, gap = "medium")

            tab2.markdown(f"*Random Load Changes at a rate of {training_domain:.2f}*")
            tab2.data_editor(df)          
            
        if load_changes == 3 and len(np.array(buses_to_change_load)) > 0:
            
            #st.sidebar.number_input("New Upper Bound for Change Rate:", min_value=training_domain_LB, value = training_domain_UB, step = 0.1, key="training_domain_UB", help=f"Update the load change rate (may require on-th-fly training)")
            change_text = st.text_input(f"Enter Load Changes for {num_changes} Buses. Separate with space")
            change_text_split = change_text.split()
            change_floats = []

            if training_domain_UB and training_domain_UB > training_domain_limit:
                #col_sb1, col_sb2 = st.sidebar.columns(2, gap = "medium")
                col_sb2.toggle("On-the-🪰", key="on_the_fly", on_change=disableRetrain, args=(False,))

            if len(np.array(buses_to_change_load)) != len(change_text_split):
                change_list = []
                st.warning(f'You have {len(np.array(buses_to_change_load))} buses yet specified {len(change_text_split)} changes.', icon="⚠️")
                change_list.append(np.random.rand(len(buses_to_change_load))*np.nan)

            else:
                change_list = []
                pos = 0
                
                for i in change_text_split:
                    changeRatio = float(i)
                    if changeRatio > training_domain_UB:
                        create_sidebar = 1
                        st.warning(f'Do you know what you are doing?\
                                   \n You have specified out-of-bound load change {changeRatio:.2f} for Bus {buses_to_change_load[pos]}.\
                                    \n Max. limit is: {training_domain_UB:.2f}\
                                   \n Default load change of {training_domain} has been set for Bus {buses_to_change_load[pos]} instead.', icon="⚠️")
                        changeRatio = training_domain
                        change_floats.append(changeRatio)

                    elif changeRatio < training_domain_LB:
                        st.warning(f'Do you know what you are doing?\
                                   \n You have specified out-of-bound load change {changeRatio:.2f} for Bus {buses_to_change_load[pos]}.\
                                    \n Min. limit is: {training_domain_LB:.2f}\
                                   \n The lowest load change of {training_domain_LB} has been set for Bus {buses_to_change_load[pos]} instead.', icon="⚠️")
                        changeRatio = training_domain_LB
                        change_floats.append(changeRatio)
                    else:
                        change_floats.append(changeRatio)
                    pos +=1
                change_list.append(np.array(change_floats))
            
            df = pd.DataFrame(data = change_list, columns =np.array(buses_to_change_load))
            load_change_df = st.data_editor(df, key="load_df")

            tab2.markdown(f"*User-defined Load Changes*")
            tab2.data_editor(df)

################################################################################################################################################
if len(change_list) > 0 and (load_changes == 1 or st.session_state.getButton == 1):
    
    #col_sb1, col_sb2 = st.sidebar.columns(2, gap = "medium")
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
    
    with tab3:
        col1_t3, col2_t3 = st.columns(2, gap = "large")

    if predict_button:
        try:
            # Make dictionary for changes in load
            buses_wt_load_MWdict = {}
            buses_wt_load_MVardict = {}
            buses_to_change_load_dict = {}
        
            for bus_ind, bus in enumerate(buses_wt_load):
                buses_wt_load_MWdict[str(bus)] = base_load_MW[bus_ind]
                buses_wt_load_MVardict[str(bus)] = base_load_MVar[bus_ind]
            
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
                    f_MVar[bus] = base_load_MVar[bus_ind]
                    f_MW_list.append(base_load_MW[bus_ind])
                    f_MVar_list.append(base_load_MVar[bus_ind])

            ###################### Predict generator VPU ############################################################################
            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.tensor(base_gen_MW_arr))) 
            f_gen = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_MW_arr)
            t_gen = torch.reshape(t, (1,-1))

            gen_VPU_pred, test_losses = predict(gen_model, f_gen, t_gen, criterion = nn.MSELoss(), on_the_fly = 0)
            gen_VPU_masked = gen_VPU_pred*base_gen_MW_mask

            tab1.write("Predicted Gen VPU (See Model information tab for more.)")
            tab1.write(gen_VPU_masked)

            col1_t3.write(f"Input: {f_gen.shape}")
            col1_t3.write(f_gen)

            col1_t3.write(f"Predicted Gen VPU: {gen_VPU_masked.shape}")
            col1_t3.write(gen_VPU_masked)
            #****************************************************************************************************************************#

            ###################### Predict Volatage Magnitude (VPU) ###############################################################################

            f = torch.cat((torch.tensor(f_MW_list), torch.tensor(f_MVar_list), torch.reshape(gen_VPU_masked, (-1,)))) 
            f_bus = torch.reshape(f, (1,-1))
            t = torch.tensor(base_gen_MW_arr)
            t_bus = torch.reshape(t, (1,-1))
            bus_VPU_pred, test_losses = predict(base_model, f_bus, t_bus, criterion = nn.MSELoss(), on_the_fly = 0)
    
            tab1.write("Predicted Bus VPU (See Model information tab for more.)")
            tab1.write(bus_VPU_pred)

            col2_t3.write(f"Input: {f_bus.shape}")
            col2_t3.write(f_bus)

            col2_t3.write(f"Predicted Bus VPU: {bus_VPU_pred.shape}")
            col2_t3.write(bus_VPU_pred)
        except:
            st.error("No model Found. Please check Directory", icon="🚨")

        ########################## PLOT Predicted MAP ################################################################
        with tab2:
            v = {}
            pred_VPU_numpy = bus_VPU_pred[0].numpy()
            #st.write(len(pred_VPU_numpy))
            for i, _vpu in enumerate(pred_VPU_numpy): #vpu_ in enumerate(bus_VPU_pred[0].numpy()):
                #st.write(i)
                v[str(i+1)] = _vpu
            
            #st.write(v)
            st.session_state.buses_df_pred = buses.apply(update_voltage, axis=1, args=(v,))

            #plot_container = st.container(border=True)
            #plot_map(st.session_state, plot_container, st.session_state.buses_df, branches)

            plot_map(st.session_state, col2_p2, st.session_state.buses_df_pred, branches,  "Predicted Bus Voltages")
        ########################## PLOT Predicted MAP ################################################################


# @st.dialog("Run on-the-🪰?")
# def retrainChoice():
#     st.write("If Yes, a new dataset for the training domain will be created using PowerModel. Note that this may take a while")
#     st.write("I want to run on-the on-the-🪰")
#     no_col, yes_col = st.columns(2)
#     no_button_ = no_col.button("No", key = "no_button")
#     yes_button_ = yes_col.button("Yes", key = "yes_button")
#     if no_button_:
#         st.session_state.retrain_choice = False
#         st.rerun()
#     elif yes_button_:
#         st.session_state.retrain_choice = True
#         st.rerun()
# with tab1:
#     if "on_the_fly" in st.session_state and st.session_state.on_the_fly:
#         if "retrain_choice" not in st.session_state:
#             retrainChoice()


with tab1:
    if "on_the_fly" in st.session_state and st.session_state.on_the_fly:
        col_confirm, col_retrain = st.columns(2, gap = "small")
        with col_confirm:
            with st.popover("Run on-the-🪰?", help="If Yes, a new dataset for the training domain will be created using PowerModel. Note that this may take a while", disabled=False, use_container_width=False):
                with st.container(border=True):
                    no_col, yes_col = st.columns(2)
                    no_button_ = no_col.button("No", key = "no_button", on_click=disableRetrain, args=(True,), disabled=st.session_state.get("disableRetrain",True))
                    yes_button_ = yes_col.button("Yes", key = "yes_button", on_click=disableRetrain, args=(True,), disabled=st.session_state.get("disableRetrain",True))
        with col_retrain:
            if yes_button_:
                julia_script_path = './RunJuliaScript.py'  

                writeTrainingDomain(retrain_folder+ "\\" + selected_dirname + domain_file, str(np.round(training_domain_UB,2)))

                with st.status("Generating On-the-🪰 Data...", expanded=False):
                    st.write("Creating julia script for PowerModel data")
                    loc = retrain_folder + '\\' + selected_dirname
                    for file in os.listdir(loc):
                        if '.m' in file:
                            test_case_m = loc + '\\' + file

                            jl_file = loc + "\juliaScript.jl"
                            writeJuliaPath(julia_params, jl_file)

                            runJulia = mld(jl_file, test_case_m)
                            runJulia.samples = 12000
                            runJulia.LB = 0.01
                            runJulia.UB = training_domain_UB
                            runJulia.jsonName = test_case_m[:-2]  + ".json"
                            runJulia.writeJLHeader()
                            runJulia.writeJLParse()
                            runJulia.writeJLGetBaseInfo()
                            runJulia.writeJLLoadChangePF()
                            runJulia.writeJLJSON()
                            runJulia.io.close()
                            st.write("Generating PowerModel data")
                            os.system("python " + julia_script_path)

                            st.write("Training New Model")
                            generateRetrainModels(retrain_json_loc = loc)
                            
                st.success("On-the-🪰 Generated!")
            
            if no_button_:
                st.info("Continue with Prediction. Make sure you are within the training domain")



# Load Base model if available
if selected_dirname:
    if os.path.exists(dir_name + base_model_name):   
        ## Load Existing data
        
        tab3.write("")
        tab3.markdown("###### Bus VPU Model Details: ")
        tab3.write(base_model)
    else:
        tab3.error("Please add a .pth file for the ML base model. See Documentation")

    
    if os.path.exists(dir_name + gen_model_name):   
       
        tab3.write("")
        tab3.markdown("###### Generator VPU Model Details: ")
        tab3.write(gen_model)
    else:
        tab3.error("Please add a .pth file for the ML Generator VPU model. See Documentation")


######################### E N D  O F  T H E  A L G O R I T M ######################################################################
