using PowerModels
import InfrastructureModels
import Memento
Memento.setlevel!(Memento.getlogger(InfrastructureModels), "error")
PowerModels.logger_config!("error")
import Ipopt
import Random
using StatsBase
using Distributions
import JuMP
import Random
using JSON

start_time = time()
data = PowerModels.parse_file(".\\etc\\on-the-fly\\01 14-Bus\\14Bus.m")
buses = []
for (i, gen) in data["gen"]
	if !(gen["gen_bus"] in buses)
		append!(buses,gen["gen_bus"])
	end
end
gen_dict = Dict{String, Any}()
genfuel = Dict{String, Any}()
gentype = Dict{String, Any}()
lookup_buses = Dict{Int, Any}()
counter = 1
for (i, gen) in data["gen"]
	if gen["gen_bus"] in keys(lookup_buses) && gen["gen_status"] == 1
		indx = lookup_buses[gen["gen_bus"]]
		gen_dict[indx]["pg"] += gen["pg"]
		gen_dict[indx]["qg"] += gen["qg"]
		gen_dict[indx]["pmax"] += gen["pmax"]
		gen_dict[indx]["pmin"] += gen["pmin"]
		gen_dict[indx]["qmax"] += gen["qmax"]
		gen_dict[indx]["qmin"] += gen["qmin"]
	elseif gen["gen_status"] == 1
		gen["index"] = counter
		gen_dict["$(counter)"] = gen
		genfuel["$(counter)"] = data["genfuel"][i]
		gentype["$(counter)"] = data["gentype"][i]
		lookup_buses[gen["gen_bus"]] = "$(counter)"
		global counter += 1
	end
end
data["gen"] = gen_dict
data["genfuel"] = genfuel
data["gentype"] = gentype
nlp_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-6, "print_level"=>0)
results = Dict{String, Any}()
results["system"] = data
result = PowerModels.solve_ac_pf(data, nlp_solver)
results["base pf"] = data
results["pf"] = Dict{Int, Any}()
results["NOpf"] = Dict{Int, Any}()
samples = 12000
for i = 1:samples
	data_ = deepcopy(data)
	l = length(keys(data_["load"]))
	n = rand(1:l)
	m = sample(1:l, n, replace=false)
	delta = Dict{Int, Any}()
	for (j, k) in enumerate(m)
		pct = rand(Uniform(0.01,8.0))
		pd = (pct) * data_["load"]["$(k)"]["pd"]
		qd = (pct) * data_["load"]["$(k)"]["qd"]
		data_["load"]["$(k)"]["pd"] = pd
		data_["load"]["$(k)"]["qd"] = qd
		delta[k] = pct
	end
	result = PowerModels.solve_ac_pf(data_, nlp_solver)
	result["load"] = data_["load"]
	result["delta"] = delta
	if result["termination_status"] == LOCALLY_SOLVED
		results["pf"][i] = result
	else
		results["NOpf"][i] = result
	end
end
open("./etc/on-the-fly/01 14-Bus/14Bus.json","w") do f
	JSON.print(f, results)
end
