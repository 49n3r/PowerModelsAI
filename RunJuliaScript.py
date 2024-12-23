import os
#import julia
from julia import Main


def readJuliaPath(txt_path):
    with open(txt_path, "r") as f:
       julia_path = f.read()
    return julia_path

if __name__ == '__main__':

    julia_params = ".\etc\Setup Files\Julia Parameters.txt"
    jl_file =  readJuliaPath(julia_params) 

    print("*************************************************")
    print()
    print(jl_file)
    print(os.getcwd())
    print()

    Main.include(jl_file)
    


    