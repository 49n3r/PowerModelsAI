# PowerModel-AI

## Abstract
The real-time creation of machine-learning models via active or on-the-fly learning has attracted considerable interest across various scientific and engineering disciplines.  These algorithms enable machines to autonomously build models while remaining operational. Through a series of query strategies, the machine can evaluate whether newly encountered data fall outside the scope of the existing training set. In this study, we introduce PowerModel-AI, an end-to-end machine learning software designed to accurately predict alternating current (AC) power flow solutions. We present detailed justifications for our model design choices and demonstrate that selecting the right input features effectively captures the load flow decoupling inherent in power flow equations. Our approach incorporates on-the-fly learning, where power flow calculations are initiated only when the machine detects a need to improve the dataset in regions where the model's performance is sub-optimal, based on specific criteria. Otherwise, the existing model is used for power flow predictions. This study includes analyses of five Texas A&M synthetic power grid cases, encompassing the 14-, 30-, 37-, 200-, and 500-bus systems.  The training and test datasets were generated using PowerModel.jl, an open-source power flow solver/optimizer developed at Los Alamos National Laboratory, United States.


# Copyright & License:
This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
