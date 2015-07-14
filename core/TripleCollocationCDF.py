'''
TC with CDF matching
'''
import numpy as np
from TripleCollocation import TripleCollocationBare
import TimeSeriesDataFrame as tsdf
def TripleCollocationCDF(timeseriesdf, n_bins = 100, degree = 5, tailscutoff = 0.01):
    variances={}
    for productname in timeseriesdf.names:
        data=[]
        names=[]
        datapn=timeseriesdf.values_from_name(productname)
        for productnameinner in timeseriesdf.names:
            names.append(productnameinner)
            if productnameinner == productname:
                data.append(datapn)
            else:
                datapni=timeseriesdf.values_from_name(productnameinner)
                datapnirs=cdf_match(datapni,datapn,n_bins=n_bins, degree = degree, tailscutoff = tailscutoff)
                data.append(datapnirs)
        timeseriesdfi=tsdf.TimeSeriesDataFrame(names,data)
        variancesoutp=TripleCollocationBare(timeseriesdfi)
        variances[productname]=variancesoutp[productname]
    return variances        

def cdf_match(inputdata, referencedata, n_bins = 100, degree = 5, tailscutoff = 0.01):
    #Adapted from function cdf_match (pytesmo-0.2.5):
    #Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
    #All rights reserved.
    
    #Redistribution and use in source and binary forms, with or without
    #modification, are permitted provided that the following conditions are met:
    #   * Redistributions of source code must retain the above copyright
    #     notice, this list of conditions and the following disclaimer.
    #    * Redistributions in binary form must reproduce the above copyright
    #      notice, this list of conditions and the following disclaimer in the
    #      documentation and/or other materials provided with the distribution.
    #    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
    #      names of its contributors may be used to endorse or promote products
    #      derived from this software without specific prior written permission.
    
    #THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    #ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    #WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    #DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, 
    #DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
    #DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    #(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    #LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    #ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    #(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    #SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    def ecdf_bins(data,n_bins,version=1):     
        hist_data,samples_data=np.histogram(data,bins=n_bins,density=True)
        ecdf_samples_data_raw=np.cumsum(hist_data)
        ecdf_samples_data=ecdf_samples_data_raw/ecdf_samples_data_raw[-1]
        return samples_data,ecdf_samples_data    
    samples_input,ecdf_samples_input=ecdf_bins(inputdata,n_bins)   
    samples_reference,ecdf_samples_reference=ecdf_bins(referencedata,n_bins)        
    ind_input = np.nonzero((ecdf_samples_input > tailscutoff) & (ecdf_samples_input < 1-tailscutoff))[0]
    ind_reference = np.nonzero((ecdf_samples_reference > tailscutoff) & (ecdf_samples_reference < 1-tailscutoff))[0]  
      
    # compute discrete operator
    disc_op = []    
    for i, value in np.ndenumerate(ecdf_samples_input[ind_input]):
        diff = value - ecdf_samples_reference[ind_reference]
        minabsdiff = min(np.abs(diff))
        minpos = np.where(np.abs(diff) == minabsdiff)[0][0]
        disc_op.append((samples_input[ind_input[i]]) - (samples_reference[ind_reference[minpos]]))
        
    # compute continuous operator
    cont_op = np.polyfit(samples_input[ind_input], disc_op, degree, full=True)    
    in_data_matched = inputdata - np.polyval(cont_op[0], inputdata)    
    return in_data_matched
