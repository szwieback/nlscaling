'''
Provides the class TimeSeriesDataFrame, which provides several routines for accessing the values of different but corresponding time series
'''
class TimeSeriesDataFrame(object):
    def __init__(self,namelist,valuelist):
        # Constructor
        # input parameters
        #    - namelist, list of strings: names of the time series
        #    - valuelist, list of numpy arrays: list of time series (all the same length)
        self.names=namelist
        self.values=valuelist
        assert len(set([len(v) for v in valuelist]))
    def dictionary_values(self):
        # returns the time series as dictionary
        return dict(zip(self.names,self.values))
    
    def values_from_name(self,name):
        # returns the array of product name
        return self.dictionary_values()[name]
    
    def list_value_instance(self,instance):
        # returns values at instance as list
        return [v[instance] for v in self.values]
    
    def dictionary_value_instance(self,instance):
        # returns values at instance as dictionary
        return dict(zip(self.names,self.list_value_instance(instance)))
    
    def get_length(self):
        # returns length of time series
        return len(self.values[0])
        