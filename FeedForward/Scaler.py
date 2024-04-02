import numpy as np

class Scaler:
    def __init__(self,array :np.ndarray, lbound=0.0, ubound = 1.0):
        if lbound >= ubound:
            statement = '\nlbound must be less than ubound.'*(lbound >= ubound)
            raise ValueError(statement)
        
        self.original = array
        self.factor = (array - array.min(axis = 0)).max(axis = 0)

        if (self.factor == 0).any():
            self.factor += (self.factor == 0)*1

        self.scaledarray = (array - array.min(axis = 0))/self.factor
        
        self.lbound = lbound
        self.ubound = ubound
        
        self.scaledarray *= (ubound-lbound)
        self.scaledarray += lbound
        
    def unscale(self,array):
        res = array - self.lbound
        res /= (self.ubound-self.lbound)
        
        res*=self.factor
        res+=self.original.min(axis = 0)
        
        return res
    def scale(self,array:np.ndarray):
        res =  array.copy()
        res = (res -self.original.min(axis = 0))/self.factor

        res *= (self.ubound - self.lbound)
        res += self.lbound

        return res