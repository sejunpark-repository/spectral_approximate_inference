from copy import copy
#from tensor.tensor import Tensor
#from tensor.tensor_tools import product_over
from gm.factor import Factor, product_over_
import numpy as np
from functools import reduce

class GraphicalModel():
    def __init__(self, variables=[], factors= []):
        if variables:
            self.variables = variables
        else:
            self.variables = []

        self.factors = []

        if factors:
            for factor in factors:
                self.add_factor(factor)

    '''
    Variable related operations
    '''

    def add_variable(self, variable):
        self.variables.append(variable)

    def add_variables_from(self, variables):
        for variable in variables:
            self.add_variable(variable)

    def remove_variable(self, variable):
        self.variables.remove(variable)

    def remove_variables_from(self, variables):
        for variable in variables:
            self.variables.remove(variable)
    def get_cardinality_for_(self, variable):
        factor = next(factor for factor in self.factors if variable in factor.variables)
        if factor:
            return factor.get_cardinality_for_(variable)
        else:
            raise ValueError("variable not in the model")
    def get_cardinalities_from(self, variables):
        cardinalities = []
        for variable in variables:
            cardinalities.append(self.get_cardinality_for_(variable))
        return cardinalities
    def contract_variable(self, variable, operator = 'sum', **kwargs):
        adj_factors = self.get_adj_factors(variable)
        new_factor = product_over_(*adj_factors).copy(rename=True)
        new_factor.marginalize([variable], operator = operator, **kwargs)
        for factor in adj_factors:
            self.remove_factor(factor)

        self.remove_variable(variable)
        self.add_factor(new_factor)
        '''
        if ~np.isfinite(np.max(new_factor.log_values)):
            for fac in adj_factors:
                print(fac)
                print(fac.values)

            raise ValueError()
        '''

        return new_factor
    def contract_variables_from(self, variable):
        for variable in variables:
            self.contract_variable(variable)

    def get_adj_factors(self, variable):
        factor_list =  []
        for factor in self.factors:
            if variable in factor.variables:
                factor_list.append(factor)

        return factor_list
    def degree(self, variable):
        return len(self.get_adj_factors(variable))

    '''
    Factor related operations
    '''

    def add_factor(self, factor):
        if set(factor.variables) - set(factor.variables).intersection(set(self.variables)):
            raise ValueError("Factors defined on variable not in the model.")
        self.factors.append(factor)

    def add_factors_from(self, factors):
        for factor in factors:
            self.add_factor(factor)

    def remove_factor(self, factor):
        self.factors.remove(factor)

    def remove_factors_from(self, factors):
        for factor in factors:
            self.factors.remove(factor)

    def get_factor(self, name):
        for factor in self.factors:
            if factor.name == name:
                return factor
    def get_factors_from(self, names):
        factors = []
        for factor in self.factors:
            if factor.name in names:
                factors.append(factor)
        return factors

    '''
    GM related operations
    '''

    def copy(self):
        return GraphicalModel(copy(self.variables), [factor.copy() for factor in self.factors])
    def summary(self):
        print(np.max([self.get_cardinality_for_(var) for var in self.variables]))
        print(np.max([len(fac.variables) for fac in self.factors]))
        print(len([fac for fac in self.factors if len(fac.variables)>1]))

    def display_factors(self):
        for fac in self.factors:
            print(fac)

def check_forney(gm):
    for variable in gm.variables:
        if gm.degree(variable) != 2:
            return False
    return True
'''
    def flatten_variables(self, variables):
        new_variable = reduce(lambda x,y: str(x)+str(y), variables)

        for factor in self.factors:
            if set(variables) < set(factor.variables):
                factor.flatten(variables, new_variable = new_variable)

        for variable in variables:
            self.remove_variable(variable)
        self.add_variable(new_variable)

        return new_variable

    def unflatten_variable(self, variable, new_variables, cardinalities):
        for factor in self.factors:
            if variable in factor.variables:
                factor.unflatten(variable, cardinalities, new_variables = new_variables)
        self.remove_variable(variable)
        for new_variable in new_variables:
            self.add_variable(new_variable)
'''
