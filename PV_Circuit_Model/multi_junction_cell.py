import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *

class MultiJunctionCell(CircuitGroup):
    def __init__(self,subcells,Rs=0.1,location=None,
                 rotation=0,name=None,temperature=25,Suns=1.0):
        self.area = subcells[0].area
        self.cells = subcells
        self.is_multi_junction_cell = True
        if Rs > 0:
            series_resistor = Resistor(cond=self.area/Rs)
            series_resistor.aux["area"] = self.area
            self.series_resistor = series_resistor
            components = subcells + [series_resistor]
        else:
            self.series_resistor = None
        super().__init__(components, connection="series",location=location,rotation=rotation,
                         name=name,extent=subcells[0].extent)
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        self.set_Suns(Suns)     
    def set_Suns(self,Suns):
        if isinstance(Suns,numbers.Number):
            Suns = [Suns]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_Suns(Suns=Suns[i])
    def set_JL(self,JL,Suns=1.0,temperature=25):
        if isinstance(JL,numbers.Number):
            JL = [JL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_JL(JL[i], Suns=Suns, temperature=temperature)
    def set_IL(self,IL,Suns=1.0,temperature=25):
        if isinstance(IL,numbers.Number):
            IL = [IL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_IL(IL[i], Suns=Suns, temperature=temperature)
    def set_temperature(self,temperature):
        super().set_temperature(temperature)
        self.temperature = temperature
    def specific_Rs_cond(self):
        if self.series_resistor is None:
            return np.inf
        return self.series_resistor.cond/self.area
    def Rs_cond(self):
        return self.specific_Rs_cond()*self.area
    def specific_Rs(self):
        return 1/self.specific_Rs_cond()
    def Rs(self):
        return 1/self.Rs_cond()
    def set_specific_Rs_cond(self,cond):
        if self.series_resistor is not None:
            self.series_resistor.set_cond(cond*self.area)
    def set_Rs_cond(self,cond):
        self.set_specific_Rs_cond(cond)
    def set_specific_Rs(self,Rs):
        self.set_specific_Rs_cond(1/Rs)
    def set_Rs(self,Rs):
        self.set_Rs_cond(1/Rs)
    