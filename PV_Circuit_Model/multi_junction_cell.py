import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class MultiJunctionCell(CircuitGroup):
    def __init__(self,subcells,Rs=0.1,connection="series",location=None,
                 rotation=0,name=None,temperature=25,Suns=1.0):
        self.area = subcells[0].area
        self.cells = subcells
        if Rs > 0:
            series_resistor = Resistor(cond=self.area/Rs)
            series_resistor.aux["area"] = self.area
            self.series_resistor = series_resistor
            components = subcells + [series_resistor]
        else:
            self.series_resistor = None
        super().__init__(components, connection,location=location,rotation=rotation,
                         name=name,extent=subcells[0].extent)
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        self.set_Suns(Suns)     
    def set_Suns(self,Suns, rebuild_IV=True):
        if isinstance(Suns,numbers.Number):
            Suns = [Suns]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_Suns(Suns=Suns[i], rebuild_IV=False)
        if rebuild_IV:
            self.build_IV()
    def set_JL(self,JL,Suns=1.0,temperature=25,rebuild_IV=True):
        if isinstance(JL,numbers.Number):
            JL = [JL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_JL(JL[i], Suns=Suns, temperature=temperature)
        if rebuild_IV:
            self.build_IV()
    def set_IL(self,IL,Suns=1.0,temperature=25,rebuild_IV=True):
        if isinstance(IL,numbers.Number):
            IL = [IL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_IL(IL[i], Suns=Suns, temperature=temperature)
        if rebuild_IV:
            self.build_IV()
    def set_temperature(self,temperature, rebuild_IV=True):
        super().set_temperature(temperature,rebuild_IV=False)
        self.temperature = temperature
        if rebuild_IV:
            self.build_IV()
    def build_IV(self, max_num_points=None, **kwargs):
        super().build_IV(max_num_points=max_num_points)
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
    