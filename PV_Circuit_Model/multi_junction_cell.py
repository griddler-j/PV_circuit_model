import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
import numbers

class MultiJunctionCell(CircuitGroup):
    _type_number = 7
    def __init__(self,subcells=None,subgroups=None,Rs=0.1,location=None,
                 rotation=0,name=None,temperature=25,Suns=1.0):
        if subgroups is not None:
            components = subgroups
        else:
            components = subcells
            components.append(Resistor(cond=subcells[0].area/Rs))
        super().__init__(components, connection="series",location=location,rotation=rotation,
                         name=name,extent=components[0].extent)
        self.cells = []
        self.series_resistor = None
        for item in self.subgroups:
            if isinstance(item,Cell):
                self.cells.append(item)
            elif isinstance(item,Resistor):
                self.series_resistor = item
        self.area = self.cells[0].area
        if self.series_resistor is not None:
            self.series_resistor.aux["area"] = self.area
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

    @classmethod
    def from_circuitgroup(cls, comp, **kwargs):
        total_Rs = 0
        subcells = []
        for item in comp.subgroups:
            if isinstance(item,Cell):
                if item.series_resistor is not None:
                    total_Rs += 1/(item.series_resistor.cond*item.area)
                    subcells.append(item.diode_branch.as_cell(area=item.area,
                        location=item.location,rotation=item.rotation,shape=item.shape,name=item.name,
                        temperature=item.temperature,Suns=item.Suns))
                else:
                    subcells.append(item.clone())
            elif isinstance(item,Resistor):
                total_Rs += 1/item.cond
            else:
                raise NotImplementedError
        return cls(subcells=subcells,Rs=total_Rs, **kwargs)
