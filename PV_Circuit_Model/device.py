import numpy as np
import PV_Circuit_Model.utilities_silicon as silicon
import PV_Circuit_Model.circuit_model as circuit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, Point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
# Backward-compat for old pickles that expect Intrinsic_Si_diode here
from PV_Circuit_Model.circuit_model import Intrinsic_Si_diode  # noqa: F401

class Device(circuit.CircuitGroup):
    """Wrapper class for CircuitGroup

    Devices behave like CircuitGroups but are treated as atomic units during
    + and | operations to prevent subgroup flattening.

    Args:
        subgroups (Sequence[CircuitComponent]): Child components or groups.
        connection (str): "series" or "parallel" connection type.
        location (Optional[Sequence[float]]): XY location for layout.
        rotation (float): Rotation in degrees.
        name (Optional[str]): Optional name for lookup.

    Returns:
        Device: The constructed device.

    Example:
        ```python
        from PV_Circuit_Model.device import Device
        from PV_Circuit_Model.circuit_model import R
        device = Device([R(1.0), R(2.0)], connection="series")
        ```
    """
    _is_atomic = True
    
    @classmethod
    def from_circuitgroup(cls, comp, **kwargs: Any) -> "Device":
        """Create a Device from an existing CircuitGroup.

        This preserves the subgroup structure and connection type.

        Args:
            comp (CircuitGroup): Source group to wrap.
            **kwargs (Any): Forwarded to Device initialization.

        Returns:
            Device: New device with the same structure.

        Example:
            ```python
            from PV_Circuit_Model.device import Device
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1.0), R(2.0))
            device = Device.from_circuitgroup(group)
            device.connection
            ```
        """
        return cls(comp.subgroups, comp.connection, **kwargs)

def set_Suns(circuit_group: circuit.CircuitGroup, suns: float) -> None:
    """Set irradiance scaling for all current sources in a group.

    This updates each CurrentSource with the new Suns multiplier.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        circuit_group (CircuitGroup): Group to update.
        suns (float): Irradiance multiplier to apply.

    Returns:
        None

    Example:
        ```python
        from PV_Circuit_Model.device import set_Suns, Cell_
        cell = Cell_()
        set_Suns(cell, suns=2.0)
        cell.findElementType("CurrentSource")[0].Suns #2
        ```
    """
    currentSources = circuit_group.findElementType(circuit.CurrentSource)
    for currentSource in currentSources:
        currentSource.changeTemperatureAndSuns(Suns=suns)
circuit.CircuitGroup.set_Suns = set_Suns

class Cell(Device,_type_number=6):
    """Solar cell device built from circuit subgroups.

    Wraps a diode branch and optional series resistor with geometry and
    irradiance/temperature handling.
    """
    photon_coupling_diodes = None
    _dont_serialize = Device._dont_serialize + ("series_resistor","diode_branch","photon_coupling_diodes")
    def __init__(
        self,
        subgroups: Sequence[circuit.CircuitComponent],
        connection: str = "series",
        area: float = 1,
        location: Optional[Sequence[float]] = None,
        rotation: float = 0,
        shape: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        temperature: float = 25,
        Suns: float = 1.0,
    ) -> None:
        """Initialize a solar cell device.

        Computes geometry extents and prepares diode branches and sources.

        Args:
            subgroups (Sequence[CircuitComponent]): Child components for the cell.
            connection (str): "series" or "parallel" connection type.
            area (float): Active area multiplier.
            location (Optional[Sequence[float]]): XY location for layout.
            rotation (float): Rotation in degrees.
            shape (Optional[np.ndarray]): Polygon vertices for layout.
            name (Optional[str]): Optional name for lookup.
            temperature (float): Temperature in Celsius.
            Suns (float): Irradiance multiplier.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.area
            ```
        """
        x_extent = 0.0
        y_extent = 0.0
        if shape is not None:
            x_extent = np.max(shape[:,0])-np.min(shape[:,0])
            y_extent = np.max(shape[:,1])-np.min(shape[:,1])
        super().__init__(subgroups, connection,location=location,rotation=rotation,
                         name=name,extent=np.array([x_extent,y_extent]).astype(float))
        self.area = area
        self.shape = shape
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        Cell.__compile__(self)

    def __post_init__(self):
        super().__post_init__()
        Cell.__compile__(self)
        
    def __compile__(self):
        self.photon_coupling_diodes = self.findElementType(circuit.PhotonCouplingDiode)
        if self.connection=="series":
            for branch in self.subgroups:
                if isinstance(branch,circuit.Resistor):
                    self.series_resistor = branch
                else:
                    self.diode_branch = branch
        else:
            self.series_resistor = None
            self.diode_branch = self
        if self.max_I is not None:
            self.max_I *= self.area
        if not hasattr(self,"shape"): # some legacy bsons don't store shape
            self.shape = None

    # a weak copy, only the parameters
    def copy_values(self, cell2: "Cell") -> None:
        """Copy temperature and diode branch parameters from another cell.

        This performs a shallow parameter copy without altering topology.

        Args:
            cell2 (Cell): Source cell with matching structure.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            a = make_solar_cell()
            b = make_solar_cell()
            a.copy_values(b)
            a.temperature == b.temperature
            ```
        """
        self.temperature = cell2.temperature
        self.Suns = cell2.Suns
        if self.series_resistor is not None:
            self.series_resistor.copy_values(cell2.series_resistor)
        for i, element in enumerate(self.diode_branch.subgroups):
            if i < len(cell2.diode_branch.subgroups) and type(element) is type(cell2.diode_branch.subgroups[i]): # noqa: E721
                element.copy_values(cell2.diode_branch.subgroups[i])

    def set_Suns(self, Suns: float) -> None:
        """Set irradiance multiplier and update current sources.

        This updates CurrentSource scaling and caches.

        Args:
            Suns (float): Irradiance multiplier.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_Suns(2.0)
            cell.Suns
            ```
        """
        self.Suns = Suns
        currentSources = self.findElementType(circuit.CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(Suns=Suns)
    
    def set_temperature(self, temperature: float) -> None:
        """Set temperature for the cell and its components.

        This updates diode thermal voltages and currents.

        Args:
            temperature (float): Temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_temperature(35)
            cell.temperature
            ```
        """
        super().set_temperature(temperature)
        self.temperature = temperature

    def JL(self) -> float:
        """Return light-generated current density (J_L).

        This sums the IL values of all CurrentSource elements.

        Args:
            None

        Returns:
            float: Current density (A/cm^2 equivalent).

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.JL()
            ```
        """
        JL = 0.0
        currentSources = self.findElementType(circuit.CurrentSource)
        for currentSource in currentSources:
            JL += currentSource.IL
        return JL

    def IL(self) -> float:
        """Return total light-generated current (I_L).

        This scales current density by the cell area.

        Args:
            None

        Returns:
            float: Total current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.IL()
            ```
        """
        return self.JL()*self.area     
    
    def set_JL(self, JL: float, Suns: float = 1.0, temperature: float = 25) -> None:
        """Set reference light current density for the primary source.

        This updates the first non-defect CurrentSource reference values.

        Args:
            JL (float): Reference current density.
            Suns (float): Reference irradiance multiplier.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_JL(0.05)
            cell.JL()
            ```
        """
        currentSources = self.findElementType(circuit.CurrentSource)
        for currentSource in currentSources:
            if currentSource.tag != "defect":
                currentSource.refSuns = Suns
                currentSource.refIL = JL
                currentSource.refT = temperature
                currentSource.changeTemperatureAndSuns(
                    temperature=self.temperature,Suns=self.Suns)
                break

    def set_IL(self, IL: float, Suns: float = 1.0, temperature: float = 25) -> None:
        """Set reference light current using total current.

        This converts total current to current density using area.

        Args:
            IL (float): Total light current.
            Suns (float): Reference irradiance multiplier.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.set_IL(0.2)
            cell.IL()
            ```
        """
        self.set_JL(IL/self.area,Suns=Suns,temperature=temperature)
    
    def J0(self, n: float) -> float:
        """Return saturation current density for diodes with factor n.

        Excludes intrinsic silicon and photon coupling diodes.

        Args:
            n (float): Ideality factor.

        Returns:
            float: Saturation current density.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.J0(n=1)
            ```
        """
        J0 = 0.0
        diodes = self.findElementType(circuit.ForwardDiode)
        for diode in diodes:
            if not isinstance(diode,circuit.Intrinsic_Si_diode) and not isinstance(diode,circuit.PhotonCouplingDiode):
                if diode.n==n:
                    J0 += diode.I0
        return J0
    def J01(self) -> float:
        """Return saturation current density for n=1 diodes.

        This is a convenience wrapper around `J0(n=1)`.

        Args:
            None

        Returns:
            float: Saturation current density.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.J01()
            ```
        """
        return self.J0(n=1)
    def J02(self) -> float:
        """Return saturation current density for n=2 diodes.

        This is a convenience wrapper around `J0(n=2)`.

        Args:
            None

        Returns:
            float: Saturation current density.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.J02()
            ```
        """
        return self.J0(n=2)     
    
    def PC_J0(self, n: float) -> float:
        """Return photon-coupling diode saturation current density.

        This sums photon coupling diodes with the given ideality factor.

        Args:
            n (float): Ideality factor.

        Returns:
            float: Saturation current density.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.PC_J0(n=1)
            ```
        """
        J0 = 0.0
        diodes = self.findElementType(circuit.PhotonCouplingDiode)
        for diode in diodes:
            if diode.n==n:
                J0 += diode.I0
        return J0
    def PC_J01(self) -> float:
        """Return photon-coupling diode saturation current density for n=1.

        This is a convenience wrapper around `PC_J0(n=1)`.

        Args:
            None

        Returns:
            float: Saturation current density.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.PC_J01()
            ```
        """
        return self.PC_J0(n=1)
    def PC_I0(self, n: float) -> float:
        """Return photon-coupling diode saturation current (area-scaled).

        This scales the current density by the cell area.

        Args:
            n (float): Ideality factor.

        Returns:
            float: Saturation current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.PC_I0(n=1)
            ```
        """
        return self.PC_J0(n)*self.area
    def PC_I01(self) -> float:
        """Return photon-coupling diode saturation current for n=1.

        This is a convenience wrapper around `PC_I0(n=1)`.

        Args:
            None

        Returns:
            float: Saturation current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.PC_I01()
            ```
        """
        return self.PC_I0(n=1)
    
    def I0(self, n: float) -> float:
        """Return saturation current for diodes with factor n.

        This scales the current density by the cell area.

        Args:
            n (float): Ideality factor.

        Returns:
            float: Saturation current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.I0(n=1)
            ```
        """
        return self.J0(n)*self.area
    def I01(self) -> float:
        """Return saturation current for n=1 diodes.

        This is a convenience wrapper around `I0(n=1)`.

        Args:
            None

        Returns:
            float: Saturation current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.I01()
            ```
        """
        return self.I0(n=1)
    def I02(self) -> float:
        """Return saturation current for n=2 diodes.

        This is a convenience wrapper around `I0(n=2)`.

        Args:
            None

        Returns:
            float: Saturation current in amperes.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.I02()
            ```
        """
        return self.I0(n=2)    
    
    def set_J0(self, J0: float, n: float, temperature: float = 25) -> None:
        """Set saturation current density for the first matching diode.

        This updates reference values and rescales for temperature.

        Args:
            J0 (float): Saturation current density.
            n (float): Ideality factor.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_J0(1e-14, n=1)
            cell.J01()
            ```
        """
        diodes = self.findElementType(circuit.ForwardDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,circuit.Intrinsic_Si_diode) and diode.n==n and not isinstance(diode,circuit.PhotonCouplingDiode):
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature)
                break
    def set_J01(self, J0: float, temperature: float = 25) -> None:
        """Set saturation current density for n=1 diodes.

        This is a convenience wrapper around `set_J0`.

        Args:
            J0 (float): Saturation current density.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_J01(1e-14)
            cell.J01()
            ```
        """
        self.set_J0(J0,n=1,temperature=temperature)
    def set_J02(self, J0: float, temperature: float = 25) -> None:
        """Set saturation current density for n=2 diodes.

        This is a convenience wrapper around `set_J0`.

        Args:
            J0 (float): Saturation current density.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_J02(1e-9)
            cell.J02()
            ```
        """
        self.set_J0(J0,n=2,temperature=temperature)

    def set_I0(self, I0: float, n: float, temperature: float = 25) -> None:
        """Set saturation current for the first matching diode.

        This converts to current density before calling `set_J0`.

        Args:
            I0 (float): Saturation current in amperes.
            n (float): Ideality factor.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.set_I0(2e-14, n=1)
            cell.I01()
            ```
        """
        self.set_J0(I0/self.area, n=n, temperature=temperature)
    def set_I01(self, I0: float, temperature: float = 25) -> None:
        """Set saturation current for n=1 diodes.

        This is a convenience wrapper around `set_I0`.

        Args:
            I0 (float): Saturation current in amperes.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.set_I01(2e-14)
            cell.I01()
            ```
        """
        self.set_I0(I0,n=1,temperature=temperature)
    def set_I02(self, I0: float, temperature: float = 25) -> None:
        """Set saturation current for n=2 diodes.

        This is a convenience wrapper around `set_I0`.

        Args:
            I0 (float): Saturation current in amperes.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(area=2.0)
            cell.set_I02(2e-9)
            cell.I02()
            ```
        """
        self.set_I0(I0,n=2,temperature=temperature)

    def set_PC_J0(self, J0: float, n: float, temperature: float = 25) -> None:
        """Set photon-coupling diode saturation current density.

        This updates the first matching photon-coupling diode.

        Args:
            J0 (float): Saturation current density.
            n (float): Ideality factor.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(J01_photon_coupling=1e-14)
            cell.set_PC_J0(1e-14, n=1)
            cell.PC_J01()
            ```
        """
        diodes = self.findElementType(circuit.PhotonCouplingDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,circuit.Intrinsic_Si_diode) and diode.n==n:
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature)
                break
    def set_PC_J01(self, J0: float, temperature: float = 25) -> None:
        """Set photon-coupling diode saturation current density for n=1.

        This is a convenience wrapper around `set_PC_J0`.

        Args:
            J0 (float): Saturation current density.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(J01_photon_coupling=1e-14)
            cell.set_PC_J01(1e-14)
            cell.PC_J01()
            ```
        """
        self.set_PC_J0(J0,n=1,temperature=temperature)
    def set_PC_I0(self, I0: float, n: float, temperature: float = 25) -> None:
        """Set photon-coupling diode saturation current.

        This converts to current density before calling `set_PC_J0`.

        Args:
            I0 (float): Saturation current in amperes.
            n (float): Ideality factor.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(J01_photon_coupling=1e-14, area=2.0)
            cell.set_PC_I0(2e-14, n=1)
            cell.PC_I01()
            ```
        """
        self.set_PC_J0(I0/self.area, n=n, temperature=temperature)
    def set_PC_I01(self, I0: float, temperature: float = 25) -> None:
        """Set photon-coupling diode saturation current for n=1.

        This is a convenience wrapper around `set_PC_I0`.

        Args:
            I0 (float): Saturation current in amperes.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(J01_photon_coupling=1e-14, area=2.0)
            cell.set_PC_I01(2e-14)
            cell.PC_I01()
            ```
        """
        self.set_PC_I0(I0,n=1,temperature=temperature)
    
    def specific_Rs_cond(self) -> float:
        """Return specific series conductance.

        Uses the configured series resistor when present.

        Args:
            None

        Returns:
            float: Specific series conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1)
            cell.specific_Rs_cond()
            ```
        """
        if self.series_resistor is None:
            return np.inf
        return self.series_resistor.cond
    def Rs_cond(self) -> float:
        """Return series conductance scaled by area.

        This scales the specific conductance by cell area.

        Args:
            None

        Returns:
            float: Series conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1, area=2.0)
            cell.Rs_cond()
            ```
        """
        return self.specific_Rs_cond()/self.area
    def specific_Rs(self) -> float:
        """Return specific series resistance.

        This is the inverse of specific series conductance.

        Args:
            None

        Returns:
            float: Specific series resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1)
            cell.specific_Rs()
            ```
        """
        return 1/self.specific_Rs_cond()
    def Rs(self) -> float:
        """Return series resistance scaled by area.

        This is the inverse of series conductance.

        Args:
            None

        Returns:
            float: Series resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1, area=2.0)
            cell.Rs()
            ```
        """
        return 1/self.Rs_cond()
    
    def set_rev_breakdown_V(self, V: float) -> None:
        """Set reverse breakdown voltage for the reverse diode.

        This updates the reverse diode and invalidates IV caches.

        Args:
            V (float): Breakdown voltage shift.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_rev_breakdown_V(5.0)
            ```
        """
        reverse_diode = self.diode_branch.findElementType(circuit.ReverseDiode)[0]
        reverse_diode.V_shift = V
        reverse_diode.null_IV()

    def set_specific_Rs_cond(self, cond: float) -> None:
        """Set specific series conductance.

        Updates the series resistor if present.

        Args:
            cond (float): Specific series conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1)
            cell.set_specific_Rs_cond(20.0)
            cell.specific_Rs_cond()
            ```
        """
        if self.series_resistor is not None:
            self.series_resistor.set_cond(cond)
    def set_Rs_cond(self, cond: float) -> None:
        """Set series conductance scaled by area.

        This converts to specific conductance before updating.

        Args:
            cond (float): Series conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1, area=2.0)
            cell.set_Rs_cond(10.0)
            cell.Rs_cond()
            ```
        """
        self.set_specific_Rs_cond(cond/self.area)
    def set_specific_Rs(self, Rs: float) -> None:
        """Set specific series resistance.

        This converts to conductance before updating.

        Args:
            Rs (float): Specific series resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1)
            cell.set_specific_Rs(0.05)
            cell.specific_Rs()
            ```
        """
        self.set_specific_Rs_cond(1/Rs)
    def set_Rs(self, Rs: float) -> None:
        """Set series resistance scaled by area.

        This converts to specific resistance before updating.

        Args:
            Rs (float): Series resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rs=0.1, area=2.0)
            cell.set_Rs(0.2)
            cell.Rs()
            ```
        """
        self.set_specific_Rs(Rs*self.area)
    
    def specific_shunt_cond(self) -> float:
        """Return specific shunt conductance.

        This sums shunt resistors in the diode branch.

        Args:
            None

        Returns:
            float: Specific shunt conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6)
            cell.specific_shunt_cond()
            ```
        """
        Rsh_cond = 0.0
        shunt_resistors = self.diode_branch.findElementType(circuit.Resistor)
        for res in shunt_resistors:
            Rsh_cond += res.cond
        return Rsh_cond
    def shunt_cond(self) -> float:
        """Return shunt conductance scaled by area.

        This scales the specific shunt conductance by area.

        Args:
            None

        Returns:
            float: Shunt conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6, area=2.0)
            cell.shunt_cond()
            ```
        """
        return self.specific_shunt_cond()/self.area
    def specific_shunt_res(self) -> float:
        """Return specific shunt resistance.

        This is the inverse of specific shunt conductance.

        Args:
            None

        Returns:
            float: Specific shunt resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6)
            cell.specific_shunt_res()
            ```
        """
        return 1/self.specific_shunt_cond()
    def shunt_res(self) -> float:
        """Return shunt resistance scaled by area.

        This is the inverse of shunt conductance.

        Args:
            None

        Returns:
            float: Shunt resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6, area=2.0)
            cell.shunt_res()
            ```
        """
        return 1/self.shunt_cond()
    
    def set_specific_shunt_cond(self, cond: float) -> None:
        """Set specific shunt conductance on the first non-defect resistor.

        Only the first matching resistor is updated.

        Args:
            cond (float): Specific shunt conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6)
            cell.set_specific_shunt_cond(1e-5)
            cell.specific_shunt_cond()
            ```
        """
        shunt_resistors = self.diode_branch.findElementType(circuit.Resistor)
        for res in shunt_resistors:
            if res.tag != "defect":
                res.set_cond(cond)
                break
    def set_shunt_cond(self, cond: float) -> None:
        """Set shunt conductance scaled by area.

        This converts to specific conductance before updating.

        Args:
            cond (float): Shunt conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6, area=2.0)
            cell.set_shunt_cond(1e-5)
            cell.shunt_cond()
            ```
        """
        self.set_specific_shunt_cond(cond/self.area)
    def set_specific_shunt_res(self, Rsh: float) -> None:
        """Set specific shunt resistance.

        This converts to conductance before updating.

        Args:
            Rsh (float): Specific shunt resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6)
            cell.set_specific_shunt_res(2e6)
            cell.specific_shunt_res()
            ```
        """
        self.set_specific_shunt_cond(1/Rsh)
    def set_shunt_res(self, Rsh: float) -> None:
        """Set shunt resistance scaled by area.

        This converts to specific resistance before updating.

        Args:
            Rsh (float): Shunt resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell(Rshunt=1e6, area=2.0)
            cell.set_shunt_res(2e6)
            cell.shunt_res()
            ```
        """
        self.set_specific_shunt_res(Rsh*self.area)
    def set_shape(self, wafer_format: str = "M10", half_cut: bool = True) -> None:
        """Update the cell shape and area from a wafer format.

        This updates both the polygon and the area attribute.

        Args:
            wafer_format (str): Format key in `wafer_formats`.
            half_cut (bool): If True, use half-cut geometry.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell
            cell = make_solar_cell()
            cell.set_shape(wafer_format="M10", half_cut=True)
            cell.shape is not None
            ```
        """
        shape_area = wafer_shape(format=wafer_format,half_cut=half_cut)
        self.shape, self.area = shape_area["shape"], shape_area["area"]

    @classmethod
    def from_circuitgroup(cls, comp: circuit.CircuitGroup, **kwargs: Any) -> "Cell":
        """Create a Cell from an existing CircuitGroup.

        This preserves subgroup structure and connection type.

        Args:
            comp (CircuitGroup): Source group to wrap.
            **kwargs (Any): Forwarded to Cell initialization.

        Returns:
            Cell: New cell with the same structure.

        Example:
            ```python
            from PV_Circuit_Model.device import Cell
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1.0), R(2.0))
            cell = Cell.from_circuitgroup(group, area=1.0)
            ```
        """
        return cls(comp.subgroups,comp.connection, **kwargs)
    

class Module(Device):
    """Module composed of multiple cells or cell groups.
    """
    _dont_serialize = Device._dont_serialize + ("interconnect_resistors","interconnect_conds","cells")
    def __init__(
        self,
        subgroups: Sequence[circuit.CircuitComponent],
        connection: str = "series",
        location: Optional[Sequence[float]] = None,
        rotation: float = 0,
        name: Optional[str] = None,
        temperature: float = 25,
        Suns: float = 1.0,
    ) -> None:
        """Initialize a module.

        Collects cell references and applies initial conditions.

        Args:
            subgroups (Sequence[CircuitComponent]): Child components/groups.
            connection (str): "series" or "parallel" connection type.
            location (Optional[Sequence[float]]): XY location for layout.
            rotation (float): Rotation in degrees.
            name (Optional[str]): Optional name for lookup.
            temperature (float): Temperature in Celsius.
            Suns (float): Irradiance multiplier.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell, Module
            module = Module([make_solar_cell() for _ in range(60)], connection="series")
            module.connection
            ```
        """
        super().__init__(subgroups, connection,location=location,rotation=rotation,name=name)
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns 
        Module.__compile__(self)

    def __post_init__(self):
        super().__post_init__()
        Module.__compile__(self)

    def __compile__(self):
        self.interconnect_resistors = self.findElementType(circuit.Resistor, Cell)
        self.interconnect_conds = []
        for r in self.interconnect_resistors:
            self.interconnect_conds.append(r.cond)
        self.cells = self.findElementType(Cell)
    
    def set_Suns(self, Suns: float) -> None:
        """Set irradiance on all cells in the module.

        This delegates to each Cell's `set_Suns` method.

        Args:
            Suns (float): Irradiance multiplier.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell, Module
            module = Module([make_solar_cell() for _ in range(60)], connection="series")
            module.set_Suns(2.0)
            module.cells[0].Suns
            ```
        """
        for cell in self.cells:
            cell.set_Suns(Suns=Suns)
    def set_temperature(self, temperature: float) -> None:
        """Set temperature on all elements in the module.

        This updates diode parameters and cached values.

        Args:
            temperature (float): Temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell, Module
            module = Module([make_solar_cell() for _ in range(60)], connection="series")
            module.set_temperature(35)
            module.temperature
            ```
        """
        super().set_temperature(temperature)
        self.temperature = temperature

    @classmethod
    def from_circuitgroup(cls, comp: circuit.CircuitGroup, **kwargs: Any) -> "Module":
        """Create a Module from an existing CircuitGroup.

        This preserves subgroup structure and connection type.

        Args:
            comp (CircuitGroup): Source group to wrap.
            **kwargs (Any): Forwarded to Module initialization.

        Returns:
            Module: New module with the same structure.

        Example:
            ```python
            from PV_Circuit_Model.device import make_solar_cell, Module
            cell = make_solar_cell()
            circuit_group = cell*60
            from PV_Circuit_Model.device import Module
            module = Module.from_circuitgroup(circuit_group)
            ```
        """
        return cls(comp.subgroups,comp.connection, **kwargs)
    def set_interconnect_resistors(
        self: circuit.CircuitGroup,
        interconnect_conds: list[float] | float | None = None
    ) -> None:
        """set half-string resistors conductance.

        Uses the module's `aux["interconnect_conds"]` when available.

        Args:
            interconnect_conds (list[float] | float | None): conductance value(s) to apply.

        Example:
            ```python
            from PV_Circuit_Model.device import make_module, make_solar_cell
            cells = [make_solar_cell() for _ in range(60)]
            module = make_module(cells, num_strings=3, num_cells_per_halfstring=20)
            module.set_interconnect_resistors()
            ```
        """
        if interconnect_conds is None:
            interconnect_conds = self.interconnect_conds
        elif isinstance(interconnect_conds,numbers.Number):
            interconnect_conds = [interconnect_conds for _ in range(len(self.interconnect_resistors))]
        for i, r in enumerate(self.interconnect_resistors):
            r.set_cond(interconnect_conds[i])


class ByPassDiode(circuit.ReverseDiode):
    """Reverse diode used as a module bypass element.

    Args:
        I0 (float): Saturation current.
        n (float): Ideality factor.
        V_shift (float): Breakdown voltage shift.
        max_I (Optional[float]): Maximum current density for IV ranges.
        tag (Optional[str]): Optional identifier.
        temperature (float): Temperature in Celsius.

    Returns:
        ByPassDiode: The constructed bypass diode.

    Example:
        ```python
        from PV_Circuit_Model.device import ByPassDiode
        diode = ByPassDiode(I0=1e-12, n=1.0)
        ```
    """
    max_I = None

# simplified initializer
def Dbypass(*args, **kwargs):
    """
    This is a shorthand constructor for :class:`ByPassDiode`.
    """
    return ByPassDiode(*args, **kwargs)

class MultiJunctionCell(Device):
    """Stacked multi-junction cell with optional series resistance.

    Combines multiple Cell instances in series with shared geometry.
    """
    _dont_serialize = Device._dont_serialize + ("series_resistor","cells")
    def __init__(
        self,
        subcells: Optional[List[Cell]] = None,
        Rs: float = 0.1,
        location: Optional[Sequence[float]] = None,
        rotation: float = 0,
        name: Optional[str] = None,
        temperature: float = 25,
        Suns: Union[float, Sequence[float]] = 1.0,
    ) -> None:
        """Initialize a multi-junction cell.

        Builds subgroup structure from subcells when not provided.

        Args:
            subcells (Optional[List[Cell]]): List of cell subcomponents.
            Rs (float): Specific series resistance used when building subgroups.
            location (Optional[Sequence[float]]): XY location for layout.
            rotation (float): Rotation in degrees.
            name (Optional[str]): Optional name for lookup.
            temperature (float): Temperature in Celsius.
            Suns (Union[float, Sequence[float]]): Irradiance multiplier(s).

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            ```
        """
        components = subcells
        components.append(circuit.Resistor(cond=subcells[0].area/Rs))
        super().__init__(components, connection="series",location=location,rotation=rotation,
                         name=name,extent=components[0].extent)
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        MultiJunctionCell.__compile__(self)  

    def __post_init__(self):
        super().__post_init__()
        MultiJunctionCell.__compile__(self)

    def __compile__(self):
        self.cells = []
        self.series_resistor = None
        for item in self.subgroups:
            if isinstance(item,Cell):
                self.cells.append(item)
            elif isinstance(item,circuit.Resistor):
                self.series_resistor = item
        self.area = self.cells[0].area
        if self.series_resistor is not None:
            self.series_resistor.aux["area"] = self.area
    
    def set_Suns(self, Suns: Union[float, Sequence[float]]) -> None:
        """Set irradiance on each subcell.

        Accepts either a single value or per-cell list.

        Args:
            Suns (Union[float, Sequence[float]]): Single value or per-cell list.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_Suns(2.0)
            ```
        """
        if isinstance(Suns,numbers.Number):
            Suns = [Suns]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_Suns(Suns=Suns[i])
    def set_JL(self, JL: Union[float, Sequence[float]], Suns: float = 1.0, temperature: float = 25) -> None:
        """Set reference current density for each subcell.

        Accepts either a single value or per-cell list.

        Args:
            JL (Union[float, Sequence[float]]): Single value or per-cell list.
            Suns (float): Reference irradiance multiplier.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_JL(0.05)
            ```
        """
        if isinstance(JL,numbers.Number):
            JL = [JL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_JL(JL[i], Suns=Suns, temperature=temperature)
    def set_IL(self, IL: Union[float, Sequence[float]], Suns: float = 1.0, temperature: float = 25) -> None:
        """Set reference current for each subcell.

        Accepts either a single value or per-cell list.

        Args:
            IL (Union[float, Sequence[float]]): Single value or per-cell list.
            Suns (float): Reference irradiance multiplier.
            temperature (float): Reference temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_IL(0.1)
            ```
        """
        if isinstance(IL,numbers.Number):
            IL = [IL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_IL(IL[i], Suns=Suns, temperature=temperature)
    def set_temperature(self, temperature: float) -> None:
        """Set temperature for all subcells.

        This updates diode thermal voltages and sources.

        Args:
            temperature (float): Temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_temperature(35)
            mj.temperature
            ```
        """
        super().set_temperature(temperature)
        self.temperature = temperature
    def specific_Rs_cond(self) -> float:
        """Return specific series conductance.

        Uses the series resistor when present.

        Args:
            None

        Returns:
            float: Specific series conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.specific_Rs_cond()
            ```
        """
        if self.series_resistor is None:
            return np.inf
        return self.series_resistor.cond/self.area
    def Rs_cond(self) -> float:
        """Return series conductance scaled by area.

        This scales the specific conductance by area.

        Args:
            None

        Returns:
            float: Series conductance.

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.Rs_cond()
            ```
        """
        return self.specific_Rs_cond()*self.area
    def specific_Rs(self) -> float:
        """Return specific series resistance.

        This is the inverse of specific series conductance.

        Args:
            None

        Returns:
            float: Specific series resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.specific_Rs()
            ```
        """
        return 1/self.specific_Rs_cond()
    def Rs(self) -> float:
        """Return series resistance scaled by area.

        This is the inverse of series conductance.

        Args:
            None

        Returns:
            float: Series resistance.

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.Rs()
            ```
        """
        return 1/self.Rs_cond()
    def set_specific_Rs_cond(self, cond: float) -> None:
        """Set specific series conductance.

        Updates the series resistor if present.

        Args:
            cond (float): Specific series conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_specific_Rs_cond(20.0)
            mj.specific_Rs_cond()
            ```
        """
        if self.series_resistor is not None:
            self.series_resistor.set_cond(cond*self.area)
    def set_Rs_cond(self, cond: float) -> None:
        """Set series conductance scaled by area.

        This delegates to `set_specific_Rs_cond`.

        Args:
            cond (float): Series conductance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_Rs_cond(10.0)
            mj.Rs_cond()
            ```
        """
        self.set_specific_Rs_cond(cond)
    def set_specific_Rs(self, Rs: float) -> None:
        """Set specific series resistance.

        This converts to conductance before updating.

        Args:
            Rs (float): Specific series resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_specific_Rs(0.05)
            mj.specific_Rs()
            ```
        """
        self.set_specific_Rs_cond(1/Rs)
    def set_Rs(self, Rs: float) -> None:
        """Set series resistance scaled by area.

        This converts to conductance before updating.

        Args:
            Rs (float): Series resistance.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            cell = make_solar_cell()
            mj = MultiJunctionCell(subcells=[cell.clone(),cell.clone()], Rs=0.1)
            mj.set_Rs(0.2)
            mj.Rs()
            ```
        """
        self.set_Rs_cond(1/Rs)

    @classmethod
    def from_circuitgroup(cls, comp: circuit.CircuitGroup, **kwargs: Any) -> "MultiJunctionCell":
        """Create a MultiJunctionCell from a CircuitGroup.

        Computes effective series resistance when possible.

        Args:
            comp (CircuitGroup): Source group to wrap.
            **kwargs (Any): Forwarded to MultiJunctionCell initialization.

        Returns:
            MultiJunctionCell: New multi-junction cell instance.

        Raises:
            NotImplementedError: If connection is not "series" or types mismatch.

        Example:
            ```python
            from PV_Circuit_Model.device import MultiJunctionCell, make_solar_cell
            from PV_Circuit_Model.circuit_model import series, R
            cell = make_solar_cell()
            group = cell*3 + R(0.1)
            mj = MultiJunctionCell.from_circuitgroup(group)
            ```
        """
        total_Rs = 0
        cell_area = -1
        subcells = []
        if comp.connection != "series":
            raise NotImplementedError
        for item in comp.subgroups:
            if isinstance(item,Cell):
                if cell_area < 0:
                        cell_area = item.area
                subcells.append(item)
            elif isinstance(item,circuit.Resistor):
                total_Rs += 1/item.cond
            else:
                raise NotImplementedError
        total_Rs *= cell_area # actually input a specific Rs
        if "Rs" not in kwargs and total_Rs > 0:
            kwargs["Rs"] = total_Rs
        return cls(subcells=subcells,**kwargs)

# colormap: choose between cm.magma, inferno, plasma, cividis, viridis, turbo, gray
def draw_cells(
    self: Union[circuit.CircuitGroup, "Cell", List[Any]],
    display: bool = True,
    show_names: bool = False,
    colour_bar: bool = False,
    colour_what: Optional[str] = "Vint",
    show_module_names: bool = False,
    fontsize: int = 9,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    title: str = "Cells Layout",
    colormap: Any = cm.plasma,
) -> Tuple[List[np.ndarray], List[Any], List[float], List[float], List[float]]:
    """Draw cell polygons with optional color mapping.

    Supports individual cells, circuit groups, or lists of devices. When
    `display=False`, it returns geometry and value arrays without plotting.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        self (Union[CircuitGroup, Cell, List[Any]]): Target object(s) to draw.
        display (bool): If True, render a matplotlib figure.
        show_names (bool): If True, annotate each cell with its name.
        colour_bar (bool): If True, show a colorbar when plotting.
        colour_what (Optional[str]): Metric to color by (e.g., "Vint").
        show_module_names (bool): If True, annotate module names.
        fontsize (int): Font size for labels.
        min_value (Optional[float]): Min color scale value.
        max_value (Optional[float]): Max color scale value.
        title (str): Figure title.
        colormap (Any): Matplotlib colormap.

    Returns:
        Tuple[List[np.ndarray], List[Any], List[float], List[float], List[float]]:
            shapes, names, Vints, EL_Vints, Is.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import Module_
        module = Module_()
        _ = draw_cells(module, display=False)
        ```
    """
    if display:
        fig, ax = plt.subplots()
    shapes = []
    names = []
    Vints = []
    EL_Vints = []
    Is = []
    if isinstance(self,list):
        for element in self:
            if hasattr(element,"extent") and element.extent is not None:
                shapes_, names_, Vints_, EL_Vints_, Is_ = element.draw_cells(display=False)
                shapes.extend(shapes_)
                names.extend(names_)
                Vints.extend(Vints_)
                EL_Vints.extend(EL_Vints_)
                Is.extend(Is_)
                if show_module_names and element.name is not None and display:
                    ax.text(element.location[0], element.location[1]+element.extent[1]/2*1.05, element.name, fontsize=fontsize, color='black', ha="center", va="center")
    elif hasattr(self,"shape"): # a solar cell
        shapes.append(self.shape.copy())
        names.append(self.name)
        if self.diode_branch.operating_point is not None:
            Vints.append(self.diode_branch.operating_point[0])
            Is.append(self.operating_point[1])
        if self.aux is not None and "EL_Vint" in self.aux:
            EL_Vints.append(self.aux["EL_Vint"])
    else:
        for element in self.subgroups:
            if hasattr(element,"extent") and element.extent is not None:
                shapes_, names_, Vints_, EL_Vints_, Is_ = element.draw_cells(display=False)
                shapes.extend(shapes_)
                names.extend(names_)
                Vints.extend(Vints_)
                EL_Vints.extend(EL_Vints_)
                Is.extend(Is_)
    has_Vint = False
    has_EL_Vint = False
    has_power = False
    has_aux = False
    norm = None
    vmin = None
    vmax = None
    if len(EL_Vints)==len(shapes) and colour_what=="EL_Vint": # every cell has a EL_Vint
        has_EL_Vint = True
        vmin = min(EL_Vints)
        vmax=max(EL_Vints)
    elif len(Vints)==len(shapes) and colour_what=="Vint": # every cell has a Vint
        has_Vint = True
        vmin=min(Vints)
        vmax=max(Vints)
    elif len(Is)==len(shapes) and colour_what=="power": # every cell has a power
        has_power = True
        powers = np.array(Vints)*np.array(Is)
        vmin=np.min(powers)
        vmax=np.max(powers)
    elif colour_what is not None and len(colour_what)>0 and hasattr(self,"aux") and colour_what in self.aux:
        has_aux = True
        all_aux = self.aux[colour_what]
        vmin=min(all_aux)
        vmax=max(all_aux)
    elif colour_what is not None and len(colour_what)>0 and hasattr(self,"cells") and hasattr(self.cells[0],"aux") and colour_what in self.cells[0].aux:
        has_aux = True
        all_aux = []
        for cell in self.cells:
            all_aux.append(cell.aux[colour_what])
        vmin=min(all_aux)
        vmax=max(all_aux)
    if min_value is not None:
        vmin = max(min_value, vmin)
    if max_value is not None:
        vmax = min(max_value, vmax)
    if vmin is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormap
    
    rotation_ = 0
    x_mirror_ = 1
    y_mirror_ = 1
    location_ = [0,0]
    if not isinstance(self,list):
        rotation_ = self.rotation
        x_mirror_ = self.x_mirror
        y_mirror_ = self.y_mirror
        location_ = self.location
    for i, shape in enumerate(shapes):
        cos = np.cos(np.pi/180*rotation_)
        sin = np.sin(np.pi/180*rotation_)
        new_shape = shape.copy()
        new_shape[:,0] = shape[:,0]*cos + shape[:,1]*sin
        new_shape[:,1] = shape[:,1]*cos - shape[:,0]*sin
        if x_mirror_ == -1:
            new_shape[:,0] *= -1
        if y_mirror_ == -1:
            new_shape[:,1] *= -1
        new_shape[:,0] += location_[0]
        new_shape[:,1] += location_[1]

        shapes[i] = new_shape
    if display:
        for i, shape in enumerate(shapes):
            color = 'gray'
            if has_EL_Vint:
                color = cmap(norm(EL_Vints[i]))
            elif has_Vint:
                color = cmap(norm(Vints[i]))
            elif has_power:
                color = cmap(norm(powers[i]))
            elif has_aux:
                color = cmap(norm(all_aux[i]))
            polygon = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
            x = 0.5*(np.max(shape[:,0])+np.min(shape[:,0]))
            y = 0.5*(np.max(shape[:,1])+np.min(shape[:,1]))
            if show_names:
                ax.text(x, y, names[i], fontsize=fontsize, color='black')
            ax.add_patch(polygon)

        # ---- Tight axes from the actual polygons (fixes big blank space) ----
        xs = np.concatenate([s[:,0] for s in shapes])
        ys = np.concatenate([s[:,1] for s in shapes])
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        w, h = xmax - xmin, ymax - ymin
        pad = 0.05 * max(w, h)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect('equal')
        plt.gcf().canvas.manager.set_window_title(title)

        # 4) Inset colorbar (doesn't shrink the main axes)

        if colour_bar and norm is not None:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            # place the bar just outside the right edge of the axes
            cax = inset_axes(
                ax, width="6%", height="90%", loc="center left",
                bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),  # x offset just to the right
                bbox_transform=ax.transAxes, borderpad=0
            )
            cbar = fig.colorbar(sm, cax=cax)
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            cbar.ax.yaxis.set_major_formatter(fmt)
            cbar.set_label(colour_what)
        else:
            fig.tight_layout()

        plt.show()
    return shapes, names, Vints, EL_Vints, Is
circuit.CircuitGroup.draw_cells = draw_cells

draw_modules = draw_cells

def wafer_shape(
    L: float = 1,
    W: float = 1,
    ingot_center: Optional[Sequence[float]] = None,
    ingot_diameter: Optional[float] = None,
    format: Optional[str] = None,
    half_cut: bool = True,
) -> Dict[str, Any]:
    """Generate a wafer polygon and area.

    Supports rectangular or ingot-trimmed shapes and standard formats.

    Args:
        L (float): Wafer length.
        W (float): Wafer width.
        ingot_center (Optional[Sequence[float]]): Circle center for trimming.
        ingot_diameter (Optional[float]): Circle diameter for trimming.
        format (Optional[str]): Wafer format key from `wafer_formats`.
        half_cut (bool): If True, apply half-cut geometry.

    Returns:
        Dict[str, Any]: Dictionary with keys "shape" (np.ndarray) and "area".

    Example:
        ```python
        from PV_Circuit_Model.device import wafer_shape
        shape_area = wafer_shape(format="M10", half_cut=True)
        ```
    """
    if format is not None and format in silicon.wafer_formats:
        size = silicon.wafer_formats[format]["size"]
        L = size
        W = size
        ingot_diameter = silicon.wafer_formats[format]["diagonal"]
        ingot_center = [0,0]
        if half_cut:
            L = size/2
            ingot_center[1] = -L/2
    rect = np.array([[-W/2,-L/2],[W/2,-L/2],[W/2,L/2],[-W/2,L/2]]) # CCW
    if ingot_center is not None and ingot_diameter is not None:
        ingot_radius = ingot_diameter/2
        circle = Point(ingot_center[0], ingot_center[1]).buffer(ingot_radius, resolution=180)
        rect_poly = Polygon(rect)
        intersection = rect_poly.intersection(circle)
        if intersection.is_empty:
            if ingot_radius < W/2 and ingot_radius < L/2:
                intersection = circle
            else:
                intersection = rect
        elif intersection.geom_type == "Polygon":
            intersection = np.array(intersection.exterior.coords)
        else:
            assert(1==0)
    else:
        intersection = rect
    
    x = intersection[:,0]
    y = intersection[:,1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return {"shape": intersection, "area": area}

# note: always made at 25C 1 Sun
def make_solar_cell(
    Jsc: float = 0.042,
    J01: float = 10e-15,
    J02: float = 2e-9,
    Rshunt: float = 1e6,
    Rs: float = 0.0,
    area: float = 1.0,
    shape: Optional[np.ndarray] = None,
    breakdown_V: float = -10,
    J0_rev: float = 100e-15,
    J01_photon_coupling: float = 0.0,
    Si_intrinsic_limit: bool = True,
    **kwargs: Any,
) -> Cell:
    """Create a default solar cell circuit model.

    Builds a diode network with source, shunt, and optional series elements.

    Args:
        Jsc (float): Short-circuit current density.
        J01 (float): Saturation current density for n=1 diode.
        J02 (float): Saturation current density for n=2 diode.
        Rshunt (float): Shunt resistance.
        Rs (float): Series resistance.
        area (float): Active area.
        shape (Optional[np.ndarray]): Polygon vertices for layout.
        breakdown_V (float): Reverse breakdown voltage (positive magnitude).
        J0_rev (float): Reverse diode saturation current density.
        J01_photon_coupling (float): Photon coupling diode J01 value.
        Si_intrinsic_limit (bool): If True, add intrinsic Si diode.
        **kwargs (Any): Extra parameters for intrinsic diode.

    Returns:
        Cell: Constructed solar cell instance.

    Example:
        ```python
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell(area=1.5)
        ```
    """
    elements = [circuit.CurrentSource(IL=Jsc, temp_coeff = silicon.Jsc_fractional_temp_coeff*Jsc),
                circuit.ForwardDiode(I0=J01,n=1),
                circuit.ForwardDiode(I0=J02,n=2)]
    if J01_photon_coupling > 0:
        elements.append(circuit.PhotonCouplingDiode(I0=J01_photon_coupling,n=1))
    if Si_intrinsic_limit:
        kwargs_to_pass = {}
        if "thickness" in kwargs:
            kwargs_to_pass["base_thickness"] = kwargs["thickness"]
        if "base_type" in kwargs:
            kwargs_to_pass["base_type"] = kwargs["base_type"]
        if "base_doping" in kwargs:
            kwargs_to_pass["base_doping"] = kwargs["base_doping"]
        elements.append(circuit.Intrinsic_Si_diode(**kwargs_to_pass))
    elements.extend([circuit.ReverseDiode(I0=J0_rev, n=1, V_shift = -breakdown_V),
                circuit.Resistor(cond=1/Rshunt)])
    if Rs == 0.0:
        cell = Cell(elements,"parallel",area=area,location=np.array([0.0,0.0]).astype(float),shape=shape,name="cell")
    else:
        group = circuit.CircuitGroup(elements,"parallel")
        cell = Cell([group,circuit.Resistor(cond=1/Rs)],"series",area=area,location=np.array([0.0,0.0]).astype(float),shape=shape,name="cell")
    return cell

# colormap: choose between cm.magma, inferno, plasma, cividis, viridis, turbo, gray        
draw_modules = draw_cells
    
def make_module(
    cells: List[Cell],
    num_strings: int = 3,
    num_cells_per_halfstring: int = 24,
    halfstring_resistor: float = 0.02,
    I0_rev: float = 1000e-15,
    butterfly: bool = False,
) -> Module:
    """Create a module from a list of cells.

    Cells are arranged into strings and optional butterfly layout.

    Args:
        cells (List[Cell]): Flat list of cells to place into strings.
        num_strings (int): Number of parallel strings.
        num_cells_per_halfstring (int): Cells per half-string.
        halfstring_resistor (float): Series resistor per half-string.
        I0_rev (float): Reverse diode saturation current.
        butterfly (bool): If True, use butterfly layout.

    Returns:
        Module: Constructed module instance.

    Example:
        ```python
        from PV_Circuit_Model.device import make_module, make_solar_cell
        cells = [make_solar_cell() for _ in range(60)]
        module = make_module(cells, num_strings=3, num_cells_per_halfstring=20)
        ```
    """
    count = 0
    cell_strings = []
    num_half_strings = 1
    if butterfly:
        num_half_strings = 2
    else:
        halfstring_resistor /= 2
    for _ in range(num_strings):
        cell_halfstrings = []
        for _ in range(num_half_strings):
            cells_ = cells[count:count+num_cells_per_halfstring]
            count += num_cells_per_halfstring
            circuit.tile_elements(cells_,cols=2, x_gap = 0.1, y_gap = 0.1, turn=True)
            components = cells_
            if halfstring_resistor > 0:
                components += [circuit.Resistor(cond=1/halfstring_resistor)]
            halfstring = circuit.CircuitGroup(components,
                                        "series",name="cell_halfstring")
            cell_halfstrings.append(halfstring)
        if butterfly:
            circuit.tile_elements(cell_halfstrings, cols = 1, y_gap = 1, yflip=True)

        bypass_diode = ByPassDiode(I0=I0_rev, n=1, V_shift = 0)
        bypass_diode.max_I = 0.2*cells[0].area
        cell_strings.append(circuit.CircuitGroup(cell_halfstrings+[bypass_diode],
                                "parallel",name="cell_string"))

    circuit.tile_elements(cell_strings, rows=1, x_gap = 1, y_gap = 0.0)
    module = Module(cell_strings,"series")
    return module

def make_butterfly_module(
    cells: List[Cell],
    num_strings: int = 3,
    num_cells_per_halfstring: int = 24,
    halfstring_resistor: float = 0.02,
    I0_rev: float = 1000e-15,
) -> Module:
    """Create a butterfly-layout module from a list of cells.

    This is a convenience wrapper around `make_module`.

    Args:
        cells (List[Cell]): Flat list of cells to place into strings.
        num_strings (int): Number of parallel strings.
        num_cells_per_halfstring (int): Cells per half-string.
        halfstring_resistor (float): Series resistor per half-string.
        I0_rev (float): Reverse diode saturation current.

    Returns:
        Module: Constructed module instance.

    Example:
        ```python
        from PV_Circuit_Model.device import make_butterfly_module, make_solar_cell
        cells = [make_solar_cell() for _ in range(120)]
        module = make_butterfly_module(cells, num_strings=3, num_cells_per_halfstring=20)
        module.connection
        ```
    """
    return make_module(cells, num_strings, num_cells_per_halfstring, 
                         halfstring_resistor, I0_rev, butterfly=True)

def get_cell_col_row(self: circuit.CircuitGroup, fuzz_distance: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a grid index mapping for cell positions.

    This groups cell centers based on a position tolerance.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        self (CircuitGroup): Group to analyze.
        fuzz_distance (float): Position rounding factor for grouping.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: cell_col_row, map, inverse_map.

    Example:
        ```python
        from PV_Circuit_Model.device import make_module, make_solar_cell
        cells = [make_solar_cell() for _ in range(60)]
        module = make_module(cells, num_strings=3, num_cells_per_halfstring=20)
        cell_col_row, map_, inverse_map = module.get_cell_col_row()
        ```
    """
    shapes, _, _, _, _ = self.draw_cells(display=False)
    xs = []
    ys = []
    indices = []
    for i, shape in enumerate(shapes):
        xs.append(int(np.round(0.5*(np.max(shape[:,0])+np.min(shape[:,0]))/fuzz_distance)))
        ys.append(int(np.round(0.5*(np.max(shape[:,1])+np.min(shape[:,1]))/fuzz_distance)))
        indices.append(i)
    xs = np.array(xs)
    ys = np.array(ys)
    indices = np.array(indices)
    unique_xs = np.unique(xs)
    unique_ys = np.unique(ys)
    unique_ys = unique_ys[::-1] # reverse y such that y increases downwards
    cell_col_row = np.zeros((len(indices),2),dtype=int)
    map = np.zeros((len(indices)),dtype=int)
    inverse_map = np.zeros((len(indices)),dtype=int)
    count = 0
    for i, x in enumerate(unique_xs):
        for j, y in enumerate(unique_ys):
            find_ = np.where((xs==x) & (ys==y))[0]
            cell_col_row[indices[find_],0] = i
            cell_col_row[indices[find_],1] = j
            map[indices[find_]] = count
            inverse_map[count] = indices[find_]
            count += 1
    return cell_col_row, map, inverse_map
circuit.CircuitGroup.get_cell_col_row = get_cell_col_row

