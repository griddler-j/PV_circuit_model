from Tandem_Cell_Fit_Tools.tandem_measurement import *
from PV_Circuit_Model.measurement import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
from PV_Circuit_Model.data_fitting import *
    
def analyze_tandem_cell_measurements(measurements,num_of_rounds=40,regularization_method=0,prefix=None,sample_info={},tandem_cell_starting_guess=None,use_fit_dashboard=True,**kwargs):
    global pbar, axs
    aux = {"regularization_method": regularization_method,"limit_order_of_mag": 1}
    aux.update(kwargs)

    test_cell_area = 1.0
    bottom_cell_thickness = 160e-4
    if "area" in sample_info:
        test_cell_area = sample_info["area"]
    if "bottom_cell_thickness" in sample_info:
        bottom_cell_thickness = sample_info["bottom_cell_thickness"]

    if tandem_cell_starting_guess is not None:
        tandem_cell = circuit_deepcopy(tandem_cell_starting_guess)
    else:
        tandem_cell = quick_tandem_cell()
        bottom_cell = None
        top_cell = None
        for measurement in measurements:
            if isinstance(measurement,Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[0,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc < bottom_cell_Isc*1e-3: # this is red spectrum Suns-Voc
                    bottom_cell_Voc = measurement.measurement_data[0,arg_max]
                    Jsc = bottom_cell_Isc/test_cell_area
                    J01, J02 = estimate_cell_J01_J02(Jsc=Jsc,Voc=bottom_cell_Voc,thickness=bottom_cell_thickness)
                    bottom_cell = make_solar_cell(Jsc, J01, J02, thickness=bottom_cell_thickness, area=test_cell_area)
                    bottom_cell.set_Suns(1.0)
                    break
        for measurement in measurements:
            if isinstance(measurement,Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[1,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc > bottom_cell_Isc*1e-2: # this is white spectrum Suns-Voc
                    tandem_cell_Voc = measurement.measurement_data[0,arg_max]
                    bottom_cell.set_IL(bottom_cell_Isc)
                    bottom_cell.build_IV()
                    top_cell_Voc = tandem_cell_Voc - bottom_cell.get_Voc()
                    Jsc = top_cell_Isc/test_cell_area
                    J01, J02 = estimate_cell_J01_J02(Jsc=Jsc,Voc=top_cell_Voc,Si_intrinsic_limit=False)
                    # just arbitrarily guess the PC
                    J01_PC = J01*0.2
                    J01 = J01*0.8
                    top_cell = make_solar_cell(Jsc, J01, J02, area=test_cell_area, 
                            Si_intrinsic_limit=False,J01_photon_coupling=J01_PC)
                    top_cell.set_Suns(1.0)
                    break
        if bottom_cell is not None and top_cell is not None:
            tandem_cell = MultiJunctionCell([bottom_cell,top_cell])

    tandem_cell.assign_measurements(measurements)
    fit_parameters = Tandem_Cell_Fit_Parameters(tandem_cell)

    fit_dashboard = None
    if use_fit_dashboard:
        fit_dashboard = Fit_Dashboard(3,3,save_file_name=prefix)
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"white"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"black"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"blue"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"blue"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"red"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"red"})
        fit_dashboard.define_plot_what(which_axs=2, measurement_type=Dark_IV_measurement, 
                                    measurement_condition={'spectrum':"red"}, 
                                    plot_type="overlap_curves",
                                        title="Dark IV", plot_style_parameters={"color":"red"})
        fit_dashboard.define_plot_what(which_axs=2, measurement_type=Dark_IV_measurement, 
                                    measurement_condition={'spectrum':"blue"}, 
                                    plot_type="overlap_curves",
                                        title="Dark IV", plot_style_parameters={"color":"blue"})
        for i, key in enumerate(Light_IV_measurement.keys):
            for j, Suns in enumerate([1.0, 0.5]):
                fit_dashboard.define_plot_what(which_axs=3*(j+1)+i, measurement_type=Light_IV_measurement, 
                                            key_parameter=key, 
                                            measurement_condition={'Suns':Suns},
                                            plot_type="overlap_key_parameter",
                                            x_axis="I_imbalance",
                                            title=str(Suns)+" Suns IV", plot_style_parameters={"color":"blue"})
                
    result = fit_routine(tandem_cell,fit_parameters,
                routine_functions={
                    "update_function":linear_regression,
                    "comparison_function":compare_experiments_to_simulations,
                },
                fit_dashboard=fit_dashboard,
                aux=aux,num_of_epochs=num_of_rounds)
    if num_of_rounds==0:
        return result
    fit_parameters.apply_to_ref(aux)
    interactive_fit_dashboard = None
    if use_fit_dashboard:
        interactive_fit_dashboard = Interactive_Fit_Dashboard(tandem_cell,fit_parameters,ref_fit_dashboard=fit_dashboard)
    return fit_parameters.ref_sample, interactive_fit_dashboard

def generate_differentials(measurements,tandem_cell):
    return analyze_tandem_cell_measurements(measurements,num_of_rounds=0,tandem_cell_starting_guess=tandem_cell)
