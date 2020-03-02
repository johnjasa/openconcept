from __future__ import division
import sys, os
sys.path.insert(0,os.getcwd())
import numpy as np
from openmdao.api import Problem, Group, ScipyOptimizeDriver
from openmdao.api import DirectSolver, SqliteRecorder,IndepVarComp,NewtonSolver,BoundsEnforceLS

# imports for the airplane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from examples.methods.weights_turboprop import SingleTurboPropEmptyWeight
from examples.propulsion_layouts.simple_turboprop import TurbopropPropulsionSystem
from examples.methods.costs_commuter import OperatingCost
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from examples.aircraft_data.TBM850 import data as acdata
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis


class TBM850AirplaneModel(Group):
    """
    A custom model specific to the TBM 850 airplane
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']
        controls = self.add_subsystem('controls',IndepVarComp(),promotes_outputs=['*'])
        controls.add_output('prop1rpm',val=np.ones((nn,))*2000,units='rpm')

        propulsion_promotes_outputs = ['fuel_flow','thrust']
        propulsion_promotes_inputs = ["fltcond|*","ac|propulsion|*","throttle"]

        self.add_subsystem('propmodel',TurbopropPropulsionSystem(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,promotes_outputs=propulsion_promotes_outputs)
        self.connect('prop1rpm','propmodel.prop1.rpm')

        if flight_phase != 'v0v1' and flight_phase != 'v1vr' and flight_phase != 'rotate':
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_cruise'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])
        else:
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_TO'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])

        self.add_subsystem('OEW',SingleTurboPropEmptyWeight(),promotes_inputs=['*',('P_TO','ac|propulsion|engine|rating')], promotes_outputs=['OEW'])
        self.connect('propmodel.prop1.component_weight','W_propeller')
        self.connect('propmodel.eng1.component_weight','W_engine')
        nn_simpson = int((nn-1)/2)
        self.add_subsystem('intfuel',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='kg', diff_units='s', time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fuel_flow'),'duration',('q_initial','fuel_used_initial')],promotes_outputs=[('q','fuel_used'),('q_final','fuel_used_final')])
        self.add_subsystem('weight',AddSubtractComp(output_name='weight',input_names=['ac|weights|MTOW','fuel_used'],units='kg',vec_size=[1,nn],scaling_factors=[1,-1]),promotes_inputs=['*'],promotes_outputs=['weight'])


class TBMAnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis.
    """
    def setup(self):
        nn = 11

        dv_comp = self.add_subsystem('dv_comp',DictIndepVarComp(acdata,seperator='|'),promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|CLmax_flaps30')
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')

        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|wing|taper')
        dv_comp.add_output_from_dict('ac|geom|wing|toverc')
        dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
        dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        connect_phases = ['rotate','climb','cruise','descent']
        connect_states = ['range','fuel_used','fltcond|h']
        extra_states_tuple = [(connect_state, connect_phases) for connect_state in connect_states]
        analysis = self.add_subsystem('analysis',FullMissionAnalysis(num_nodes=nn,
                                                                     aircraft_model=TBM850AirplaneModel,
                                                                     extra_states=extra_states_tuple),
                                                 promotes_inputs=['*'],promotes_outputs=['*'])


if __name__ == "__main__":
    num_nodes = 11
    prob = Problem()
    prob.model= TBMAnalysisGroup()
    prob.model.nonlinear_solver=NewtonSolver(iprint=1)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=False)

    # prob.model.add_design_var('cruise.fltcond|Ueas', lower=50*np.ones((num_nodes,)), upper=120*np.ones((num_nodes,)))
    # prob.model.add_design_var('climb.fltcond|Ueas', lower=50*np.ones((num_nodes,)), upper=120*np.ones((num_nodes,)))
    # prob.model.add_design_var('descent.fltcond|Ueas', lower=50*np.ones((num_nodes,)), upper=120*np.ones((num_nodes,)))
    # prob.model.add_design_var('climb.fltcond|vs', lower=0.5*np.ones((num_nodes,)), upper=15*np.ones((num_nodes,)))
    # prob.model.add_design_var('descent.fltcond|vs', lower=-15*np.ones((num_nodes,)), upper=-0.5*np.ones((num_nodes,)))
    # prob.model.add_design_var('cruise.fltcond|vs', lower=-0.5*np.ones((num_nodes,)), upper=0.5*np.ones((num_nodes,)))

    # prob.model.add_constraint('climb.throttle',upper=np.ones((num_nodes,)))
    # prob.model.add_constraint('descent.throttle',lower=0.1*np.ones((num_nodes,)),upper=np.ones((num_nodes,)))
    # prob.model.add_constraint('cruise.fltcond|h',upper=9735*np.ones((num_nodes,)))
    # prob.model.add_constraint('cruise.throttle',upper=np.ones((num_nodes,)))

    # prob.model.add_objective('descent.fuel_used_final')
    # prob.driver = ScipyOptimizeDriver()
    # prob.driver.options['dynamic_simul_derivs'] = False

    prob.setup(check=True,mode='fwd')
    # set some (optional) guesses for takeoff speeds and (required) mission parameters
    prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
    prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('rotate.fltcond|Utrue',np.ones((num_nodes))*80,units='kn')
    prob.set_val('rotate.accel_vert',np.ones((num_nodes))*0.1,units='m/s**2')
    prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*201, units='kn')
    prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    prob['climb.OEW.structural_fudge'] = 1.67
    prob['v0v1.throttle'] = np.ones((num_nodes)) / 1.21
    prob['v1vr.throttle'] = np.ones((num_nodes)) / 1.21
    prob['rotate.throttle'] = np.ones((num_nodes)) / 1.21
    prob.run_model()

    # list some outputs
    units=['lb','lb','lb','ft']
    for i, thing in enumerate(['ac|weights|MTOW','climb.OEW','descent.fuel_used_final','rotate.range_final']):
        print(thing+' '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    plots = True
    if plots:
        from matplotlib import pyplot as plt
        x_variable = 'range'
        x_units='ft'
        y_variables = ['fltcond|Ueas','fltcond|h']
        y_units = ['kn','ft']
        phases_to_plot = ['v0v1','v1vr','rotate','v1v0']
        val_list=[]
        for phase in phases_to_plot:
            val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        x_vec = np.concatenate(val_list)

        for i, y_var in enumerate(y_variables):
            val_list = []
            for phase in phases_to_plot:
                val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
            y_vec = np.concatenate(val_list)
            plt.figure()
            plt.plot(x_vec, y_vec,'o')
            plt.xlabel(x_variable)
            plt.ylabel(y_var)
            plt.title('takeoff / rejected takeoff')
        plt.show()

        phases_to_plot = ['climb','cruise','descent']
        x_variable = 'range'
        x_units='NM'
        y_variables = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs']
        y_units = ['ft','kn','lbm',None,'ft/min']

        val_list= []
        for phase in phases_to_plot:
            val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        x_vec = np.concatenate(val_list)

        for i, y_var in enumerate(y_variables):
            val_list = []
            for phase in phases_to_plot:
                val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
            y_vec = np.concatenate(val_list)
            plt.figure()
            plt.plot(x_vec, y_vec)
            plt.xlabel(x_variable)
            plt.ylabel(y_var)
            plt.title('mission profile')
        plt.show()
