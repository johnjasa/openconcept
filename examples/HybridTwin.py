from __future__ import division
import sys, os
sys.path.insert(0,os.getcwd())
import logging

import numpy as np
from openmdao.api import Problem, Group, ScipyOptimizeDriver, BalanceComp, ExplicitComponent, ExecComp
from openmdao.api import DirectSolver, SqliteRecorder,IndepVarComp,NewtonSolver,BoundsEnforceLS

# imports for the airplane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.dvlabel import DVLabel
from methods.weights_twin_hybrid import TwinSeriesHybridEmptyWeight
from examples.propulsion_layouts.simple_series_hybrid import TwinSeriesHybridElectricPropulsionSystem
from examples.methods.costs_commuter import OperatingCost
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from examples.aircraft_data.KingAirC90GT import data as acdata
from openconcept.analysis.performance.mission_profiles_raymer import FullMissionAnalysis
from openconcept.utilities.linearinterp import LinearInterpolator

spec_energy = 300

class AugmentedFBObjective(ExplicitComponent):
    def setup(self):
        self.add_input('fuel_burn', units='kg')
        self.add_input('ac|weights|MTOW', units='kg')
        self.add_output('mixed_objective', units='kg')
        self.declare_partials(['mixed_objective'], ['fuel_burn'], val=1)
        self.declare_partials(['mixed_objective'], ['ac|weights|MTOW'], val=1/100)
    def compute(self, inputs, outputs):
        outputs['mixed_objective'] = inputs['fuel_burn'] + inputs['ac|weights|MTOW']/100

class SeriesHybridTwinModel(Group):
    """
    A custom model specific to a series hybrid twin turboprop-class airplane
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']
        controls = self.add_subsystem('controls',IndepVarComp(),promotes_outputs=['*'])
        controls.add_output('proprpm',val=np.ones((nn,))*2000,units='rpm')
        if flight_phase == 'climb' or flight_phase == 'cruise' or flight_phase == 'descent':
            controls.add_output('hybridization',val=0.0)
        else:
            controls.add_output('hybridization',val=1.0)

        hybrid_factor = self.add_subsystem('hybrid_factor', LinearInterpolator(num_nodes=nn), promotes_inputs=[('start_val','hybridization'),('end_val','hybridization')])

        propulsion_promotes_outputs = ['fuel_flow','thrust']
        propulsion_promotes_inputs = ["fltcond|*","ac|propulsion|*","throttle","propulsor_active","ac|weights*",'duration']

        self.add_subsystem('propmodel',TwinSeriesHybridElectricPropulsionSystem(num_nodes=nn, specific_energy=spec_energy),
                           promotes_inputs=propulsion_promotes_inputs,promotes_outputs=propulsion_promotes_outputs)
        self.connect('proprpm',['propmodel.prop1.rpm','propmodel.prop2.rpm'])
        self.connect('hybrid_factor.vec','propmodel.hybrid_split.power_split_fraction')

        if flight_phase != 'v0v1' and flight_phase != 'v1vr' and flight_phase != 'rotate':
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_cruise'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])
        else:
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_TO'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])

        nn_simpson = int((nn-1)/2)
        self.add_subsystem('intfuel',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='kg', diff_units='s', time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fuel_flow'),'duration',('q_initial','fuel_used_initial')],promotes_outputs=[('q','fuel_used'),('q_final','fuel_used_final')])
        self.add_subsystem('weight',AddSubtractComp(output_name='weight',input_names=['ac|weights|MTOW','fuel_used'],units='kg',vec_size=[1,nn],scaling_factors=[1,-1]),promotes_inputs=['*'],promotes_outputs=['weight'])
        # TODO add operating cost back in
        self.add_subsystem('OEW',TwinSeriesHybridEmptyWeight(),promotes_inputs=[('P_TO','ac|propulsion|engine|rating'),'*'],promotes_outputs=['OEW'])
        self.connect('propmodel.propellers_weight','W_propeller')
        self.connect('propmodel.eng1.component_weight','W_engine')
        self.connect('propmodel.gen1.component_weight','W_generator')
        self.connect('propmodel.motors_weight','W_motors')


class ElectricTwinAnalysisGroup(Group):
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
        dv_comp.add_output_from_dict('ac|weights|W_battery')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')
        dv_comp.add_output_from_dict('ac|propulsion|generator|rating')
        dv_comp.add_output_from_dict('ac|propulsion|motor|rating')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')
        dv_comp.add_output_from_dict('ac|num_engines')

        mission_data_comp = self.add_subsystem('mission_data_comp',IndepVarComp(),promotes_outputs=["*"])
        # mission_data_comp.add_output('cruise|h0',val=6000, units='m')
        # mission_data_comp.add_output('design_range',val=150,units='NM')
        mission_data_comp.add_output('batt_soc_target', val=0.1, units=None)


        connect_phases = ['rotate','climb','cruise','descent']
        connect_states = ['range','fuel_used','fltcond|h','propmodel.batt1.SOC']
        extra_states_tuple = [(connect_state, connect_phases) for connect_state in connect_states]
        extra_states_tuple.append(('propmodel.batt1.SOC',['v0v1','v1vr','rotate']))
        analysis = self.add_subsystem('analysis',FullMissionAnalysis(num_nodes=nn,
                                                                     aircraft_model=SeriesHybridTwinModel,
                                                                     extra_states=extra_states_tuple),
                                                 promotes_inputs=['*'],promotes_outputs=['*'])
        # self.add_subsystem('hybrid',BalanceComp(name='hybridization',units=None,eq_units=None,lhs_name='batt_end_soc',rhs_name='batt_end_soc_desired',normalize=False))
        # self.connect('hybrid.hybridization',['climb.hybridization','cruise.hybridization','descent.hybridization'])
        # self.connect('descent.propmodel.batt1.SOC_final','hybrid.batt_end_soc')
        # self.connect('batt_soc_target','hybrid.batt_end_soc_desired')
        margins = self.add_subsystem('margins',ExecComp('MTOW_margin = MTOW - OEW - total_fuel - W_battery - payload',
                                                        MTOW_margin={'units':'lbm','value':100},
                                                        MTOW={'units':'lbm','value':10000},
                                                        OEW={'units':'lbm','value':5000},
                                                        total_fuel={'units':'lbm','value':1000},
                                                        W_battery={'units':'lbm','value':1000},
                                                        payload={'units':'lbm','value':1000}),
                                                        promotes_inputs=['payload'])
        self.connect('cruise.OEW','margins.OEW')
        self.connect('descent.fuel_used_final','margins.total_fuel')
        self.connect('ac|weights|MTOW','margins.MTOW')
        self.connect('ac|weights|W_battery','margins.W_battery')

        augobj = self.add_subsystem('aug_obj', AugmentedFBObjective(), promotes_outputs=['mixed_objective'])
        self.connect('ac|weights|MTOW','aug_obj.ac|weights|MTOW')
        self.connect('descent.fuel_used_final','aug_obj.fuel_burn')

if __name__ == "__main__":
    num_nodes=11

    #prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=True)

    design_ranges = [300,350,400,450,500,550,600,650,700]
    specific_energies = [250,300,350,400,450,500,550,600,650,700,750,800]
    # specific_energies = [400]
    # design_ranges = [500]
	# #redo spec range 450, spec energy 700, 750, 800
    logging.basicConfig(filename='opt.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    for design_range in design_ranges:
        for this_spec_energy in specific_energies:
            try:
                prob = Problem()
                prob.model= ElectricTwinAnalysisGroup()


                prob.model.nonlinear_solver=NewtonSolver(iprint=1)
                prob.model.options['assembled_jac_type'] = 'csc'
                prob.model.linear_solver = DirectSolver(assemble_jac=True)
                prob.model.nonlinear_solver.options['solve_subsystems'] = True
                prob.model.nonlinear_solver.options['maxiter'] = 10
                prob.model.nonlinear_solver.options['atol'] = 1e-7
                prob.model.nonlinear_solver.options['rtol'] = 1e-7
                # prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=False)

                spec_energy = this_spec_energy

                run_type = 'optimization'
                if run_type == 'optimization':
                    print('======Performing Multidisciplinary Design Optimization===========')
                    prob.model.add_design_var('ac|weights|MTOW', lower=4000, upper=5700)
                    prob.model.add_design_var('ac|geom|wing|S_ref',lower=15,upper=40)
                    prob.model.add_design_var('ac|propulsion|engine|rating',lower=1,upper=3000)
                    prob.model.add_design_var('ac|propulsion|motor|rating',lower=450,upper=3000)
                    prob.model.add_design_var('ac|propulsion|generator|rating',lower=1,upper=3000)
                    prob.model.add_design_var('ac|weights|W_battery',lower=20,upper=2250)
                    prob.model.add_design_var('ac|weights|W_fuel_max',lower=500,upper=3000)
                    prob.model.add_design_var('cruise.hybridization', lower=0.001, upper=0.999)
                    prob.model.add_design_var('climb.hybridization', lower=0.001, upper=0.999)
                    prob.model.add_design_var('descent.hybridization', lower=0.01, upper=1.0)

                    prob.model.add_constraint('margins.MTOW_margin',lower=0.0)
    #                 prob.model.add_constraint('design_mission.residuals.fuel_capacity_margin',lower=0.0)

                    prob.model.add_constraint('rotate.range_final',upper=1357)
                    prob.model.add_constraint('v0v1.Vstall_eas',upper=42.0)
                    prob.model.add_constraint('descent.propmodel.batt1.SOC_final',lower=0.0)
                    prob.model.add_constraint('climb.throttle',upper=1.05*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('cruise.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('cruise.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('cruise.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('descent.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('descent.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('descent.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('v0v1.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('engineoutclimb.gamma',lower=0.02)
                    prob.model.add_objective('mixed_objective') # TODO add this objective

                elif run_type == 'comp_sizing':
                    print('======Performing Component Sizing Optimization===========')
                    prob.model.add_design_var('ac|propulsion|engine|rating',lower=1,upper=3000)
                    prob.model.add_design_var('ac|propulsion|motor|rating',lower=1,upper=3000)
                    prob.model.add_design_var('ac|propulsion|generator|rating',lower=1,upper=3000)
                    prob.model.add_design_var('ac|weights|W_battery',lower=20,upper=2250)
                    prob.model.add_design_var('cruise.hybridization', lower=0.01, upper=0.5)

                    prob.model.add_constraint('margins.MTOW_margin',equals=0.0) # TODO implement
                    prob.model.add_constraint('rotate.range_final',upper=1357) # TODO check units
                    prob.model.add_constraint('descent.propmodel.batt1.SOC_final',lower=0.0)
                    prob.model.add_constraint('v0v1.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('v0v1.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('v0v1.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(num_nodes))
                    prob.model.add_constraint('climb.throttle',upper=1.05*np.ones(num_nodes))
                    prob.model.add_objective('fuel_burn')



                else:
                    print('======Analyzing Fuel Burn for Given Mision============')
                    prob.model.add_design_var('cruise.hybridization', lower=0.01, upper=0.5)
                    prob.model.add_constraint('descent.propmodel.batt1.SOC_final',lower=0.0)
                    prob.model.add_objective('descent.fuel_used_final')


                prob.driver = ScipyOptimizeDriver()
                prob.driver.options['dynamic_simul_derivs'] = True
                #prob.driver.options['tol'] = 1e-13
                filename_to_save = 'case_'+str(spec_energy)+'_'+str(design_range)+'.sql'
                if os.path.isfile(filename_to_save):
                    print('Skipping '+filename_to_save)
                    continue
                recorder = SqliteRecorder(filename_to_save)
                prob.driver.add_recorder(recorder)
                prob.driver.recording_options['includes'] = []
                prob.driver.recording_options['record_objectives'] = True
                prob.driver.recording_options['record_constraints'] = True
                prob.driver.recording_options['record_desvars'] = True

                prob.setup(check=False)
                # set some (optional) guesses for takeoff speeds and (required) mission parameters
                prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
                prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
                prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
                # mission parameters
                prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
                prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
                prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
                prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*170, units='kn')
                prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
                prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')

                prob.set_val('cruise|h0',29000,units='ft')
                prob.set_val('mission_range',design_range,units='NM')
                prob.set_val('payload',1000,units='lb')

                prob['analysis.cruise.acmodel.OEW.const.structural_fudge'] = 2.0
                prob['ac|propulsion|propeller|diameter'] = 2.2
                prob['ac|propulsion|engine|rating'] = 1117.2

                run_flag = prob.run_driver()
                if run_flag:
                    raise ValueError('Opt failed')

            except BaseException as e:
                logging.error('Optimization '+filename_to_save+' failed because '+repr(e))
                prob.cleanup()
                try:
                    os.rename(filename_to_save, filename_to_save.split('.sql')[0]+'_failed.sql')
                except WindowsError as we:
                    logging.error('Error renaming file: '+repr(we))
                    os.remove(filename_to_save)

    # list some outputs
    units=['lb','lb',None,None]
    for i, thing in enumerate(['ac|weights|MTOW','descent.fuel_used_final','descent.propmodel.batt1.SOC_final','cruise.hybridization']):
        if units[i] is not None:
            print(thing+' '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])
        else:
            print(thing+' '+str(prob.get_val(thing,units=units[i])[0]))

    print('Design range: '+str(prob.get_val('mission_range', units='NM')))
    print('MTOW: '+str(prob.get_val('ac|weights|MTOW', units='lb')))
    print('OEW: '+str(prob.get_val('cruise.OEW', units='lb')))
    print('Battery wt: '+str(prob.get_val('ac|weights|W_battery', units='lb')))
    print('Fuel cap:'+str(prob.get_val('ac|weights|W_fuel_max', units='lb')))
    print('MTOW margin: '+str(prob.get_val('margins.MTOW_margin', units='lb')))
    print('Battery margin: '+str(prob.get_val('descent.propmodel.batt1.SOC_final', units=None)))

    print('Eng power:'+str(prob.get_val('ac|propulsion|engine|rating', units='hp')))
    print('Gen power:'+str(prob.get_val('ac|propulsion|generator|rating', units='hp')))
    print('Motor power:'+str(prob.get_val('ac|propulsion|motor|rating', units='hp')))
    print('Hybrid split|'+str(prob.get_val('cruise.hybridization', units=None)))
    print('Prop diam:'+str(prob.get_val('ac|propulsion|propeller|diameter', units='m')))

    print('TO (continue):'+str(prob.get_val('rotate.range_final', units='ft')))
    print('TO (abort):'+str(prob.get_val('v1v0.range_final', units='ft')))
    print('Stall speed'+str(prob.get_val('v0v1.Vstall_eas', units='kn')))
    print('Rotate speed'+str(prob.get_val('v0v1.takeoff|vr', units='kn')))
    # print('Decision speed'+str(prob.get_val('v0v1.takeoff|v1', units='kn')))
    print('S_ref: ' +str(prob.get_val('ac|geom|wing|S_ref', units='ft**2')))

    print('Mission Fuel burn: '+ str(prob.get_val('descent.fuel_used_final', units='lb')))
    print('TO fuel burn: '+ str(prob.get_val('rotate.fuel_used_final', units='lb')))
    print('Total fuel burn:' +str(prob.get_val('descent.fuel_used_final', units='lb')))
    print('EO climb angle: '+str(prob.get_val('engineoutclimb.gamma')))
    # plot some stuff
    plots = True
    save_file = False
    load_file = False
    file_base = 'compressible'

    if plots:
        from matplotlib import pyplot as plt
        # x_variable = 'range'
        # x_units='ft'
        # y_variables = ['fltcond|Ueas','fltcond|h']
        # y_units = ['kn','ft']
        # phases_to_plot = ['v0v1','v1vr','rotate','v1v0']
        # val_list=[]
        # for phase in phases_to_plot:
        #     val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        # x_vec = np.concatenate(val_list)

        # for i, y_var in enumerate(y_variables):
        #     val_list = []
        #     for phase in phases_to_plot:
        #         val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
        #     y_vec = np.concatenate(val_list)
        #     plt.figure()
        #     plt.plot(x_vec, y_vec,'o')
        #     plt.xlabel(x_variable)
        #     plt.ylabel(y_var)
        #     plt.title('takeoff / rejected takeoff')
        # plt.show()

        phases_to_plot = ['v0v1','v1vr','rotate','climb','cruise','descent']
        x_variable = 'range'
        x_units='NM'
        y_variables = ['fltcond|h','fltcond|Ueas','throttle','fltcond|vs','weight','propmodel.eng1.throttle','propmodel.batt1.SOC']
        y_units = ['ft','kn',None,'ft/min','lb',None,None]

        val_list= []
        for phase in phases_to_plot:
            val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        x_vec = np.concatenate(val_list)
        if save_file:
            np.save(file_base+'_x',x_vec)

        for i, y_var in enumerate(y_variables):
            val_list = []
            for phase in phases_to_plot:
                val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
            y_vec = np.concatenate(val_list)
            if save_file:
                filename = file_base+'_'+y_var.replace("|","")
                np.save(filename,y_vec)
            plt.figure()
            if load_file:
                x_loaded = np.load(file_base+'_x'+'.npy')
                y_loaded = np.load(file_base+'_'+y_var.replace("|","")+'.npy')
                plt.plot(x_vec,y_vec,x_loaded,y_loaded)
            else:
                plt.plot(x_vec, y_vec)
            plt.xlabel(x_variable)
            plt.ylabel(y_var)
            plt.title('mission profile')
        plt.show()

