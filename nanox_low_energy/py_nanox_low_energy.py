"""
Contains functions to calculate cell survivals under NanOx low energy hypothesis.
Alpha coefficients tables and tables of conversion of energy in let are contained in the module.
"""

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from os import path
import pandas as pd
import pkg_resources
import scipy.integrate
import scipy.interpolate as interpolate


# Define constants and unit conversion factors

KEV_IN_J = 1.60218 * 1e-16
WATER_DENSITY = 1e-15  #kg/µm³
UNIT_COEFFICIENT_A = KEV_IN_J / WATER_DENSITY # Gy.µm³.keV-1
BETAG = pd.DataFrame({"HSG" : [0.0961], "V79" : [0.0405], "CHO-K1" : [0.0625]}) #Gy-2. constant of Monini et al. 2019
MOLECULE_PER_100_EV_IN_MOL_PER_J = 1 / (9.6 * 10**6)
ETA = 0.8 # Fraction of energy lost associated with biological damage
G_REF = 6.33582 # Chemical yield for reference photon radiation considered at t = 1e-11 s

radius_nucleus_cell_line = pd.DataFrame({"HSG" : [7], "V79" : [4.9], "CHO-K1" : [5.9]}) # µm
#simulated_radius_nucleus_cell_line = pd.DataFrame({"HSG" : [6.7], "V79" : [5.2], "CHO-K1" : [3.85]}) # µm
simulated_radius_nucleus_cell_line = pd.DataFrame({"HSG" : [7], "V79" : [5.2], "CHO-K1" : [3.85]}) # µm
length_of_cylinderslice_cell = 1 #µm

#Test for ellipsoid dimensions:
#radius_nucleus_cell_line = pd.DataFrame({"HSG" : [7.05], "V79" : [4.9], "CHO-K1" : [5.9]}) # µm
#simulated_radius_nucleus_cell_line = pd.DataFrame({"HSG" : [3.535], "V79" : [5.2], "CHO-K1" : [3.85]}) # µm
#length_of_cylinderslice_cell = 1.25 #µm

#######################


def cell_survival_lethal(ei, ef, cell_line, particle, physics_list, option="cumulated"):
    """
    Only works for Helium ions for now
    :param ei: Entrance energy of particle in nucleus. Can be double or numpy array
    :param ef: Exit energy of particle in nucleus. Can be double or numpy array
    :param cell_line: HSG, V79 or CHO-K1
    :param particle: only Helium for now
    :param physics_list: em or dna
    :param option: "cumulated" = cell survival to all given impacts,
     "mean_to_one_impact" = mean cell survival to one impact
    :return: returns the cell survival to lethal events of one cell
    """
    assert (particle == "Helium")
    dn1_de_continuous_pre_calculated = \
        dn1_de_continuous_mv_tables_global_events_correction(cell_line, physics_list)
    emax = np.max(ei)
    n1_function = number_of_lethal_events_for_alpha_traversals(dn1_de_continuous_pre_calculated, emax)

    n_lethal = (n1_function(ei) - n1_function(ef))

    match option:
        case "cumulated":
            lethal_survival = np.exp(-np.sum(n_lethal))
        case "mean_to_one_impact":
            lethal_survival = np.exp(-n_lethal)
        case _:
            raise InvalidOption("Choose cumulated or mean_to_one_impact option")
    return lethal_survival


def cell_survival_global(ei, ef, cell_line, particle, option = "cumulated"):
    """
    Only works for Helium ions for now
    :param ei: Entrance energy of particle in nucleus. Can be double or numpy array
    :param ef: Exit energy of particle in nucleus. Can be double or numpy array
    :param cell_line: HSG, V79 or CHO-K1
    :param particle: only Helium for now
    :param option: "cumulated" = cell survival to all given impacts,
      "mean_to_one_impact" = mean cell survival to one impact
    :return: returns the cell survival to global events of one cell
    """
    assert (particle == "Helium")
    beta_G = BETAG[cell_line].iloc[0]

    z_tilde = z_tilde_func(ei, ef, cell_line)

    match option:
        case "cumulated":
            global_survival = np.exp(-beta_G * np.sum(z_tilde) ** 2)
        case "mean_to_one_impact":
            global_survival = np.exp(-beta_G * z_tilde ** 2)
        case _:
            raise InvalidOption("Choose cumulated or mean_to_one_impact option")

    return global_survival

def cell_survival_total(ei, ef, cell_line, particle, physics_list, option = "cumulated"):
    """
    Only works for Helium ions for now
    :param ei: Entrance energy of particle in nucleus in keV. Can be double or numpy array
    :param ef: Exit energy of particle in nucleus in keV. Can be double or numpy array
    :param cell_line: HSG, V79 or CHO-K1
    :param particle: only Helium for now
    :param physics_list: em or dna
    :param option: "cumulated" = cell survival to all given impacts,
      "mean_to_one_impact" = mean cell survival to one impact
    :return: returns the cell survival to lethal events of one cell
    """
    assert(particle == "Helium")

    match option:
        case "cumulated":
            lethal_survival = cell_survival_lethal(ei, ef, cell_line, particle, physics_list)
            global_survival = cell_survival_global(ei, ef, cell_line, particle)
        case "mean_to_one_impact":
            lethal_survival = cell_survival_lethal(ei, ef, cell_line, particle, physics_list,
                                                   option="mean_to_one_impact")
            global_survival = cell_survival_global(ei, ef, cell_line, particle,
                                                   option="mean_to_one_impact")
        case _:
            raise InvalidOption("Choose cumulated or mean_to_one_impact option")

    return lethal_survival * global_survival

def cell_survival_lethal_without_global_correction(ei, ef, cell_line, particle, physics_list, option = "cumulated"):
    """
    !!!!!!
    This method is considering only lethal events, but their are calculated with alpha coefficients that contains
    global events. To use cleaner calculations, prefer the other cell survival functions.
    !!!!!!

    Only works for Helium ions for now
    :param ei: Entrance energy of particle in nucleus. Can be double or numpy array
    :param ef: Exit energy of particle in nucleus. Can be double or numpy array
    :param cell_line: HSG, V79 or CHO-K1
    :param particle: only Helium for now
    :param physics_list: em or dna
    :param option: "cumulated" = cell survival to all given impacts,
     "mean_to_one_impact" = mean cell survival to one impact
    :return: returns the cell survival to lethal events of one cell
    """
    assert(particle == "Helium")
    dn1_de_continuous_pre_calculated = dn1_de_continuous_mv_tables(cell_line, physics_list)
    emax = np.max(ei)
    n1 = number_of_lethal_events_for_alpha_traversals(dn1_de_continuous_pre_calculated, emax)

    n_run = (n1(ei) - n1(ef))

    match option:
        case "cumulated":
            lethal_survival = np.exp(-np.sum(n_run))
        case "mean_to_one_impact":
            lethal_survival = np.exp(-n_run)
        case _:
            raise InvalidOption("Choose cumulated or mean_to_one_impact option")
    return lethal_survival

def z_tilde_func(ei, ef, cell_line):
    """
    Function to compute the chemical specific energy within the low-energy NanOx formalism

    Parameters
    ----------
    ei : double or np.array
        Entrance energy of particle in nucleus.
    ef : double or np.array
        Exit energy of particle in nucleus.
    cell_line : str
        HSG, V79 or CHO-K1

    Returns
    -------
    z_tilde :
        A function to compute the chemical specific energy.

    """
    h = chemical_yield_and_primitive()[1]
    #volume_sensitive = (4 / 3) * math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0]) ** 3
    #volume_sensitive = (4/3) * math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0]**2)*length_of_cylinderslice_cell
    volume_sensitive = math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0]**2)*length_of_cylinderslice_cell

    sensitive_mass = WATER_DENSITY * volume_sensitive #kg
    z_tilde = (ETA / (sensitive_mass * G_REF)) * (h(ei) - h(ef)) * KEV_IN_J
    return z_tilde

def dn1_de_continuous_mv_tables(cell_line, physics_list, particle = 0, method_threshold = "Interp"):
    """
    Returns a continous function that calculates dn1_de in function of energy. It depends on the radiobiological alpha
    coefficient. These are extracted from alpha tables that Mario Alcocer-Avila calculated.

    The data are smoothered via a moving average method.

    method_threshold argument corresponds to the extrapolation method under 100 keV/n:
    - Interp is a linear interpolation between dn1_de = 0 at energy = 0 and the last alpha point
    - Zero sets a strict value of 0 for every dn1_de under the threshold
    - Last sets the value of every alpha under the threshold as the last dn1_de data, i.e. alpha(100 keV/n)
    """

    #resource_path = f"resources/AlphasTables/alpha_He_{cell_line}.csv"
    # file_content = pkg_resources.resource_stream(__name__, resource_path)
    resources_dir = path.join(path.dirname(__file__), 'resources')

    # Lithium :
    if particle == 2 :
        alpha_table = pd.read_csv(f"{resources_dir}/AlphasTables/alpha_Li_{cell_line}.csv")

    # Helium :
    else :
        alpha_table = pd.read_csv(f"{resources_dir}/AlphasTables/alpha_He_{cell_line}.csv")


    alpha_discrete_from_tables = alpha_table["Alpha (Gy-1)"].to_numpy().astype(float)
    e_discrete_from_tables = alpha_table["E(MeV/n)"].to_numpy().astype(float)*1000*4   #keV

    surface_centerslice_cell_line = math.pi * radius_nucleus_cell_line[cell_line].iloc[0] ** 2   # µm²

    let_discrete_from_tables = alpha_table["LET (keV/um)"].to_numpy().astype(float)
    _conversion_energy_in_let_srim = let_discrete_from_tables
    _conversion_energy_in_let_g4 = _conversion_energy_in_let(f"G4_{physics_list}", e_discrete_from_tables, particle)

    e_discrete_from_tables_with_0 = np.insert(e_discrete_from_tables, 0, 0, axis=0)

    dn1_de = -np.log(1 - alpha_discrete_from_tables  * UNIT_COEFFICIENT_A \
                             * _conversion_energy_in_let_srim / surface_centerslice_cell_line) \
             / (length_of_cylinderslice_cell * _conversion_energy_in_let_g4)
                    #calculation of number of lethal events per keV

    dn1_de = _moving_average_dn1_de_tables(dn1_de, cell_line)

    if method_threshold == "Interp":
        dn1_de = np.insert(dn1_de, 0, 0, axis=0)
        dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables_with_0, dn1_de, kind="linear")
    elif method_threshold == "Zero":
        dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables, dn1_de,
                                                                  fill_value=(0,"extrapolate"), kind="linear",
                                                                  bounds_error=False)
    elif method_threshold == "Last":
        dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables, dn1_de,
                                                              fill_value=(dn1_de[0],"extrapolate"), kind="linear",
                                                              bounds_error=False)
    else:
        raise UnexistingMethodAlphaCoeffExtrapolation("Choose an existing method for the call of "
                                                      "dn1_de_continuous_mv_tables")

    return dn1_de_continuous

def _plot_dn1_de():
    energy = np.linspace(400, 200000, 200000)
    dn1_dE = dn1_de_continuous_mv_tables("HSG", "em") ###TO DO: change the call of func
    fig, ax = plt.subplots(figsize=(15, 12))
    plt.tick_params(axis='both', which='major', pad=9, length=15, width=2, colors='black')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', pad=9, direction='in', length=10, width=1)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.plot(energy / 4000, dn1_dE(energy), marker='>', linestyle="solid",
            markersize=0, color="blue", label="a")
    plt.ylabel('dn1/dE (keV-1)', fontsize=25, fontname="Arial", fontweight='bold', labelpad=9,
               color='black')
    plt.xlabel('Kinetic energy of alpha particles (MeV/n)', fontsize=25, fontname="Arial",
               fontweight='bold', labelpad=9, color='black')
    plt.xticks(fontsize=21, fontname="Arial", color='black')
    plt.yticks(fontsize=21, fontname="Arial", color='black')
    plt.grid(True)
    ax.set_yscale('log')
    plt.xscale('log')
    plt.rc('font', family='Arial')  # legend font
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig("dn1_dE.png")
    plt.show()
    return dn1_dE

def dn1_de_continuous_mv_tables_global_events_correction(cell_line, physics_list, method_threshold = "Interp"):
    """
    Returns a continous function that calculates dn1_de in function of energy. It depends on the radiobiological alpha
    coefficient. These are extracted from alpha tables that Mario Alcocer-Avila calculated. These coefficients are
    fitted on curves that contain both lethal and global events. Hence, a correction is needed to keep only lethal
    events.

    The data are smoothered via a moving average method.

    method_threshold argument corresponds to the extrapolation method under 100 keV/n:
    - Interp is a linear interpolation between dn1_de = 0 at energy = 0 and the last alpha point
    - Zero sets a strict value of 0 for every dn1_de under the threshold
    - Last sets the value of every alpha under the threshold as the last dn1_de data, i.e. alpha(100 keV/n)
    """

    resources_dir = path.join(path.dirname(__file__), 'resources')
    alpha_table = pd.read_csv(f"{resources_dir}/AlphasTables/alpha_He_{cell_line}.csv")
    alpha_discrete_from_tables = alpha_table["Alpha (Gy-1)"].to_numpy().astype(float)
    e_discrete_from_tables = alpha_table["E(MeV/n)"].to_numpy().astype(float)*1000*4   #keV

    surface_centerslice_cell_line = math.pi * radius_nucleus_cell_line[cell_line].iloc[0] ** 2   #µm²

    let_discrete_from_tables = alpha_table["LET (keV/um)"].to_numpy().astype(float)
    _conversion_energy_in_let_srim = let_discrete_from_tables
    _conversion_energy_in_let_g4 = _conversion_energy_in_let(f"G4_{physics_list}", e_discrete_from_tables)

    e_discrete_from_tables_with_0 = np.insert(e_discrete_from_tables, 0, 0, axis=0)

    #volume_sensitive = (4/3) * math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0])**3
    #volume_sensitive = (4/3) * math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0]**2)*length_of_cylinderslice_cell
    volume_sensitive = math.pi * (simulated_radius_nucleus_cell_line[cell_line].iloc[0]**2)*length_of_cylinderslice_cell

    sensitive_mass = WATER_DENSITY * volume_sensitive
    beta_G = BETAG[cell_line].iloc[0]
    G = chemical_yield_and_primitive()[0]

    _lethal_global_part = (-np.log(1 - alpha_discrete_from_tables  * UNIT_COEFFICIENT_A
                             * _conversion_energy_in_let_srim / surface_centerslice_cell_line)
                          /
                          (length_of_cylinderslice_cell * _conversion_energy_in_let_g4))

    _global_correction = (beta_G * (G(e_discrete_from_tables)*ETA / (sensitive_mass * G_REF))**2
                          * _conversion_energy_in_let_g4 * length_of_cylinderslice_cell) * (KEV_IN_J**2)

    dn1_de = _lethal_global_part - _global_correction
    #calculation of number of lethal events per keV

    ####### testing volume ratio

    volume_ratio = (math.pi * radius_nucleus_cell_line[cell_line].iloc[0]**2 * 1) \
              /(math.pi * simulated_radius_nucleus_cell_line[cell_line].iloc[0]**2 * length_of_cylinderslice_cell)

    dn1_de = dn1_de * volume_ratio

    ########

    dn1_de = _moving_average_dn1_de_tables(dn1_de, cell_line)

    match method_threshold:
        case "Interp":
            dn1_de = np.insert(dn1_de, 0, 0, axis=0)
            dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables_with_0, dn1_de, kind="linear")
        case "Zero":
            dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables, dn1_de,
                                                                      fill_value=(0,"extrapolate"), kind="linear",
                                                                      bounds_error=False)
        case "Last":
            dn1_de_continuous = interpolate.interp1d(e_discrete_from_tables, dn1_de,
                                                                  fill_value=(dn1_de[0],"extrapolate"), kind="linear",
                                                                  bounds_error=False)
        case _:
            raise UnexistingMethodAlphaCoeffExtrapolation("Choose an existing method for the call of "
                                                          "dn1_de_continuous_mv_tables")

    return dn1_de_continuous



def number_of_lethal_events_for_alpha_traversals(dn1_de_function, max_energy):
    """
    Returns the function that converts an energy E into the cumulated number of lethal damage from 0 to E
    """
    bins = 200
    energie_table_binned = np.linspace(0, max_energy, num=bins)
    f_he_cumulative_int = scipy.integrate.cumtrapz(dn1_de_function(energie_table_binned),
                                                   energie_table_binned, initial=0)
    n1 = interpolate.interp1d(energie_table_binned, f_he_cumulative_int, fill_value="extrapolate",
                              kind="linear")  #continuous primitive function of energy
    return n1


def chemical_yield_and_primitive():
    resources_dir = path.join(path.dirname(__file__), 'resources')
    chemical_yields_file = pd.read_csv(f"{resources_dir}/ChemicalYield/chemical_yields_HELIUM.csv")
    energy = chemical_yields_file["Energy (MeV/n)"] * 4000 # keV
    chemical_yields = chemical_yields_file["Chemical yield"]
    chemical_yields_function_as_energy = interpolate.interp1d(energy, chemical_yields, kind= "linear",
                                                              fill_value="extrapolate")
    chemical_yields_primitive = scipy.integrate.cumtrapz(chemical_yields_function_as_energy(energy), energy, initial=0)
    chemical_yields_primitive_function = interpolate.interp1d(energy, chemical_yields_primitive, kind="linear")
    return chemical_yields_function_as_energy, chemical_yields_primitive_function


def _conversion_energy_in_let(data_base, energy, particle = 0):
    """
    Returns a function that converts an input energy into the corresponding LET from a given data base

    Input
    -------
    data_base in string format
    energy in keV
    """
    try:

        # Lithium
        if particle == 2 :
            resource_path = f"resources/E_TEL/conversion_tables_{data_base}_Li7.xlsx"

        # Helium
        else :
            resource_path = f"resources/E_TEL/conversion_tables_{data_base}_He.xlsx"

        file_content = pkg_resources.resource_stream(__name__, resource_path)
        tables__conversion_energy_in_let = pd.read_excel(file_content).to_records()
    except:
        raise RemovedConversionFileEinLET("Conversion tables of E in LET in the E_TEL folder are no longer present")
    energy_list = tables__conversion_energy_in_let['E(keV)']
    corresponding_let_list = tables__conversion_energy_in_let['LET(keV/um)']
    continuous_function_to_convert_energy_in_let = interpolate.interp1d(energy_list, corresponding_let_list,
                                                                        fill_value="extrapolate", kind= "linear")
    return continuous_function_to_convert_energy_in_let(energy)

def _moving_average_dn1_de_tables(dn1_de_from_tables, cell_line):
    """
    Returns a numpy array with alpha tables smoothered by moving average method
    """
    _temp_df = pd.DataFrame()
    _temp_df["dn1_de"] = dn1_de_from_tables
    cell_line_specific_window = 0
    if cell_line == "HSG": #HSG
        cell_line_specific_window = 7 #Manual adjustment of moving average
    elif cell_line == "V79": #V79
        cell_line_specific_window = 5
    elif cell_line == "CHO-K1": #CHO-K1
        cell_line_specific_window = 5
    moving_average_dn1_de = _temp_df['dn1_de'].rolling(window=cell_line_specific_window, center=True,
                                                     min_periods=1).mean()
    moving_average_dn1_de_np = moving_average_dn1_de.to_numpy()

    return moving_average_dn1_de_np

class UnexistingMethodAlphaCoeffExtrapolation(Exception):
    """
    If the method_threshold argument is wrong
    """
    pass

class RemovedConversionFileEinLET(Exception):
    """
    If the file to convert energy in let is not existing
    """
    pass

class InvalidOption(Exception):
    """
    Choose valid option
    """
    pass


