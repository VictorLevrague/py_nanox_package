The nanox_low_energy package allows to calculate cell survivals under NanOx low energy hypotheses (cf WIP article). 

In NanOx, only damage to the nucleus can induce cell death. Then, the cell survivals depend only on the initial energy of an ion when it enters a nucleus, and its exit energy. Local lethal events and sublethal global events are considered.
For now, only helium ions are considered.

It contains the following functions:
- cell_survival_lethal: cell survival to lethal events only
- cell_survival_global: cell survival to global events only
- cell_survival_total: cell survival to lethal and global events
The requirements of these functions are: ei and ef (single number or array), the particle name, the Geant4 physics list considered ("em", or "dna"), and the option to calculate the survival. With the default "cumulated" option, all events are summed up to return one survival. With the "mean_to_one_impact", the survival to each couple of (Ei, Ef) will be returned.

Details of all the functions are in the py_nanox_low_energy.py file.


The package contains several excel files. Among them are the alpha coefficient tables, the conversions of energy into linear energy transfer and the chemical yield tables.
