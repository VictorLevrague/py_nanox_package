from setuptools import setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

alphas_tables_dir = 'nanox_low_energy/resources/AlphasTables'
alphas_tables_files = []
for file in os.listdir(alphas_tables_dir):
    if file.endswith('.csv'):
        alphas_tables_files.append(os.path.join(alphas_tables_dir, file))

e_tel_dir = 'nanox_low_energy/resources/E_TEL'
e_tel_files = []
for file in os.listdir(e_tel_dir):
    if file.endswith('.xlsx') or file.endswith('.ods'):
        e_tel_files.append(os.path.join(e_tel_dir, file))

chemical_yield_dir = 'nanox_low_energy/resources/ChemicalYield'
chemical_yield_files = []
for file in os.listdir(chemical_yield_dir):
    if file.endswith('.csv'):
        chemical_yield_files.append(os.path.join(chemical_yield_dir, file))


setup(
    name = 'nanox_low_energy',
    version = '1.1.2',
    description = "Calculates cell survivals with NanOx low energy hypotheses",
    author = 'Victor Levrague',
    packages = ["nanox_low_energy"],
    install_requires = required,
    package_data = {"nanox_low_energy": ['resources/*']},
    include_package_data = True,
    data_files=[('resources/AlphasTables', alphas_tables_files), ('resources/E_TEL', e_tel_files), ('resources/ChemicalYield', chemical_yield_files)]
)