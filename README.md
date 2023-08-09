# AminoSwitch
You can find the datapoints and information for classified switch residues through β2AR receptor here: 
https://drive.google.com/file/d/13Hmu9U-cHUlgrlKw9ow4ONeIZJtMuWgJ/view?usp=sharing
The datapoints for switch residues across Fs_peptide protein is in switch_residues_Fs_peptide_traj#15.pkl file.
The fs_peptide_trajectory.zip file contains the trajectory#15 downloaded from this resourse: https://doi.org/10.6084/m9.figshare.1030363.v1
The resourse for β2AR receptor trajectories used in our study is: https://exhibits.stanford.edu/data/catalog/vp873ky1987
You can find all the switch residues within any protein with this script: switch_all_over_new_protein.py
where the training dataset for the ML classifies is provided in this dataset: training_dataset_bimodal_switches.csv
The conformation structures used in our study for Fs-peptide and β2AR receptor are fs-peptide.pdb and B2_active_crystal_reference.pdb, respectively.
The Instability Ratio.py script measures the Instability ratio for each angle dataset and the Logistic Regression model predic if it is SS or US residue.
The training dataset for the Logistic Regression model is training_dataset_SS_US.csv
