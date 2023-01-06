import pandas as pd
import os

# Directory
directory = "test"
  
# Parent Directory path
parent_dir = './data/pandemic/centralized/'
data_dir = 'data-backup/'
  
# Path
path = os.path.join(parent_dir, directory)

if not os.path.isdir(path):
    os.mkdir(path)
    print("Directory '%s' created" %directory)

dn1 = 'va_activity_location_assignment.csv.gz'
dn2 = 'va_activity_locations.csv.gz'
dn3 = 'va_disease_outcome_target.csv.gz'
dn4 = 'va_disease_outcome_training.csv.gz'
dn5 = 'va_household.csv.gz'
dn6 = 'va_person.csv.gz'
dn7 = 'va_population_network.csv.gz'
dn8 = 'va_residence_locations.csv.gz'

datasets = [dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8]
NUM_ROWS = 2000

for i in range(len(datasets)):
    data = pd.read_csv(parent_dir+data_dir+datasets[i], compression='gzip',nrows = NUM_ROWS,error_bad_lines=False)
    data.to_csv(path+'/'+datasets[i], compression='gzip')