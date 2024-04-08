import os
import wfdb

# # Download MIT-BIH Arrhythmia Database
# if os.path.isdir("mitdb"):
#     print('MIT-BIH Arrhythmia Database: You already have the data.')
# else:
#     wfdb.dl_database('mitdb', 'mitdb')
#     print('MIT-BIH Arrhythmia Database: Download complete.')

# # Download MIT-BIH Noise Stress Test Database
# if os.path.isdir("nstdb"):
#     print('MIT-BIH Noise Stress Test Database: You already have the data.')
# else:
#     wfdb.dl_database('nstdb', 'nstdb')
#     print('MIT-BIH Noise Stress Test Database: Download complete.')

# Check if the PhysioNet/CinC Challenge Database is already downloaded
if os.path.isdir("challenge-2015"):
    print('PhysioNet/CinC Challenge Database already downloaded.')
else:
    # Download the PhysioNet/CinC Challenge Database
    wfdb.dl_database('challenge-2015', 'challenge-2015')

# Check if the PTB Diagnostic ECG Database is already downloaded
if os.path.isdir("ptbdb"):
    print('PTB Diagnostic ECG Database already downloaded.')
else:
    # Download the PTB Diagnostic ECG Database
    wfdb.dl_database('ptbdb', 'ptbdb')