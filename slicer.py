import h5py
import numpy as np
from scipy.io import savemat

file_path = './SSVEP_dataset/data_s2_64.mat'
output_file = '13hz_data.mat'

with h5py.File(file_path, 'r') as f:
    data = np.array(f['datas'])
    

    #index [0, 6:13, :, :, 0]
    extracted = data[1, 13-1, :, :, 1].T 

    extracted_array = np.array(extracted)

savemat(output_file, {'extracted_data': extracted_array})

print(f"Extracted data shape: {extracted_array.shape}")
print(f"Saved extracted data to {output_file}")
