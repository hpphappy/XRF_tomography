# XRF_tomography

To generate simulated data X-ray fluorescence tomography, choose one set of parameters in 3D/create_XRF_data.py and execute it. 
To generate simulated data X-ray transmission tomography, choose one set of parameters in 3D/create_XRT_data.py and execute it.

To run the reconstruction, use the Jupyter notebook file, 3D/reconstruct_joint_XRFT_size5.ipynb or 3D/reconstruct_joint_XRFT_size5.ipynb, to run the reconstruction for a 5x5x5 or a 64x64x64 object.

CAUTION: MPI is not applied yet, the reconstruction of a 64x64x64 object is not reccommended for now.
