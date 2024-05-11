import numpy as np
from PIL import Image
import nibabel as nib
from scipy import ndimage as ndimage
from skimage import measure

seg1 = nib.load('SA003/SA003-FLAIR-1.nii.gz')
d1 = seg1.get_fdata()


slice = 14 #this will be command line parameter
img = nib.load('Bmr14041970_t1_spc_sag_p2_iso_1.0_20190114045614_5.nii.gz')
seg = nib.load('lesion1_p2.nii.gz') #NOT NEEDED ANYMORE. (needed to based on an exisiting segmentation.)
data = seg.get_data()

data[:,:,list(range(259))+ list(range(260,512))] = 0
new_seg = nib.Nifti1Image(data, seg.affine, seg.header) #create a new segment based on the data of the original image
nib.save(new_seg, 'lesion1_p2_single_layer.nii.gz')
