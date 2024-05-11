from skimage.segmentation import chan_vese
from skimage.filters import median
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
import nibabel as nib


# helper functions
def load_case(file_path):
    """
    function to load case
    :return: case file, case data
    """
    case = nib.load(file_path)
    case_data = case.get_fdata()
    return case, case_data


def show_slice(data, slice_num=None, title="", cmap='gray', show=True, save=True, fname=None):
    """ shows the given slice """
    plt.figure()
    if slice_num:
        plt.imshow(np.fliplr(np.rot90(data[:, :, slice_num])), cmap=cmap, aspect='auto')
    else:
        plt.imshow(np.fliplr(np.rot90(data)), cmap='gray', aspect='auto')

    plt.title(title)
    if save:
        plt.savefig(fname=fname)
    if show:
        plt.show()


def process_slice(slice):
    """
    Some preprocessing of slice to  help the chan vese algorithm
    :param slice:
    :return:
    """
    slice = rescale_intensity(slice, out_range=(0, 255))
    slice = median(slice, selem=np.ones((5, 5)))  # passes a median filter

    return slice


def fill_seg(seg):
    """
    Finds largest connected component and fills any holes in it.
    :param seg:
    :return:
    """
    labels = label(seg)
    if labels.max() == 0:  # assume at least 1 CC
        return None, True
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    filled_seg = binary_fill_holes(largestCC)

    return filled_seg, False


def find_bounds(seg):
    """
    finds bounds of given segmentation
    :param seg:
    :return:
    """
    inds = np.where(seg != 0)
    x, y = inds[0], inds[1]
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)

    return x_min, x_max, y_min, y_max


def check_bounds(x_min_curr, x_max_curr, y_min_curr, y_max_curr, xMax, yMax):
    if x_min_curr < 0:
        x_min_curr = 0
    if y_min_curr < 0:
        y_min_curr = 0
    if x_max_curr > xMax:
        x_max_curr = xMax
    if y_max_curr > yMax:
        y_max_curr = yMax

    return x_min_curr, x_max_curr, y_min_curr, y_max_curr


def run():
    img_case, img = load_case('./0Pilot/Bmr14041970/Nifti/data/scan1.nii.gz')
    # _, mask = load_case('./0Pilot/Bmr14041970/Nifti/c5Bmr14041970_t1_spc_sag_p2_iso_1.0_20190114045614_5.nii')
    _, seg = load_case('./0Pilot/Bmr14041970/Nifti/data/scan1_seg_single_slice.nii.gz')
    # ground truth
    _, gtData = load_case('./0Pilot/Bmr14041970/Nifti/lesion1_p2.nii.gz')

    min_slice, max_slice = 226, 301  # axial bounds of tumor

    margin = 2  # margin to add around segmentation

    # find axial slice of initial segmentation.
    seg_ind = np.where(seg != 0)[2][0]
    seg_slice = seg[:, :, seg_ind]
    for mu in [0.15]:  # np.arange(0.1,0.7,0.05):
        error = False
        total_seg = np.zeros(img.shape)  # final output segmentation

        end_slice = max_slice
        step = 1

        for j in range(2):
            if error:
                break
            # find bounds of initial segmentation
            x_min_curr, x_max_curr, y_min_curr, y_max_curr = find_bounds(seg_slice)

            for i in range(seg_ind, end_slice, step):
                # print("slice", i)
                # extract ROI of slice
                x_min_curr, x_max_curr, y_min_curr, y_max_curr = check_bounds(x_min_curr, x_max_curr, y_min_curr,
                                                                              y_max_curr,
                                                                              seg_slice.shape[0], seg_slice.shape[1])
                curr_slice = img[:, :, i]
                # tmp_img = np.copy(img[:,:,i])
                # tmp_img[x_min_curr:x_max_curr,y_min_curr]=np.max(img)
                # tmp_img[x_min_curr:x_max_curr,y_max_curr-1]=np.max(img)
                # tmp_img[x_min_curr,y_min_curr:y_max_curr]=np.max(img)
                # tmp_img[x_max_curr,y_min_curr:y_max_curr]=np.max(img)
                # show_slice(tmp_img)
                # show_slice(curr_slice)

                img_roi = curr_slice[x_min_curr:x_max_curr, y_min_curr:y_max_curr]
                # img_roi[mask_slice[x_min_curr:x_max_curr, y_min_curr:y_max_curr]>0.9] = 0
                # before process
                show = False
                if show:
                    show_slice(img_roi)
                img_roi = process_slice(img_roi)
                # after process
                if show:
                    show_slice(img_roi)
                # find tumor in slice + add to final output
                temp_seg, phi, energies = chan_vese(img_roi, mu=mu, extended_output=True)
                if show:
                    show_slice(temp_seg)
                filled_seg, error = fill_seg(temp_seg)
                if show:
                    show_slice(filled_seg)
                if error:
                    break
                x_min_roi_next, x_max_roi_next, y_min_roi_next, y_max_roi_next = find_bounds(filled_seg)
                total_seg[x_min_curr:x_max_curr, y_min_curr:y_max_curr, i] = filled_seg
                show = False
                if show:
                    tmp_img = np.copy(img[:, :, i])
                    tmp_img[x_min_curr:x_max_curr, y_min_curr] = np.max(img)
                    tmp_img[x_min_curr:x_max_curr, y_max_curr - 1] = np.max(img)
                    tmp_img[x_min_curr, y_min_curr:y_max_curr] = np.max(img)
                    tmp_img[x_max_curr, y_min_curr:y_max_curr] = np.max(img)
                    tmp_img[x_min_curr:x_max_curr, y_min_curr:y_max_curr] += filled_seg * 250
                    show_slice(tmp_img)

                # order is important as min is used for max so first setup max
                x_max_curr = x_min_curr + x_max_roi_next + margin + 1
                x_min_curr += x_min_roi_next - margin
                y_max_curr = y_min_curr + y_max_roi_next + margin + 1
                y_min_curr += y_min_roi_next - margin

                if show:
                    tmp_img[x_min_curr:x_max_curr, y_min_curr] = np.max(img)
                    tmp_img[x_min_curr:x_max_curr, y_max_curr - 1] = np.max(img)
                    tmp_img[x_min_curr, y_min_curr:y_max_curr] = np.max(img)
                    tmp_img[x_max_curr, y_min_curr:y_max_curr] = np.max(img)
                    show_slice(tmp_img)

            end_slice = min_slice
            step = -1

        # A union B
        unionData = total_seg + gtData
        unionData[unionData > 0] = 1

        # A intersection B
        interData = np.sum(total_seg[gtData == 1])

        dice = interData * 2.0 / (np.sum(total_seg) + np.sum(gtData))
        vod = (1 - (interData / np.sum(unionData)))

        print(f'Mu - {mu} Dice - {int(dice * 100)}% VOD - {int(vod * 100)}%')
        if dice < 0.5:
            break

    # save final output
    # nii = nib.Nifti1Image(total_seg, img_case.affine, img_case.header)
    # nib.save(nii, 'seg.nii')


run()
