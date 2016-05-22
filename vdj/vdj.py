import numpy as np
import scipy.ndimage
import scipy.spatial

import skimage.feature
import skimage.filters
import skimage.morphology
import skimage.segmentation


def good_frames(ic, frac_diff=0.95, bad_frames=[], return_covs=False):
    """
    Determine which frames in an list of images are in focus.

    Parameters
    ----------
    ic : list of ndarrays
        List of images. This is *not* a skimage.ImageCollection.
    frac_diff : float, default = 0.95
        Frames whose coefficient of variation is less than frac_diff
        times the mean of the coefficient of variation of all frames
        are considered bad.
    bad_frame : list, default []
        List of indices of frames that are known to be bad.
    return_covs : bool
        If True, also return coefficients of varation of frames.

    Returns
    -------
    output : list of ints
        List of indices of good frames.
    covs : ndarray, only returns if return_covs is True
        Array of coefficient of variation of each frame.
    """
    covs = np.array([im.std() / im.mean() for im in ic])
    thresh = frac_diff * covs.mean()
    gf = [i for i in range(len(ic))
                    if covs[i] > thresh and i not in bad_frames]

    if return_covs:
        return gf, covs
    else:
        return gf


def preprocess(im, sigma=1, selem_width=5, neg_im=False):
    """
    Perform bluring and background subtraction of image.

    Parameters
    ----------
    im : ndarray
        Image to preprocess.
    sigma : float
        The characteristic length scale of noise in the images
        in units of pixels.  This is used to set the sigma value
        in the Gaussian blur.
    selem_width : int
        Width of square structuring element for mean filter.  Should be an
        odd integer greater than the pixel radius, but smaller than
        interparticle distance.
    neg_im : bool
        If True, use the negative of the image.

    Returns
    -------
    output : ndarray, shape as im, dtype float
        Blurred, background subtracted image.
    """

    # Convert image to float
    im = skimage.img_as_float(im)

    # Make the negative
    if neg_im:
        im = im.max() - im

    # Set up structuring element
    selem = skimage.morphology.square(selem_width)

    # Perform a mean filter (this is much faster than skimage.filters.rank.mean)
    im_bg = scipy.ndimage.uniform_filter(im, selem_width)

    # Gaussian blur
    im_blur = skimage.filters.gaussian(im, sigma)

    # Background subtract
    bg_subtracted_image = im_blur - im_bg

    # Return with negative pixels set to zero
    return np.maximum(bg_subtracted_image, 0)


def get_peaks(im, particle_size=3, thresh_abs=None, thresh_perc=None):
    """
    Find local maxima in pixel intensity.  Designed for use
    with images containing particles of a given size.

    Parameters
    ----------
    im : array_like
        Image in which to find local maxima.
    particle_size : int, default 3
        Diameter of particles in units of pixels.
    thresh_abs : int or float
        Minimum intensity of peaks. Overrides thresh_perc. If both
        thresh_abs and thresh_perc are None, Otsu's method is used to
        determine threshold.
    thresh_perc : float in range of [0, 100], default None
        Only pixels with intensities above the thresh_perc
        percentile are considered.  Default = 70.  Ignored if
        thresh_abs is not None.

    Returns
    -------
    output: ndarray of bools
        Boolean array shaped like image, with peaks represented by
        True values.
    """

    if thresh_abs is None:
        if thresh_perc is None:
            thresh_abs = skimage.filters.threshold_otsu(im)
        else:
            thresh_abs = np.percentile(im, thresh_perc)

    return skimage.feature.peak_local_max(
        im, min_distance=particle_size, threshold_abs=thresh_abs, indices=False,
        exclude_border=True)


def get_all_peaks(ic, particle_size=3, thresh_abs=None, thresh_perc=None):
    """
    Find local maxima in pixel intensity for each image in a list of
    images. Designed for use with images containing particles of a
    given size.

    Parameters
    ----------
    ic : list of images
        Images in which to find local maxima.
    particle_size : int, default 3
        Diameter of particles in units of pixels.
    thresh_abs : int or float
        Minimum intensity of peaks. Overrides thresh_perc. If both
        thresh_abs and thresh_perc are None, Otsu's method is used to
        determine threshold.
    thresh_perc : float in range of [0, 100], default None
        Only pixels with intensities above the thresh_perc
        percentile are considered.  Default = 70.  Ignored if
        thresh_abs is not None.

    Returns
    -------
    output: list of ndarray of bools
        List of Boolean arrays shaped like image, with peaks
        represented by True values.
    """

    return [get_peaks(im, particle_size=3, thresh_abs=None, thresh_perc=None)
                for im in ic]


def peak_rois(peaks, r):
    """
    Draw square ROIs around peaks.

    Parameters
    ----------
    peaks : ndarray, type bool
        Boolean array shaped like image, with peaks represented by
        True values.
    r : int
        Radius of ROI.  The ROIs are squares centered at the
        respective peak with side length 2*r + 1.

    Returns
    -------
    output : list of slice objects
        A list of slice objects corresponding to ROIs.
    """

    # Get peaks as indices
    peaks_ind = np.transpose(peaks.nonzero())

    # Dimension of image
    im_shape = peaks.shape

    # Get ROIs around peaks
    rois = []
    for peak in peaks_ind:
        if peak[0] >= r and peak[0] < im_shape[0] - r \
                and peak[1] >= r and peak[1] < im_shape[1] - r:
            rois.append(np.s_[peak[0]-r:peak[0]+r+1, peak[1]-r:peak[1]+r+1])
    return rois


def centroid_mag(im):
    """
    Return magnitude of the centroid of an image

    Parameters
    ----------
    im : ndarray
        Image for which to compute centroid

    Returns
    -------
    output : float
        Magnitude of the centroid.
    """
    m = skimage.measure.moments(im, order=1)
    return np.sqrt(m[0,1]**2 + m[1,0]**2) / m[0,0]


def filter_rois(ic, rois, thresh_std=(1, 1)):
    """
    Draw square ROIs around peaks.

    Parameters
    ----------
    ic : list of images
        Images to use in analysis.
    rois : list of slice objects
        List of ROIs to consider.
    thresh_std : 2-tuple
        If m is the mean intensity present in all ROIs in ic[0]
        and s is the standard deviation of intensities in the ROIs
        in ic[0], them we only keep ROIs with total intensity between
        m - thresh_std[0] * s and m + thresh_std[1] * s.

    Returns
    -------
    filtered_rois : list of slice objects
        List of ROIs to consider in analysis
    fiducial_rois : list of slice objects
        List of ROIs to use for fiducial markers
    """

    # Comute intensities in ROI
    roi_ints = np.array([ic[0][roi].sum() for roi in rois])

    # Determine thresholds
    thresh_high = roi_ints.mean() + thresh_std[0] * roi_ints.std()
    thresh_low = roi_ints.mean() - thresh_std[1] * roi_ints.std()

    # Only keep ROIs with intensity within threshold
    return [roi for i, roi in enumerate(rois)
                    if roi_ints[i] < thresh_high and roi_ints[i] > thresh_low]


def fiducial_beads(ic, rois, peaks, n_frames=5, n_fiducial=11):
    """
    Draw square ROIs around peaks.

    Parameters
    ----------
    ic : list of images
        Images to use in analysis.
    rois : list of slice objects
        List of ROIs to consider.
    peaks : ndarray, shape (n_peaks, 2)
        List of indicies of peaks in first frame of stack of images.
    n_frames : int, default 5
        Number of frames to use to determing fiducial ROIs
    n_fiducial : int, default 11 (must be odd)
        Number of fiducial ROIs to keep

    Returns
    -------
    output : ndarray, shape (n_fiducial, 2)
        Array of indices of fiducial beads
    """

    if n_fiducial % 2 == 0:
        raise RuntimeError('n_fiducial must be odd.')

    # Convert peaks to indices
    peaks_ind = np.transpose(peaks.nonzero())

    # Compute centroids of ROIS
    covs = np.empty(len(rois))
    for i, roi in enumerate(rois):
        centroids = np.array([centroid_mag(im[roi]) for im in ic[:n_frames]])
        covs[i] = centroids.std() / centroids.mean()

    # Sort coefficients of variation to get indicies of still ROIs
    roi_inds = np.argsort(covs)

    # Find positions of beads in fiducial ROIs
    i = 0
    j = 0
    fid_peaks = np.empty((n_fiducial, 2), dtype=np.int)
    while i < len(roi_inds) and j  < n_fiducial:
        roi = rois[roi_inds[i]]

        # Find peaks in ROI (only use ROI with a single bead)
        ind = np.where((roi[0].start <= peaks_ind[:,0])
                     & (peaks_ind[:,0] < roi[0].stop)
                     & (roi[1].start <= peaks_ind[:,1])
                     & (peaks_ind[:,1] < roi[1].stop))[0]
        if len(ind) == 1:
            fid_peaks[j,:] = peaks_ind[ind[0],:]
            j += 1
        i += 1

    if j < n_fiducial:
        raise RuntimeError('Failed to find %d fiducial peaks.' % n_fiducial)

    return fid_peaks


def drift_correct(ic, peaks, fid_peaks):
    """
    Perform drift correction
    """
    # Convert to indices
    peaks_ind = [np.transpose(peak.nonzero()) for peak in peaks]

    # Initialize shifts and peak positions
    fid_0 = fid_peaks.copy()
    shift = np.array([0, 0])


    ic_out = [ic[0]]
    for i, im in enumerate(ic[1:]):
        # Make KD tree for current peak locations
        kd = scipy.spatial.KDTree(peaks_ind[i])

        # Find the index closest to fiducial peaks
        fid = np.array([peaks_ind[i][kd.query(fp)[1]] for fp in fid_0])

        # Compute drift and shift
        drift_mag = (fid[:,0] - fid_0[:,0])**2 + (fid[:,1] - fid_0[:,1])**2
        j = np.argwhere(drift_mag == np.median(drift_mag))[0][0]
        shift += np.array([fid[j,0] - fid_0[j,0], fid[j,1] - fid_0[j,1]])

        print(drift_mag)

        # Make shifted output image
        pad = np.abs(shift).max()
        im_out = np.pad(im, np.abs(shift).max(), 'constant')
        ic_out.append(im[pad+shift[0]:pad+shift[0]+im.shape[0],
                         pad+shift[1]:pad+shift[1]+im.shape[1]])

        # Update fiducial points
        fid_0 = fid

    return ic_out
