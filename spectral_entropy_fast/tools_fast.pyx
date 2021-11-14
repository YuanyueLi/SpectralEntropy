# cython: infer_types=True

import numpy as np
cimport numpy as np
from libc.math cimport log2,log,pow

ctypedef np.float32_t float32
ctypedef np.int64_t int_64

cpdef int apply_weight_to_intensity(float32[:,::1] spectrum)nogil:
    """
    Apply the weight to the intensity
    """
    cdef double entropy=spectral_entropy(spectrum)
    cdef double weight, intensity_sum
    if entropy<3:
        weight = 0.25 + 0.25 * entropy
        intensity_sum = 0.
        for i in range(spectrum.shape[0]):
            spectrum[i,1] = pow(spectrum[i,1],weight)
            intensity_sum += spectrum[i,1]

        if intensity_sum>0:
            for i in range(spectrum.shape[0]):
                spectrum[i,1] /= intensity_sum
            return 1
    return 0


cpdef double spectral_entropy(float32[:,::1] spectrum) nogil:
    """
    Compute the spectral entropy of a spectrum.
    """
    cdef double entropy=0.
    cdef float32 intensity
    for i in range(spectrum.shape[0]):
        intensity=spectrum[i,1]
        if intensity>0:
            entropy+=-intensity*log(intensity)
    return entropy


cpdef double intensity_entropy(float32[:] intensity) nogil:
    """
    Compute the spectral entropy of a spectrum.
    """
    cdef double entropy=0.
    cdef float32 intensity_cur
    for i in range(intensity.shape[0]):
        intensity_cur=intensity[i]
        if intensity_cur>0:
            entropy+=-intensity_cur*log(intensity_cur)
    return entropy


cpdef double spectral_entropy_log2(float32[:,::1] spectrum) nogil:
    """
    Compute the spectral entropy of a spectrum.
    """
    cdef double entropy=0.
    cdef float32 intensity
    for i in range(spectrum.shape[0]):
        intensity=spectrum[i,1]
        if intensity>0:
            entropy+=-intensity*log2(intensity)
    return entropy


cpdef double merged_spectral_entropy_log2(float32[:,::1] spectrum_a,float32[:,::1] spectrum_b,ms2_ppm=None, ms2_da=None):
    """
    Calculate the unweighted entropy similarity between two spectra.
    Both spectra need to be centroided, sorted by m/z and their intensities need to be normalized to sum to 1.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is not None:
        return merged_spectral_entropy_log2_c(spectrum_a, spectrum_b, ms2_ppm=-1, ms2_da=ms2_da)
    else:
        return merged_spectral_entropy_log2_c(spectrum_a,spectrum_b,ms2_ppm=ms2_ppm,ms2_da=-1)


cpdef centroid_spectrum(float32[:,::1] spectrum,ms2_ppm=None, ms2_da=None):
    """
    Calculate centroid spectrum from a spectrum.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spectrum: The spectrum should be a 2D array with the first dimension being the m/z values and 
                     the second dimension being the intensity values.
                     The spectrum need to be sorted by m/z.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    if ms2_da is None:
        ms2_da = -1.
    else:
        ms2_ppm = -1.

    # Check whether the spectrum needs to be centroided or not.
    cdef int need_centroid = check_centroid_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    while need_centroid:
        spectrum = centroid_spectrum_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        need_centroid = check_centroid_c(spectrum, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    return np.asarray(spectrum)


cdef double merged_spectral_entropy_log2_c(float32[:,:] spec_a,float32[:,:] spec_b,double ms2_ppm,double ms2_da)nogil:
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef float32 peak_b_int = 0.
    cdef float32 mass_delta_da

    cdef double entropy = 0.
    cdef double peak_cur = 0.

    while a < spec_a.shape[0] and b < spec_b.shape[0]:
        if ms2_ppm > 0:
            ms2_da = ms2_ppm * 1e6 * spec_a[a,0]
        mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

        if mass_delta_da < -ms2_da:
            # Peak only existed in spec a.
            peak_cur = (spec_a[a, 1] + peak_b_int)/2.
            entropy += -peak_cur*log2(peak_cur)
            peak_b_int = 0.

            a += 1
        elif mass_delta_da > ms2_da:
            # Peak only existed in spec b.
            peak_cur = spec_b[b, 1]/2.
            entropy += -peak_cur*log2(peak_cur)

            b += 1
        else:
            # Peak existed in both spec.
            peak_b_int += spec_b[b, 1]
            b += 1

    if peak_b_int > 0.:
        peak_cur = (spec_a[a, 1] + peak_b_int)/2.
        entropy += -peak_cur*log2(peak_cur)
        peak_b_int = 0.
        a += 1

    # Fill the rest into merged spec
    for i in range(b,spec_b.shape[0]):
        peak_cur = spec_b[i, 1]/2.
        entropy += -peak_cur*log2(peak_cur)

    for i in range(a,spec_a.shape[0]):
        peak_cur = spec_a[i, 1]/2.
        entropy += -peak_cur*log2(peak_cur)

    return entropy



cdef centroid_spectrum_c(float32[:,::1] spec,double ms2_ppm, double ms2_da):
    """
    Calculate centroid spectrum from a spectrum.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.

    :param spec: the spectrum should be a 2D array with the first dimension being the m/z values and 
                    the second dimension being the intensity values.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    cdef int_64[:] intensity_order = np.argsort(spec[:, 1])
    cdef float32[:,::1] spec_new=np.zeros((spec.shape[0],2),dtype=np.float32,order='C')
    cdef int spec_new_i=0

    cdef double mz_delta_allowed
    cdef Py_ssize_t idx,x
    cdef Py_ssize_t i_left,i_right
    cdef float32 mz_delta_left,mz_delta_right,intensity_sum,intensity_weighted_sum

    with nogil:
        for x in range(intensity_order.shape[0]-1, -1, -1):
            idx = intensity_order[x]
            if ms2_da >= 0:
                mz_delta_allowed = ms2_da
            else:
                mz_delta_allowed = ms2_ppm * 1e-6 * spec[idx, 0]

            if spec[idx, 1] > 0:
                # Find left board for current peak
                i_left = idx - 1
                while i_left >= 0:
                    mz_delta_left = spec[idx, 0] - spec[i_left, 0]
                    if mz_delta_left <= mz_delta_allowed:
                        i_left -= 1
                    else:
                        break
                i_left += 1

                # Find right board for current peak
                i_right = idx + 1
                while i_right < spec.shape[0]:
                    mz_delta_right = spec[i_right, 0] - spec[idx, 0]
                    if mz_delta_right <= mz_delta_allowed:
                        i_right += 1
                    else:
                        break

                # Merge those peaks
                intensity_sum = 0
                intensity_weighted_sum = 0
                for i_cur in range(i_left, i_right):
                    intensity_sum += spec[i_cur, 1]
                    intensity_weighted_sum += spec[i_cur, 0]*spec[i_cur, 1]

                spec_new[spec_new_i, 0] = intensity_weighted_sum / intensity_sum
                spec_new[spec_new_i, 1] = intensity_sum
                spec_new_i += 1
                spec[i_left:i_right, 1] = 0

    spec_result = spec_new[:spec_new_i,:]
    spec_result=np.array(spec_result)[np.argsort(spec_result[:,0])]
    return spec_result



cdef int check_centroid_c(float32[:,::1] spectrum,double ms2_ppm, double ms2_da) nogil:
    """
    Check whether the spectrum needs to be centroided or not.
    At least one of the ms2_ppm or ms2_da need be not None. If both ms2_da and ms2_ppm is given, ms2_da will be used.
    
    :param spectrum: the spectrum should be a 2D array with the first dimension being the m/z values and 
                     the second dimension being the intensity values.
                     The spectrum need to be sorted by m/z.
                     The spectrum should be in C order.
    :param ms2_ppm: the mass accuracy in ppm.
    :param ms2_da: the mass accuracy in Da.
    """
    if spectrum.shape[0]<=1:
        return 0

    if ms2_da>0:
        # Use Da accuracy
        for i in range(1, spectrum.shape[0]):
            if spectrum[i, 0]-spectrum[i-1, 0] < ms2_da:
                return 1
    else:
        # Use ppm accuracy
        for i in range(1, spectrum.shape[0]):
            if spectrum[i, 0]-spectrum[i-1, 0] < spectrum[i, 0]*ms2_ppm*1e-6:
                return 1
    return 0
