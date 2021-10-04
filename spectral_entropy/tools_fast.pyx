import numpy as np
cimport numpy as np

ctypedef np.float32_t float32
ctypedef np.int64_t np_int_64


def centroid_spec(spec,ms2_ppm=None, ms2_da=None):
    if ms2_ppm is None:
        ms2_ppm=-1.
    if ms2_da is None:
        ms2_da=-1.
    spec=np.asarray(spec,dtype=np.float32,order="C")
    if spec.shape[0]>0:
        need_centroid = check_centroid_c(spec, ms2_ppm, ms2_da)
        while need_centroid:
            # Centroid spectrum
            spec = centroid_spec_c(spec, ms2_ppm, ms2_da)
            spec = np.asarray(spec,dtype=np.float32)
            spec = spec[np.argsort(spec[:, 0])]
            need_centroid = check_centroid_c(spec, ms2_ppm, ms2_da)
        return spec
    else:
        return np.asarray([],dtype=np.float32,order="C")


def clean_spectrum(spec,ms2_ppm=None, ms2_da=None):
    if ms2_ppm is None:
        ms2_ppm=-1.
    if ms2_da is None:
        ms2_da=-1.
    spec=np.asarray(spec,dtype=np.float32,order="C")
    if spec.shape[0]>0:
        return np.asarray(clean_spec_c(spec,ms2_ppm,ms2_da))
    else:
        return np.asarray([],dtype=np.float32,order="C")


def merge_spectrum(spec_a, spec_b, ms2_ppm=None, ms2_da=None) :
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is None:
        ms2_da=-1
    else:
        ms2_ppm=-1

    spec_a=np.asarray(spec_a,dtype=np.float32,order="C")
    spec_b=np.asarray(spec_b,dtype=np.float32,order="C")

    assert spec_a.shape[0]+spec_b.shape[0]>0
    return np.asarray(merge_spectrum_c(spec_a,spec_b,float(ms2_ppm),float(ms2_da)))

def match_spectrum(spec_a, spec_b, ms2_ppm=None, ms2_da=None) :
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is None:
        ms2_da=-1
    else:
        ms2_ppm=-1

    spec_a=np.asarray(spec_a,dtype=np.float32,order="C")
    spec_b=np.asarray(spec_b,dtype=np.float32,order="C")
    return np.asarray(match_spectrum_c(spec_a,spec_b,float(ms2_ppm),float(ms2_da)))

def match_spectrum_output_number(spec_a, spec_b, ms2_ppm=None, ms2_da=None) :
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    if ms2_da is None:
        ms2_da=-1
    else:
        ms2_ppm=-1

    spec_a=np.asarray(spec_a,dtype=np.float32,order="C")
    spec_b=np.asarray(spec_b,dtype=np.float32,order="C")
    return np.asarray(match_spectrum_output_number_c(spec_a,spec_b,float(ms2_ppm),float(ms2_da)))


def clean_spectrum_by_remove_0_intensity(spec):
    return np.asarray(clean_spectrum_by_remove_0_intensity_c(spec))

def normalize_spectrum_by_intensity_sum_in_place(spec):
    normalize_spectrum_by_intensity_sum_in_place_c(spec)

def sort_spectrum_by_mz_in_place(spec):
    """
    Sort the spectra in place by m/z
    :param spec: the dtype need to be np.float32
    """
    assert spec.dtype == np.float32
    assert spec.data.c_contiguous
    sort_spectrum_by_mz_in_place_c(spec)


cdef float32[:] merge_spectrum_c(float32[:,:] spec_a,float32[:,:] spec_b,float32 ms2_ppm,float32 ms2_da):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef float32[:] spec_merged = np.zeros(spec_a.shape[0]+spec_b.shape[0],dtype=np.float32)
    cdef int spec_merged_len=0
    cdef float32 peak_b_int = 0.
    cdef float32 mass_delta_da

    with nogil:
        while a < spec_a.shape[0] and b < spec_b.shape[0]:
            if ms2_ppm > 0:
                ms2_da = ms2_ppm * 1e6 * spec_a[a,0]
            mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

            if mass_delta_da < -ms2_da:
                # Peak only existed in spec a.
                spec_merged[spec_merged_len]=  spec_a[a, 1] + peak_b_int
                spec_merged_len+=1
                peak_b_int = 0.
                a += 1
            elif mass_delta_da > ms2_da:
                # Peak only existed in spec b.
                spec_merged[spec_merged_len]= spec_b[b, 1]
                spec_merged_len+=1
                b += 1
            else:
                # Peak existed in both spec.
                peak_b_int += spec_b[b, 1]
                b += 1

        if peak_b_int > 0.:
            spec_merged[spec_merged_len]= spec_a[a, 1] + peak_b_int
            spec_merged_len+=1
            peak_b_int = 0.
            a += 1

        # Fill the rest into merged spec
        for i in range(b,spec_b.shape[0]):
            spec_merged[spec_merged_len]= spec_b[i,1]
            spec_merged_len+=1

        for i in range(a,spec_a.shape[0]):
            spec_merged[spec_merged_len]= spec_a[i,1]
            spec_merged_len+=1

    # Shrink the merged spec.
    if spec_merged_len==0:
        spec_merged_len+=1
    spec_merged = spec_merged[:spec_merged_len]
    return spec_merged


cdef float32[:,:] match_spectrum_c(float32[:,:] spec_a,float32[:,:] spec_b,float32 ms2_ppm,float32 ms2_da):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef float32[:,:] spec_merged = np.zeros((spec_a.shape[0]+spec_b.shape[0],3),dtype=np.float32)
    cdef int spec_merged_len=0
    cdef float32 peak_b_int = 0.
    cdef float32 mass_delta_da

    with nogil:
        while a < spec_a.shape[0] and b < spec_b.shape[0]:
            if ms2_ppm > 0:
                ms2_da = ms2_ppm * 1e6 * spec_a[a,0]
            mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

            if mass_delta_da < -ms2_da:
                # Peak only existed in spec a.
                spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                    spec_a[a, 0], spec_a[a, 1], peak_b_int
                spec_merged_len+=1
                peak_b_int = 0.
                a += 1
            elif mass_delta_da > ms2_da:
                # Peak only existed in spec b.
                spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                    spec_b[b, 0], 0., spec_b[b, 1]
                spec_merged_len+=1
                b += 1
            else:
                # Peak existed in both spec.
                peak_b_int += spec_b[b, 1]
                b += 1

        if peak_b_int > 0.:
            spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                    spec_a[a, 0], spec_a[a, 1], peak_b_int
            spec_merged_len+=1
            peak_b_int = 0.
            a += 1

        # Fill the rest into merged spec
        for i in range(b,spec_b.shape[0]):
            spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                spec_b[i,0], 0., spec_b[i,1]
            spec_merged_len+=1

        for i in range(a,spec_a.shape[0]):
            spec_merged[spec_merged_len,0],spec_merged[spec_merged_len,1],spec_merged[spec_merged_len,2]= \
                spec_a[i,0],spec_a[i,1],0
            spec_merged_len+=1

    # Shrink the merged spec.
    if spec_merged_len==0:
        spec_merged_len+=1
    spec_merged = spec_merged[:spec_merged_len,:]
    return spec_merged


cdef np_int_64[:,:] match_spectrum_output_number_c(float32[:,:] spec_a,float32[:,:] spec_b,float32 ms2_ppm,float32 ms2_da):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    cdef int a = 0
    cdef int b = 0
    cdef int i

    cdef np_int_64[:,:] spec_merged = np.zeros((spec_a.shape[0]+spec_b.shape[0],2),dtype=np.int64)
    cdef int spec_merged_len=0
    cdef float32 peak_b_int = 0.
    cdef int peak_b_no=-1
    cdef float32 mass_delta_da

    with nogil:
    #if 1:
        while a < spec_a.shape[0] and b < spec_b.shape[0]:
            if ms2_ppm > 0:
                ms2_da = ms2_ppm * 1e6 * spec_a[a,0]
            mass_delta_da = spec_a[a, 0] - spec_b[b, 0]

            if mass_delta_da < -ms2_da:
                # Peak only existed in spec a.
                spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
                spec_merged_len += 1
                peak_b_int = 0.
                peak_b_no = -1
                a += 1
            elif mass_delta_da > ms2_da:
                # Peak only existed in spec b.
                spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = -1, b
                spec_merged_len += 1
                b += 1
            else:
                # Peak existed in both spec.
                if peak_b_int > 0:
                    if peak_b_int > spec_b[b, 1]:
                        # Use previous one
                        spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
                        spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = -1, b
                    else:
                        # Use this one
                        spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = -1, peak_b_no
                        spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, b
                    
                    spec_merged_len += 2
                    a += 1
                    peak_b_int = 0.
                    peak_b_no = -1
                else:
                    # Record this one
                    peak_b_int = spec_b[b, 1]
                    peak_b_no = b
                b += 1

        if peak_b_int > 0.:
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, peak_b_no
            spec_merged_len += 1
            a += 1

        # Fill the rest into merged spec
        for i in range(b,spec_b.shape[0]):
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = -1, b
            spec_merged_len += 1

        for i in range(a,spec_a.shape[0]):
            spec_merged[spec_merged_len,0], spec_merged[spec_merged_len,1] = a, -1
            spec_merged_len += 1

    # Shrink the merged spec.
    if spec_merged_len==0:
        spec_merged_len+=1
    spec_merged = spec_merged[:spec_merged_len,:]
    return spec_merged


cdef float32[:,:] clean_spec_c(float32[:,:]spec,float32 ms2_ppm=-1,float32 ms2_da=0.05):
    """
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    # Remove intensity==0
    spec=clean_spectrum_by_remove_0_intensity_c(spec)
    cdef int need_centroid

    if spec.shape[0]>1:
        # The spec need to be sorted by m/z
        sort_spectrum_by_mz_in_place_c(spec)
        # Fast check is the spectrum need centroid.
        need_centroid = check_centroid_c(spec, ms2_ppm, ms2_da)
        if need_centroid==1:
            # Centroid spectrum
            spec_new = centroid_spec_c(spec, ms2_ppm, ms2_da)
            sort_spectrum_by_mz_in_place_c(spec_new)
            spec=spec_new

    # Normalize the spectrum to sum(intensity)==1
    normalize_spectrum_by_intensity_sum_in_place_c(spec)

    return spec


cdef int check_centroid_c(float32[:,:]spec, float32 ms2_ppm, float32 ms2_da) :
    cdef int i
    cdef int not_need_centroid = 0
    cdef int need_centroid = 1
    cdef float32 mz_delta_cur

    if spec.shape[0]<=1:
        return not_need_centroid

    if ms2_da > 0:
        for i in range(1, spec.shape[0]):
            if spec[i, 0]-spec[i-1, 0] < ms2_da:
                return need_centroid
    elif ms2_ppm > 0:
        for i in range(1, spec.shape[0]):
            mz_delta_cur = spec[i, 0]-spec[i-1, 0]
            if mz_delta_cur/spec[i-1, 0]*1e6 <= ms2_ppm:
                return need_centroid
    else:
        return need_centroid
    return not_need_centroid


cdef float32[:,:] centroid_spec_c(float32[:,:]spec, float32 ms2_ppm, float32 ms2_da):
    # Centroid the spectrum
    cdef np_int_64[:] intensity_order = np.argsort(spec[:, 1])
    cdef float32[:,:] spec_new=np.zeros((spec.shape[0],2),dtype=np.float32)
    cdef int spec_new_i=0

    cdef float mz_delta_allowed
    cdef int idx,x
    cdef int i_left,i_right
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
    return spec_result



################################
cdef float32[:,:] clean_spectrum_by_remove_0_intensity_c(float32[:,:] spec):
    cdef int i,new_i

    # Test whether need remove intensity==0 or not.
    for i in range(spec.shape[0]):
        if spec[i,1]==0:
            break
    else:
        return spec

    # Remove intensity=0
    new_i=0
    cdef float32[:,:] spec_new=np.zeros((spec.shape[0],2),dtype=np.float32)
    for i in range(spec.shape[0]):
        if spec[i,1]>0:
            spec_new[new_i, 0], spec_new[new_i, 1] = spec[i, 0], spec[i, 1]
            new_i+=1
    spec = spec_new[:new_i,:]
    return spec


cdef void normalize_spectrum_by_intensity_sum_in_place_c(float32[:,:]spec) nogil:
    cdef np_int_64 i
    cdef float32 spec_sum = 0
    for i in range(spec.shape[0]):
        spec_sum+=spec[i,1]
    if spec_sum!=1:
        for i in range(spec.shape[0]):
            spec[i, 1] /= spec_sum
################################


################################
# Sort spectra
cdef void sort_spectrum_by_mz_in_place_c(float32[:,:] spec) nogil:
    cdef long n = spec.shape[0]
    cdef long new_n = 0
    cdef long i

    while n > 1:
        new_n = 0
        for i in range(1, n):
            if spec[i - 1][0] > spec[i][0]:
                spec[i - 1][0], spec[i][0] = spec[i][0], spec[i - 1][0]
                spec[i - 1][1], spec[i][1] = spec[i][1], spec[i - 1][1]
                new_n = i
        n = new_n


cdef void sort_spectrum_by_intensity_in_place_c(float32[:,:] spec) nogil:
    cdef long n = spec.shape[0]
    cdef long new_n = 0
    cdef long i

    while n > 1:
        new_n = 0
        for i in range(1, n):
            if spec[i - 1][1] > spec[i][1]:
                spec[i - 1][0], spec[i][0] = spec[i][0], spec[i - 1][0]
                spec[i - 1][1], spec[i][1] = spec[i][1], spec[i - 1][1]
                new_n = i
        n = new_n
################################
