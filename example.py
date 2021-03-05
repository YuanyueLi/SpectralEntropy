import numpy as np
import spectral_similarity

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

# Get entropy distance.
print('-' * 30)
similarity = spectral_similarity.similarity(spec_query, spec_reference, method="entropy",
                                            ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))

# Get dynamic weighted entropy distance.
print('-' * 30)
similarity = spectral_similarity.similarity(spec_query, spec_reference, method="unweighted_entropy",
                                            ms2_da=0.05)
print("Unweighted entropy similarity:{}.".format(similarity))

# Get all distances.
print('-' * 30)
all_dist = spectral_similarity.all_similarity(spec_query, spec_reference, ms2_da=0.05)
for dist_name in all_dist:
    method_name = spectral_similarity.methods_name[dist_name]
    print("Method name: {}, similarity score:{}.".format(method_name, all_dist[dist_name]))
