import numpy as np
import spectral_entropy

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)


# Calculate spectral entropy
entropy = spectral_entropy.calculate_entropy(spec_reference)
print("Spectral entropy is: {}.".format(entropy))
print('-' * 30)

# Clean the spectrum
spectrum = np.array([[41.04, 0.3716], [69.071, 7.917962], [69.070, 100.], [86.0969, 66.83]], dtype=np.float32)
clean_spectrum = spectral_entropy.clean_spectrum(spectrum,
                                                 max_mz=85,
                                                 noise_removal=0.01,
                                                 ms2_da=0.05)
print("Clean spectrum will be:{}".format(clean_spectrum))
print('-' * 30)

# Calculate spectral entropy
spec_query = spectral_entropy.clean_spectrum(spec_query)
entropy = spectral_entropy.calculate_entropy(spec_query)
print("Spectral entropy is: {}.".format(entropy))
print('-' * 30)

# Calculate entropy similarity.
similarity = spectral_entropy.calculate_entropy_similarity(spec_query, spec_reference, ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))
print('-' * 30)

# Another way to calculate entropy similarity, the result from this method is the same as the previous method.
similarity = spectral_entropy.similarity(spec_query, spec_reference, method="entropy", ms2_da=0.05)
print("Entropy similarity:{}.".format(similarity))
print('-' * 30)

# Calculate unweighted entropy distance.
similarity = spectral_entropy.similarity(spec_query, spec_reference, method="unweighted_entropy",
                                         ms2_da=0.05)
print("Unweighted entropy similarity:{}.".format(similarity))
print('-' * 30)

# Calculate all different spectral similarity.
all_dist = spectral_entropy.all_similarity(spec_query, spec_reference, ms2_da=0.05)
for dist_name in all_dist:
    method_name = spectral_entropy.methods_name[dist_name]
    print("Method name: {}, similarity score:{}.".format(method_name, all_dist[dist_name]))
