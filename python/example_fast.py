import numpy as np
import spectral_entropy_fast as spectral_entropy

spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]], dtype=np.float32)
spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]], dtype=np.float32)

# Calculate spectral entropy
entropy = spectral_entropy.spectral_entropy(spec_reference)
print("Spectral entropy is: {}.".format(entropy))
print('-' * 30)

# Clean the spectrum
spectrum = np.array([[41.04, 0.3716], [69.071, 7.917962], [69.070, 100.], [86.0969, 66.83]], dtype=np.float32)
clean_spectrum = spectral_entropy.clean_spectrum(spectrum,
                                                 max_mz=85,
                                                 noise_threshold=0.01,
                                                 ms2_da=0.05)
print("Clean spectrum will be:{}".format(clean_spectrum))
print('-' * 30)

# Calculate spectral entropy
spec_query = spectral_entropy.clean_spectrum(spec_query)
entropy = spectral_entropy.spectral_entropy(spec_query)
print("Spectral entropy is: {}.".format(entropy))
print('-' * 30)

# Calculate entropy similarity.
similarity = spectral_entropy.entropy_similarity(spec_query, spec_reference)
print("Entropy similarity:{}.".format(similarity))
print('-' * 30)
