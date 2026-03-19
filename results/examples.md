# Qualitative Examples (Defanged)

Source: `/Users/vayu/Documents/Playground/hybrid_url_detector/results/predictions_test.csv`. Threshold: `0.5`.

| example | url (defanged) | label | p_meta | p_electra_mean | p_electra_std | CI | uncertain | p_fusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Correct malicious (ELECTRA) | `hxxp://www[.]alliedmanagedcare[.]com/about-amc/our-locations/hawaii[.]html` | 1 | 0.953 | 0.977 | 0.003 | [0.976, 0.979] | 0 | 0.999 |
| Correct benign (ELECTRA) | `timesdispatch[.]com/news/local-news/2011/apr/26/tdmain01-bruce-jamerson-clerk-of-house-of-delegate-ar-995322/` | 0 | 0.007 | 0.022 | 0.003 | [0.020, 0.023] | 0 | 0.002 |
| False positive (ELECTRA) | `www[.]vodafone[.]co[.]uk[.]tilequest[.]com/` | 0 | 0.578 | 0.972 | 0.004 | [0.970, 0.974] | 0 | 0.933 |
| False negative (ELECTRA) | `naifanet[.]com/main-pub[.]cfm?usr=560000` | 1 | 0.043 | 0.025 | 0.004 | [0.022, 0.027] | 0 | 0.003 |
| Uncertain example | `encyclopedia[.]farlex[.]com/1902` | 0 | 0.122 | 0.503 | 0.413 | [0.247, 0.759] | 1 | 0.021 |
