### Datasets pulled from Earth Engine:

AlphaEarth embeddings — GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
64 features (A00..A63) per pixel, derived from satellite imagery. 10 m resolution.

IrrMapper — UMT/Climate/IrrMapper_RF/v1_2
A mask of irrigated areas. Where the mask exists, we treat it as label = 1; elsewhere label = 0. 30 m resolution.

USDA Cropland Data Layer (CDL) — USDA/NASS/CDL
A land-cover map with cropland codes at 30 m resolution. We use this as a cropland mask (keep pixels where the code > 0) and also keep the code as a column (cdl_code) for reference.

States / Counties (TIGER) — define boundaries and attach county names/FIPS to each sampled point.

### What this script does:
For each state and year you ask for, it writes a CSV to your Drive for with rows as cropland pixels and the following columns:

64 AlphaEarth features (A00..A63)
label (0/1) from IrrMapper
cdl_code from CDL
year, state, counts, fips
point geometry (.geo)

### Options
random sampling over cropland (unbalanced) or stratified sampling to target a positive rate you pick (e.g., 30% irrigated).

There’s also an optional border buffer: e.g., when building training data for OR/ID/MT, you can carve out a 10-km strip next to WA so you’re not training on fields right across the border from your WA test area. This is to prevent spatial leakage (the alphaearth embeddings clearly contain geolocation information that I dont want the model to be reliant on)

### Masking and band alignment:

Process works like building one stacked image per state/year, then pulling points from it.

First, mosaic the AlphaEarth and IrrMapper image collections for that year and state boundary, then build a stack of bands:


1. A00..A63 (AlphaEarth features)

2. label (IrrMapper mask → 1 where irrigated, 0 elsewhere)

3. cdl_code (CDL cropland class code)

Then we mask the stack to cropland only using CDL (keep pixels where cdl_code > 0) to only sample agricultural land. 

Reproject each band (10 m and 30 m) to a common 30 m target scale at the query point. Return the value at that point for each band. For point sampling, this is effectively nearest-neighbor.

Returns the value at that point for each band. (For point sampling, this is effectively nearest-neighbor unless you explicitly change resampling.) This does not average 10 m pixels into a 30 m cell.
