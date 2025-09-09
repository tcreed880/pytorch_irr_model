# pyright: reportPrivateImportUsage=false
# ee_export_wa_irrmapper_alphaearth.py
import argparse
import ee

PROJECT_ID = "water-model"

# Default params
YEARS = [2018, 2019, 2020, 2021, 2022]  # IrrMapper v1_2 goes through 2022
POINTS_PER_CLASS = 5000                 # per-year, per-class sample points
EXPORT_DESC_PREFIX = "WA_alphaearth_irrmapper"
DRIVE_FOLDER = "ee_exports"             # Google Drive folder
RANDOM_SEED = 88                        # deterministic sampling

# --- Datasets (declared at module scope; evaluated lazily by EE) ---

# WA boundary
states = ee.FeatureCollection("TIGER/2018/States")
WA = states.filter(ee.Filter.eq("NAME", "Washington")).geometry()

# Counties (for per-county export and attribution)
counties = ee.FeatureCollection("TIGER/2018/Counties") \
             .filter(ee.Filter.eq("STATEFP", "53"))  # WA FIPS

# USDA CDL — use 2021 cropland as fallback base mask
cdl_2021 = ee.ImageCollection("USDA/NASS/CDL") \
    .filterDate("2021-01-01", "2022-01-01").first()
cropland = ee.Image(cdl_2021).select("cropland").gt(0).rename("cropland")

# Labels: IrrMapper (v1_2), annual collection (irrigation signal carried by MASK)
irrIC = ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")

# AlphaEarth Satellite Embedding (annual, 64 bands: A00..A63 at 10 m)
aeIC = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")


# --- Helpers ---

def init_ee():
    """Initialize Earth Engine; authenticate only if needed."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def year_image(ic: ee.ImageCollection, year: int, aoi: ee.Geometry) -> ee.Image:
    """Mosaic a collection for a calendar year over an AOI."""
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ic.filterDate(start, end).filterBounds(aoi).mosaic()


def get_cdl_cropland_code(year: int) -> ee.Image:
    """Return the CDL 'cropland' band for the year (fallback to 2021) as 'cdl_code'."""
    cdl_year = ee.ImageCollection("USDA/NASS/CDL") \
        .filterDate(f"{year}-01-01", f"{year+1}-01-01") \
        .filterBounds(WA).first()
    cdl_year = ee.Image(ee.Algorithms.If(cdl_year, cdl_year, cdl_2021))
    return cdl_year.select("cropland").rename("cdl_code")


def attach_county_fips(samples_fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """Attach county FIPS and name to each point using a spatial join (null-safe)."""
    joined = ee.Join.saveBest(matchKey="county", measureKey="dist").apply(
        primary=samples_fc,
        secondary=counties,
        condition=ee.Filter.withinDistance(distance=1, leftField=".geo", rightField=".geo")
    )

    def _pull(f):
        # If no county found, set UNKNOWN values
        has_county = f.get("county")
        c = ee.Feature(f.get("county"))
        return ee.Feature(ee.Algorithms.If(
            has_county,
            ee.Feature(f).set({
                "county_fips": c.get("COUNTYFP"),
                "county_name": c.get("NAME")
            }),
            ee.Feature(f).set({"county_fips": "000", "county_name": "UNKNOWN"})
        ))

    return ee.FeatureCollection(joined).map(_pull)


# --- Export modes ---

def export_one_year(year: int, debug: bool = False):
    """Balanced stratified points (equal irrigated/non-irrigated), with county attribution."""
    # Label from IrrMapper mask (1 = irrigated, 0 otherwise)
    irr = year_image(irrIC, year, WA).select(0)  # first band robust to name changes
    label = irr.mask().rename("label").unmask(0).toInt()

    # Optional sanity check: label histogram (client-side)
    if debug:
        counts = label.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=WA, scale=30, maxPixels=1e9, tileScale=2
        )
        print(f"[DEBUG] IrrMapper label histogram {year}: {counts.getInfo()}")

    # AlphaEarth year (keep native projection; resample to smooth 10→30 m when sampling)
    ae_year = aeIC.filterDate(f"{year}-01-01", f"{year+1}-01-01") \
                  .filterBounds(WA).mosaic().select([f"A{i:02d}" for i in range(64)])

    # CDL cropland code (numeric), named 'cdl_code'
    cdl_code = get_cdl_cropland_code(year)

    # Keep positives anywhere + CDL cropland elsewhere
    cropland_mask = cropland.updateMask(cropland).clip(WA)
    positives_mask = label.eq(1)
    keep_mask = cropland_mask.unmask(0).Or(positives_mask)

    # Build stack; avoid forcing an explicit reproject — let EE handle scale at sampling
    stack = ae_year.addBands([label, cdl_code]).updateMask(keep_mask).clip(WA)

    # Stratified sampling balanced on 'label'
    samples = stack.stratifiedSample(
        numPoints=0,  # placeholder when using classValues/classPoints
        classBand="label",
        classValues=[0, 1],
        classPoints=[POINTS_PER_CLASS, POINTS_PER_CLASS],
        region=WA,
        geometries=True,   # keep points to allow county join
        seed=RANDOM_SEED,
        scale=30,          # sample at ~30 m
        tileScale=4
    ).map(lambda f: f.set({"year": year}))

    # County attribution
    samples = attach_county_fips(samples)

    # Export (short, deterministic description)
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f"{EXPORT_DESC_PREFIX}_{year}",
        folder=DRIVE_FOLDER,
        fileFormat="CSV"
    )
    task.start()
    print(f"Started export task (balanced) for {year}. Check Tasks in Code Editor.")


def export_all_cropland_year_by_county(year: int):
    """Export every cropland pixel per county (strict cropland mask)."""
    # Prepare per-year stack (strict cropland)
    irr = year_image(irrIC, year, WA).select(0)
    label = irr.mask().rename("label").unmask(0).toInt()
    ae_year = aeIC.filterDate(f"{year}-01-01", f"{year+1}-01-01") \
                  .filterBounds(WA).mosaic().select([f"A{i:02d}" for i in range(64)])
    cdl_code = get_cdl_cropland_code(year)
    cropland_only = cropland.updateMask(cropland).clip(WA)

    stack_all = ae_year.addBands([label, cdl_code]).updateMask(cropland_only).clip(WA)

    # Fetch county metadata once (two client calls total)
    n = counties.size().getInfo()
    clist = counties.toList(n)
    names = counties.aggregate_array("NAME").getInfo()
    fips_list = counties.aggregate_array("COUNTYFP").getInfo()

    for i in range(n):
        c = ee.Feature(clist.get(i))
        name = names[i]
        fips = fips_list[i]

        # Sample all pixels within the county polygon at ~30 m
        fc = stack_all.sampleRegions(
            collection=ee.FeatureCollection([c]),
            scale=30,
            geometries=False,
            tileScale=4
        ).map(lambda f: f.set({"county_name": name, "county_fips": fips, "year": year}))

        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=f"WA_allpix_{year}_{fips}",  # shorter description to keep UI clean
            folder=DRIVE_FOLDER,
            fileFormat="CSV"
        )
        task.start()
        print(f"Started county export {year}: {name} ({fips}). Check Tasks in Code Editor.")


# --- CLI / Main ---

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export AlphaEarth + IrrMapper samples for WA (balanced or all-pixels-by-county)."
    )
    p.add_argument(
        "--mode",
        choices=["balanced", "all"],
        default="balanced",
        help="'balanced' = stratified irrigated/non-irrigated sample; 'all' = every cropland pixel per county."
    )
    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=YEARS,
        help="Years to export, e.g., --years 2019 2020 2021"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print lightweight client-side diagnostics (e.g., label histogram)."
    )
    return p


def main():
    init_ee()

    args = build_argparser().parse_args()

    for y in args.years:
        if args.mode == "balanced":
            export_one_year(y, debug=args.debug)
        else:
            export_all_cropland_year_by_county(y)


if __name__ == "__main__":
    main()
