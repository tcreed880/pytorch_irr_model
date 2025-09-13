# pyright: reportPrivateImportUsage=false
# ee_export_wa_irrmapper_alphaearth.py
import argparse
import ee

PROJECT_ID = "water-model"

# Defaults
YEARS = [2018, 2019, 2020, 2021, 2022]
POINTS_PER_CLASS = 5000                 # total per year = 2 * POINTS_PER_CLASS
EXPORT_DESC_PREFIX = "WA_alphaearth_irrmapper_unbalanced_val"
DRIVE_FOLDER = "ee_exports"
RANDOM_SEED = 89

# -------- EE init --------
def init_ee():
    """Initialize Earth Engine; authenticate only if needed."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

# -------- Lazy getters (create EE objects only AFTER init) --------
def get_states():
    return ee.FeatureCollection("TIGER/2018/States")

def get_WA():
    states = get_states()
    return states.filter(ee.Filter.eq("NAME", "Washington")).geometry()

def get_counties():
    # WA FIPS = 53
    return ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq("STATEFP", "53"))

def get_cdl_collection():
    return ee.ImageCollection("USDA/NASS/CDL")

def get_irrIC():
    return ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")

def get_aeIC():
    return ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

def get_cropland_fallback_2021():
    cdl_2021 = get_cdl_collection().filterDate("2021-01-01", "2022-01-01").first()
    return ee.Image(cdl_2021).select("cropland").gt(0).rename("cropland")

# -------- Helpers --------
def year_image(ic: ee.ImageCollection, year: int, aoi: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ic.filterDate(start, end).filterBounds(aoi).mosaic()

def get_cdl_cropland_code(year: int, WA: ee.Geometry) -> ee.Image:
    cdl_year = get_cdl_collection().filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(WA).first()
    cdl_2021 = get_cdl_collection().filterDate("2021-01-01", "2022-01-01").first()
    cdl_year = ee.Image(ee.Algorithms.If(cdl_year, cdl_year, cdl_2021))
    return ee.Image(cdl_year).select("cropland").rename("cdl_code")

def attach_county_fips(samples_fc: ee.FeatureCollection, counties: ee.FeatureCollection) -> ee.FeatureCollection:
    joined = ee.Join.saveBest(matchKey="county", measureKey="dist").apply(
        primary=samples_fc,
        secondary=counties,
        condition=ee.Filter.withinDistance(distance=1, leftField=".geo", rightField=".geo"),
    )

    def _pull(f):
        has_county = f.get("county")
        c = ee.Feature(f.get("county"))
        return ee.Feature(ee.Algorithms.If(
            has_county,
            ee.Feature(f).set({"county_fips": c.get("COUNTYFP"), "county_name": c.get("NAME")}),
            ee.Feature(f).set({"county_fips": "000", "county_name": "UNKNOWN"})
        ))

    return ee.FeatureCollection(joined).map(_pull)

def filter_counties_fc(counties_fc, only_fips=None, only_names=None):
    if only_fips:
        fips_list = [f"{int(f):03d}" for f in only_fips]
        counties_fc = counties_fc.filter(ee.Filter.inList("COUNTYFP", ee.List(fips_list)))
    if only_names:
        counties_fc = counties_fc.filter(ee.Filter.inList("NAME", ee.List(only_names)))
    return counties_fc

# -------- Export: per-year CSV sample (balanced or unbalanced) --------
def export_one_year(year: int, debug: bool = False, unbalanced: bool = False):
    WA = get_WA()
    counties = get_counties()
    cropland_fallback = get_cropland_fallback_2021()

    # label: 0/1 from IrrMapper mask
    irr = year_image(get_irrIC(), year, WA).select(0)
    label = irr.mask().rename("label").unmask(0).toInt()

    if debug:
        counts = label.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(), geometry=WA, scale=30, maxPixels=1e9, tileScale=2
        )
        print(f"[DEBUG] IrrMapper label histogram {year}: {counts.getInfo()}")

    # features
    ae_year = (get_aeIC()
               .filterDate(f"{year}-01-01", f"{year+1}-01-01")
               .filterBounds(WA)
               .mosaic()
               .select([f"A{i:02d}" for i in range(64)]))

    cdl_code = get_cdl_cropland_code(year, WA)

    # keep positives anywhere + CDL cropland elsewhere
    positives_mask = label.eq(1)
    cropland_mask = cropland_fallback.updateMask(cropland_fallback).clip(WA)
    keep_mask = cropland_mask.unmask(0).Or(positives_mask)

    stack = ae_year.addBands([label, cdl_code]).updateMask(keep_mask).clip(WA)

    total_points = 2 * POINTS_PER_CLASS  # 10,000 by default
    if unbalanced:
        # same total size, no per-class balancing
        samples = stack.sample(
            region=WA, numPixels=total_points, seed=RANDOM_SEED, scale=30, geometries=True
        ).map(lambda f: f.set({"year": year}))
        desc_suffix = "unbalanced"
    else:
        # stratified by 'label'
        samples = stack.stratifiedSample(
            numPoints=0,
            classBand="label",
            classValues=[0, 1],
            classPoints=[POINTS_PER_CLASS, POINTS_PER_CLASS],
            region=WA,
            geometries=True,
            seed=RANDOM_SEED,
            scale=30,
            tileScale=4,
        ).map(lambda f: f.set({"year": year}))
        desc_suffix = "balanced"

    samples = attach_county_fips(samples, counties)

    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f"{EXPORT_DESC_PREFIX}_{year}_{desc_suffix}",
        folder=DRIVE_FOLDER,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started export task ({desc_suffix}) for {year}. Check Tasks in Code Editor.")

# -------- Export: per-county multiband GeoTIFF --------
def export_all_cropland_year_by_county_image(year: int, counties_fc=None, scale_m=30, cloud_optimized=True):
    WA = get_WA()
    counties = counties_fc if counties_fc is not None else get_counties()
    cropland_mask = get_cropland_fallback_2021().updateMask(get_cropland_fallback_2021()).clip(WA)

    irr = year_image(get_irrIC(), year, WA).select(0)
    label = irr.mask().rename("label").unmask(0).toFloat()

    ae_year = (get_aeIC()
               .filterDate(f"{year}-01-01", f"{year+1}-01-01")
               .filterBounds(WA)
               .mosaic()
               .select([f"A{i:02d}" for i in range(64)])
               .toFloat())

    cdl_code = get_cdl_cropland_code(year, WA).toFloat()

    stack_all = (ae_year.addBands([label, cdl_code])
                        .updateMask(cropland_mask)
                        .toFloat())

    n = counties.size().getInfo()
    clist = counties.toList(n)
    names = counties.aggregate_array("NAME").getInfo()
    fips_list = counties.aggregate_array("COUNTYFP").getInfo()

    for i in range(n):
        c = ee.Feature(clist.get(i))
        name = names[i]
        fips = fips_list[i]
        region = c.geometry()

        task = ee.batch.Export.image.toDrive(
            image=stack_all.clip(region),
            description=f"WA_allpix_img_{year}_{fips}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"WA_allpix_img_{year}_{fips}",
            region=region,
            scale=scale_m,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": cloud_optimized},
        )
        task.start()
        print(f"Started county IMAGE export {year}: {name} ({fips}).")

# -------- CLI --------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export AlphaEarth + IrrMapper samples (CSV) or per-county GeoTIFFs for Washington State."
    )
    p.add_argument(
        "--mode",
        choices=["balanced", "all_image"],
        default="balanced",
        help="'balanced' = per-year CSV sample (use --unbalanced for random sample of same size); "
             "'all_image' = per-county multiband GeoTIFF.",
    )
    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=YEARS,
        help="Years to export, e.g., --years 2019 2020 2021",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print lightweight client-side diagnostics (e.g., label histogram).",
    )
    p.add_argument(
        "--unbalanced",
        action="store_true",
        help="For --mode balanced: draw the same total sample size (2*POINTS_PER_CLASS) without class balancing.",
    )
    # all_image filters/options
    p.add_argument("--only_fips", nargs="*", help="3-digit WA county FIPS codes (e.g., 033 035). Zero-padding optional.")
    p.add_argument("--only_names", nargs="*", help="County name(s), e.g., Kitsap Yakima; quote 'San Juan'.")
    p.add_argument("--scale", type=int, default=30, help="Export scale (meters) for --mode all_image (default 30).")
    p.add_argument("--no_cog", action="store_true", help="Disable cloud-optimized GeoTIFF for --mode all_image.")
    return p

def main():
    init_ee()
    args = build_argparser().parse_args()

    for y in args.years:
        if args.mode == "balanced":
            export_one_year(y, debug=args.debug, unbalanced=args.unbalanced)
        else:
            c_fc = filter_counties_fc(get_counties(), args.only_fips, args.only_names)
            count = c_fc.size().getInfo()
            if count == 0:
                print(f"No counties matched --only_fips/--only_names for {y}; skipping.")
                continue
            print(f"Exporting {count} county(ies) for {y}...")
            export_all_cropland_year_by_county_image(
                y, counties_fc=c_fc, scale_m=args.scale, cloud_optimized=(not args.no_cog)
            )

if __name__ == "__main__":
    main()

'''
Per-year balanced sample (10k rows):
poetry run python irr/cli/gee_python_api.py --mode balanced --years 2018 2019 2020

Same size but unbalanced:
poetry run python irr/cli/gee_python_api.py --mode balanced --years 2020 --unbalanced

Per-county GeoTIFF for one county by name:
poetry run python irr/cli/gee_python_api.py --mode all_image --years 2020 --only_names Kitsap

Per-county GeoTIFF by FIPS code:
poetry run python irr/cli/gee_python_api.py --mode all_image --years 2020 --only_fips 35

Multiple counties:
--only_fips 35 041 057 or --only_names Yakima "San Juan"
'''