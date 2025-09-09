# pyright: reportPrivateImportUsage=false
# ee_export_wa_irrmapper_alphaearth.py
import ee
import argparse


PROJECT_ID = "water-model"

def main():
    ee.Authenticate()     # run once and follow the link
    ee.Initialize(project=PROJECT_ID)


# set parameters
YEARS = [2018, 2019, 2020, 2021, 2022]  # IrrMapper v1_2 goes through 2022
POINTS_PER_CLASS = 5000      # per-year, per-class sample points, can adjust for memory issues
EXPORT_DESC = "WA_alphaearth_irrmapper_2018_2022"
DRIVE_FOLDER = "ee_exports"  # folder in my Drive

# get datasets (boundaries, crop mask, irrmapper, alphaearth):
# Washington boundary from TIGER censsus
states = ee.FeatureCollection("TIGER/2018/States")
WA = states.filter(ee.Filter.eq("NAME", "Washington")).geometry()

# get county data so we can do spatially grouped splits later
counties = ee.FeatureCollection("TIGER/2018/Counties") \
             .filter(ee.Filter.eq("STATEFP", "53"))  # WA FIPS

# using USDA CDL mask: cropland = any non-zero land cover code (exclude background)
cdl_2021 = ee.ImageCollection("USDA/NASS/CDL").filterDate("2021-01-01", "2022-01-01").first()
cropland = cdl_2021.select("cropland").gt(0).rename("cropland")

# "validation labels": IrrMapper (v1_2), annual collection (irrigation carried by MASK)
irrIC = ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")

# AlphaEarth Satellite Embedding (annual, 64 bands: A00 to A63 at 10m)
aeIC  = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")


# year mosaic by date 
def year_image(ic, year, aoi):
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")
    return ic.filterDate(start, end).filterBounds(aoi).mosaic()


# assigns county name to each pixel/point feature
def attach_county_fips(fc_points):
    # spatial join: for each point, find the county polygon it falls within
    joined = ee.Join.saveBest(matchKey="county", measureKey="dist").apply(
        primary=fc_points,
        secondary=counties,
        # spatial filter: find counties within 1 meter of point (point-in-polygon)
        condition=ee.Filter.withinDistance(
            distance=1, leftField=".geo", rightField=".geo"
        )
    )
    # adds the county code and name to each point
    def _pull(f):
        c = ee.Feature(f.get("county"))
        return ee.Feature(f).set({
            "county_fips": c.get("COUNTYFP"),
            "county_name": c.get("NAME")
        })
    return ee.FeatureCollection(joined).map(_pull)


def export_one_year(year: int):
    # IrrMapper label: use mask as the signal (unmasked = irrigated)
    irr = year_image(irrIC, year, WA).select(0)       # first band (robust if name varies)
    # label 1 where unmasked (irrigated), 0 elsewhere
    label = irr.mask().rename("label").unmask(0).toInt()

    # debug: histogram of labels for sanity
    counts = label.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=WA,
        scale=30,
        maxPixels=1e9,
        tileScale=2,
    )
    print("IrrMapper label histogram for", year, ":", counts.getInfo())

    # Alphaearth bands for this year (mosaic)
    ae_year_ic = aeIC.filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(WA)
    ae  = ae_year_ic.mosaic().select([f"A{i:02d}" for i in range(64)])  # A00..A63

    # align AE features to ~30 m like IrrMapper (snap to IrrMapper/CDL grid, bilinear for continuous)
    irr_proj = irr.projection()
    target_proj = irr_proj.atScale(30)
    ae = ae.resample('bilinear').reproject(crs=target_proj, scale=30)

    # add CDL crop type code for this year (aligned to the same 30 m grid)
    cdl_year = ee.ImageCollection("USDA/NASS/CDL") \
        .filterDate(f"{year}-01-01", f"{year+1}-01-01") \
        .filterBounds(WA) \
        .first()
    cdl_year = ee.Algorithms.If(cdl_year, cdl_year, cdl_2021)
    cdl_year = ee.Image(cdl_year).select("cropland").rename("cdl_code") \
        .reproject(crs=target_proj, scale=30)

    # apply Washington cropland mask to the alphaearth features and labels,
    # but keep positives even if CDL misses them 
    cropland_mask = cropland.updateMask(cropland).clip(WA)
    positives_mask = label.eq(1)
    keep_mask = cropland_mask.unmask(0).Or(positives_mask)

    stack = ae.addBands([label, cdl_year]).updateMask(keep_mask).clip(WA)

    # balanced stratified sample of points from the stack, with attached year property and lat/long geo
    samples = stack.stratifiedSample(
        numPoints=0,  # required placeholder when using classValues/classPoints
        classBand="label",
        classValues=[0, 1],
        classPoints=[POINTS_PER_CLASS, POINTS_PER_CLASS],
        region=WA,
        geometries=True,
        seed=88,
        scale=30
    ).map(lambda f: f.set({"year": year}))

    # attach county code and name to each point
    samples = attach_county_fips(samples)

    # export to drive as CSV (one task per year to avoid memory spikes)
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f"WA_alphaearth_irrmapper_{year}",
        folder=DRIVE_FOLDER,
        fileFormat="CSV"
    )
    task.start()
    print(f"Started export task for {year}. Monitor in https://code.earthengine.google.com/ → Tasks.")



def export_all_cropland_year_by_county(year: int):
    # --- same stack_all as above, with STRICT cropland mask and county_img band ---
    irr = year_image(irrIC, year, WA).select(0)
    label = irr.mask().rename("label").unmask(0).toInt()
    ae  = aeIC.filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(WA) \
              .mosaic().select([f"A{i:02d}" for i in range(64)])
    irr_proj = irr.projection()
    target_proj = irr_proj.atScale(30)
    ae = ae.resample('bilinear').reproject(crs=target_proj, scale=30)

    cdl_year = ee.ImageCollection("USDA/NASS/CDL").filterDate(f"{year}-01-01", f"{year+1}-01-01") \
                                                   .filterBounds(WA).first()
    cdl_year = ee.Image(ee.Algorithms.If(cdl_year, cdl_year, cdl_2021)) \
        .select("cropland").rename("cdl_code").reproject(crs=target_proj, scale=30)

    county_img = counties.reduceToImage(["COUNTYFP"], ee.Reducer.first()) \
                         .rename("county_fips").reproject(crs=target_proj, scale=30)

    cropland_only = cropland.updateMask(cropland).clip(WA)
    stack_all = ae.addBands([label, cdl_year, county_img]).updateMask(cropland_only).clip(WA)

    # Client-side loop over ~39 WA counties
    n = counties.size().getInfo()
    clist = counties.toList(n)
    for i in range(n):
        c = ee.Feature(clist.get(i))
        name = c.get("NAME").getInfo()
        fips = c.get("COUNTYFP").getInfo()

        fc = stack_all.sampleRegions(
            collection=ee.FeatureCollection([c]),
            scale=30,
            geometries=False,
            tileScale=2
        )

        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=f"WA_cropland_allpixels_{year}_{fips}_{name}",
            folder=DRIVE_FOLDER,
            fileFormat="CSV"
        )
        task.start()
        print(f"Started county export for {year}: {name} ({fips}), Monitor in https://code.earthengine.google.com/ → Tasks.")


    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["balanced", "all"], default="balanced",
                   help="'balanced' = your current stratified sample; 'all' = every cropland pixel (per county).")
    p.add_argument("--years", nargs="*", type=int, default=YEARS,
                   help="Years to export, e.g. --years 2019 2020 2021")
    args = p.parse_args()

    for y in args.years:
        if args.mode == "balanced":
            export_one_year(y)
        else:
            export_all_cropland_year_by_county(y)

if __name__ == "__main__":
    main()