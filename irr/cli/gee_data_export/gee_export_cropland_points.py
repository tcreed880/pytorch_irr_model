# pyright: reportPrivateImportUsage=false
# ee_export_cropland_points.py
import argparse
import ee

# -------------- Defaults --------------
PROJECT_ID = "water-model"
DEFAULT_YEARS = [2018, 2019, 2020, 2021, 2022]
DEFAULT_STATES = ["WA", "OR", "ID", "MT"]
DEFAULT_POINTS = 20000
DEFAULT_SEED = 90
DEFAULT_DESC_PREFIX = "alphaearth_irrmapper_training"
DEFAULT_DRIVE_FOLDER = "ee_exports"

# -------------- Earth Engine init --------------
def init_ee():
    """Initialize Earth Engine; authenticate only if needed."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

# -------------- EE helpers --------------
def get_states_fc():
    return ee.FeatureCollection("TIGER/2018/States")

def state_geom_by_abbr(abbr: str) -> ee.Geometry:
    f = get_states_fc().filter(ee.Filter.eq("STUSPS", abbr)).first()
    return ee.Feature(f).geometry()

def state_fips_by_abbr(abbr: str) -> str:
    f = get_states_fc().filter(ee.Filter.eq("STUSPS", abbr)).first()
    return ee.String(ee.Feature(f).get("STATEFP")).getInfo()  # client-side (once per state)

def counties_for_state(abbr: str) -> ee.FeatureCollection:
    states = get_states_fc()
    st = states.filter(ee.Filter.eq("STUSPS", abbr)).first()
    # server-side get of STATEFP (no .getInfo)
    st_fp = ee.Feature(st).get("STATEFP")
    counties = ee.FeatureCollection("TIGER/2018/Counties")
    return counties.filter(ee.Filter.eq("STATEFP", st_fp))

def get_cdl_collection():
    return ee.ImageCollection("USDA/NASS/CDL")

def get_irrIC():
    return ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")

def get_aeIC():
    return ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

def year_image(ic: ee.ImageCollection, year: int, aoi: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ic.filterDate(start, end).filterBounds(aoi).mosaic()

def get_cdl_cropland_code(year: int, aoi: ee.Geometry, force_2021: bool = False) -> ee.Image:
    if force_2021:
        cdl = get_cdl_collection().filterDate("2021-01-01", "2022-01-01").first()
    else:
        cdl_year = get_cdl_collection().filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(aoi).first()
        cdl_2021 = get_cdl_collection().filterDate("2021-01-01", "2022-01-01").first()
        cdl = ee.Image(ee.Algorithms.If(cdl_year, cdl_year, cdl_2021))
    return ee.Image(cdl).select("cropland").rename("cdl_code")

def attach_county_fips(samples_fc: ee.FeatureCollection, counties: ee.FeatureCollection) -> ee.FeatureCollection:
    # nearest-county join
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

# -------------- Main exporter --------------
def export_random_cropland_for_state_year(
    abbr: str,
    year: int,
    points: int,
    *,
    seed: int = DEFAULT_SEED,
    desc_prefix: str = DEFAULT_DESC_PREFIX,
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    balance: str = "random",            # "random" or "stratified"
    pos_frac: float = 0.5,              # used when balance="stratified": desired fraction of positives
    exclude_near_state: str | None = None,
    buffer_m: int = 0,
    force_mask_2021: bool = False,
    debug: bool = False,
):
    """
    Export a CSV of cropland points for a given state-year with AlphaEarth features + IrrMapper label.

    Args:
        abbr: state USPS code (e.g., "WA").
        year: e.g., 2018..2022.
        points: total number of points to sample.
        seed: sampling seed.
        desc_prefix: export description prefix.
        drive_folder: Google Drive folder name.
        balance: "random" (unbalanced) or "stratified" (by IrrMapper label).
        pos_frac: target positive fraction when balance="stratified" (0..1).
        exclude_near_state: if provided, carve out a buffer around this state from the AOI (reduce proximity leakage).
        buffer_m: width (meters) of the exclusion buffer.
        force_mask_2021: if True, use CDL 2021 cropland mask for all years (stable mask).
        debug: print small diagnostics (label histogram).
    """
    # AOI (optionally remove a border buffer next to another state)
    aoi = state_geom_by_abbr(abbr)
    if exclude_near_state:
        target = state_geom_by_abbr(exclude_near_state)
        if buffer_m and buffer_m > 0:
            aoi = aoi.difference(target.buffer(buffer_m))
        else:
            aoi = aoi.difference(target)

    counties = counties_for_state(abbr)

    # Label from IrrMapper (1 where irrigated mask exists, else 0)
    irr = year_image(get_irrIC(), year, aoi).select(0)
    label = irr.mask().rename("label").unmask(0).toInt()

    # Features: AlphaEarth embeddings A00..A63 for that year
    ae_year = (get_aeIC()
               .filterDate(f"{year}-01-01", f"{year+1}-01-01")
               .filterBounds(aoi)
               .mosaic()
               .select([f"A{i:02d}" for i in range(64)]))

    # Cropland mask from CDL
    cdl_code = get_cdl_cropland_code(year, aoi, force_2021=force_mask_2021)
    cropland_mask = cdl_code.gt(0)  # cropland = code > 0

    # Stack and restrict to cropland
    stack = (ae_year.addBands([label, cdl_code.rename("cdl_code")])
                     .updateMask(cropland_mask)
                     .clip(aoi))

    # Seed per state-year (string concat keeps it deterministic but distinct by year)
    seed_sy = int(f"{seed}{year}") if year < 2100 else seed

    # Sampling
    if balance == "stratified":
        # classPoints for negatives (0) and positives (1)
        pos_points = int(max(0, min(points, round(points * float(pos_frac)))))
        neg_points = int(max(0, points - pos_points))
        samples = (stack.stratifiedSample(
            numPoints=0,
            classBand="label",
            classValues=[0, 1],
            classPoints=[neg_points, pos_points],
            region=aoi,
            seed=seed_sy,
            scale=30,
            tileScale=4,
            geometries=True,
        ))
        balance_tag = f"strat_pos{pos_points}_neg{neg_points}"
    else:
        # unbalanced / random over cropland
        samples = (stack.sample(
            region=aoi,
            numPixels=points,
            seed=seed_sy,
            scale=30,
            geometries=True,
        ))
        balance_tag = "random"

    # Annotate & attach county info
    samples = samples.map(lambda f: f.set({"year": year, "state": abbr}))
    samples = attach_county_fips(samples, counties)

    # Optional quick histogram
    if debug:
        try:
            hist = samples.aggregate_histogram("label").getInfo()
            print(f"[DEBUG] {abbr} {year} label hist:", hist)
        except Exception:
            pass

    # Export
    desc = f"{abbr}_{desc_prefix}_{year}_{balance_tag}_n{points}"
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=desc,
        folder=drive_folder,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started export: {desc}")

# -------------- CLI --------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export AlphaEarth embeddings + IrrMapper labels at random cropland points per state/year."
    )
    p.add_argument("--states", nargs="*", default=DEFAULT_STATES,
                   help="State USPS codes, e.g., WA OR ID MT.")
    p.add_argument("--years", nargs="*", type=int, default=DEFAULT_YEARS,
                   help="Years to export, e.g., --years 2018 2019 2020 2021 2022")
    p.add_argument("--points", type=int, default=DEFAULT_POINTS,
                   help="Total points per state-year (default 20000).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed (default 90).")
    p.add_argument("--desc-prefix", type=str, default=DEFAULT_DESC_PREFIX, help="Export description prefix.")
    p.add_argument("--drive-folder", type=str, default=DEFAULT_DRIVE_FOLDER, help="Drive folder for CSV exports.")

    # balancing
    p.add_argument("--balance", choices=["random", "stratified"], default="random",
                   help="Sampling mode: random (unbalanced) or stratified by label.")
    p.add_argument("--pos-frac", type=float, default=0.5,
                   help="Target fraction of positives when --balance=stratified (e.g., 0.5 for 50/50).")

    # proximity leakage control
    p.add_argument("--exclude-near-state", type=str, default=None,
                   help="If set, exclude a buffer around this state from the AOI (e.g., 'WA').")
    p.add_argument("--buffer-m", type=int, default=0,
                   help="Buffer width in meters for --exclude-near-state (e.g., 10000).")

    # mask choice and debug
    p.add_argument("--force-mask-2021", action="store_true",
                   help="Use CDL 2021 cropland mask for all years (stable mask).")
    p.add_argument("--debug", action="store_true", help="Print small diagnostics (label histogram).")
    return p

def main():
    init_ee()
    args = build_argparser().parse_args()

    for abbr in args.states:
        for y in args.years:
            export_random_cropland_for_state_year(
                abbr=abbr,
                year=y,
                points=args.points,
                seed=args.seed,
                desc_prefix=args.desc_prefix,
                drive_folder=args.drive_folder,
                balance=args.balance,
                pos_frac=args.pos_frac,
                exclude_near_state=args.exclude_near_state,
                buffer_m=args.buffer_m,
                force_mask_2021=args.force_mask_2021,
                debug=args.debug,
            )

if __name__ == "__main__":
    main()


"""
Usage for training data export
poetry run python irr/cli/gee_data_export/gee_export_cropland_points.py \
  --states OR ID MT \
  --years 2018 2019 2020 2021 2022 \
  --points 20000 \
  --balance random \
  --exclude-near-state WA \
  --buffer-m 10000 \
  --debug
"""