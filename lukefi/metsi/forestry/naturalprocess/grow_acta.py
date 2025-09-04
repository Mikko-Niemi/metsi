from collections import defaultdict
import math
from statistics import median
import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.model import ReferenceTree, TreeSpecies, ForestStand, SiteType, SoilPeatlandCategory
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height
from lukefi.metsi.data.vector_model import ReferenceTrees


def Pukkala_diameter_growth_by_species(
    spe: TreeSpecies,
    initialDiameter: float,
    G_plot: float,
    BAL_Total: float,
    BAL_Spruce: float,
    BAL_S_B: float,
    TS: int,
    sitetype: SiteType,
    soilpeat: SoilPeatlandCategory
) -> float:
    
    # Parameters (Pukkala et al. 2021):                                           
    D_param = { 'Intercept':[-7.1552, -12.7527, -8.6306], 'sqrt_d':[0.4415, 0.1693, 0.5097], 'd':[-0.0685, -0.0301, -0.0829], \
            'ln_G_1':[-0.2027, -0.1875, -0.3864], 'BAL_Total':[-0.1236, -0.0563, 0], 'BAL_Spruce':[0, -0.0870, 0], \
            'BAL_Spruce_Broadleaf':[0, 0, -0.0545], 'ln_TS':[1.1198, 1.9747, 1.3163], 'Peat':[-0.2425, 0, 0], 'd_Pendula_or_Aspen':[0, 0, 0.0253], \
            'Fertility':{'Herb-rich':[0.1438,0.2688,0.2566], 'Mesic':[0,0,0], 'Sub-xeric':[-0.1754,-0.2145,-0.2256], 'Xeric':[-0.5163,-0.6179,-0.3237]} }

    # Re-codings
    if spe == TreeSpecies.PINE:
        spi = 0
    elif spe == TreeSpecies.SPRUCE:
        spi = 1
    else:
        spi = 2

    if sitetype == (SiteType.VERY_RICH_SITE or SiteType.RICH_SITE):
        site = 'Herb-rich'
    elif sitetype == SiteType.DAMP_SITE:
        site = 'Mesic'
    elif sitetype == SiteType.SUB_DRY_SITE:
        site = 'Sub-xeric'
    else:
        site = 'Xeric'

    if soilpeat != SoilPeatlandCategory.MINERAL_SOIL:
        peat = 1
    else:
        peat = 0

    ln_D_increment = (D_param['Intercept'][spi]
             + D_param['sqrt_d'][spi] * math.sqrt(initialDiameter)
             + D_param['d'][spi] * initialDiameter
             + D_param['ln_G_1'][spi] * math.log(G_plot+1)
             + D_param['BAL_Total'][spi] * (BAL_Total/math.sqrt(initialDiameter+1))
             + D_param['BAL_Spruce'][spi] * (BAL_Spruce/math.sqrt(initialDiameter+1)) 
             + D_param['BAL_Spruce_Broadleaf'][spi] * (BAL_S_B/math.sqrt(initialDiameter+1)) 
             + D_param['ln_TS'][spi] * math.log(TS) 
             + D_param['Fertility'][site][spi] 
             + D_param['Peat'][spi] * peat
             + D_param['d_Pendula_or_Aspen'][spi] * 0 * initialDiameter)       # Deciduous trees assumed Betula Pubescens 
    
    return math.exp(ln_D_increment)


def Pukkala_survival_by_species(
    spe: TreeSpecies,
    initialDiameter: float,
    BAL_Total: float,
    BAL_Pine: float,
    BAL_Spruce: float,
    BAL_S_B: float,
    soilpeat: SoilPeatlandCategory
) -> float:
    
    # Parameters (Pukkala et al. 2021):                                           
    S_param = { 'Intercept':[1.41223, 5.01677, 1.60895], 'sqrt_d':[1.8852, 0.36902, 0.71578], 'd':[-0.21317, -0.07504, -0.08236], \
            'BAL_Total':[-0.25637, 0, 0], 'BAL_Pine':[0, 0, -0.04814], 'BAL_Spruce':[0, -0.2319, 0], 'BAL_Spruce_Broadleaf':[0, 0, -0.13481], \
            'Peat':[-0.39878, -0.47361, -0.31789], 'Aspen':[0, 0, 0.56311], 'Birch':[0, 0, 1.40145] }
    
    # Re-codings
    if spe == TreeSpecies.PINE:
        spi = 0
    elif spe == TreeSpecies.SPRUCE:
        spi = 1
    else:
        spi = 2
    
    if soilpeat != SoilPeatlandCategory.MINERAL_SOIL:
        peat = 1
    else:
        peat = 0
    
    f_survival = (S_param['Intercept'][spi]
                  + S_param['sqrt_d'][spi] * math.sqrt(initialDiameter)
                  + S_param['d'][spi] * initialDiameter
                  + S_param['BAL_Total'][spi] * (BAL_Total/math.sqrt(initialDiameter+1))
                  + S_param['BAL_Pine'][spi] * (BAL_Pine/math.sqrt(initialDiameter+1))
                  + S_param['BAL_Spruce'][spi] * (BAL_Spruce/math.sqrt(initialDiameter+1))
                  + S_param['BAL_Spruce_Broadleaf'][spi] * (BAL_S_B/math.sqrt(initialDiameter+1))
                  + S_param['Peat'][spi] * peat
                  + S_param['Birch'][spi] * 1)              # Deciduous trees assumed Betula spp.
    
    return 1 / ( 1 + math.exp(-f_survival) )


def grow_diameter_and_height(
    stand: ForestStand,
    step: int = 5
) -> tuple[list[float], list[float]]:
    if len(stand.reference_trees) == 0:
        return [], []
    
    # Site attributes
    sitetype = stand.site_type_category
    landclass = stand.soil_peatland_category 
    
    # Temperature sum is not inputted in the forest data standard.
    # If global variable 'TemperatureSum' is given, then use it, else TS=1300.
    try:
        TS = TemperatureSum
    except NameError:
        TS = 1300
    
    # Tree attributes
    trees = stand.reference_trees
    ds = [t.breast_height_diameter or 0 for t in trees]
    ds_predicted = ds
    hs = [t.height for t in trees]    
    hs_predicted = hs
    sps = [t.species for t in trees]
    stems = [t.stems_per_ha for t in trees]
    stems_predicted = stems
    gs = [t.stems_per_ha*math.pi*(0.01*0.5*d)**2 for t,d in zip(trees,ds)]
    G = sum(gs)
    D_pines = [ds[tree] for tree,sp in enumerate(sps) if sp==TreeSpecies.PINE]
    G_pines = [gs[tree] for tree,sp in enumerate(sps) if sp==TreeSpecies.PINE]
    D_spruces = [ds[tree] for tree,sp in enumerate(sps) if sp==TreeSpecies.SPRUCE]
    G_spruces = [gs[tree] for tree,sp in enumerate(sps) if sp==TreeSpecies.SPRUCE]
    D_spruces_broads = [ds[tree] for tree,sp in enumerate(sps) if sp != TreeSpecies.PINE]
    G_spruces_broads = [gs[tree] for tree,sp in enumerate(sps) if sp != TreeSpecies.PINE]
    
    for i,t in enumerate(trees):
        D_current = t.breast_height_diameter
        if D_current is None:
            D_current = 0
        BAL = sum([gs[tree] for tree,diam in enumerate(ds) if diam > D_current])
        BAL_pines = sum([G_pines[tree] for tree,diam in enumerate(D_pines) if diam > D_current])
        BAL_spruces = sum([G_spruces[tree] for tree,diam in enumerate(D_spruces) if diam > D_current])
        BAL_spruces_broads = sum([G_spruces_broads[tree] for tree,diam in enumerate(D_spruces_broads) if diam > D_current])
    
        if hs[i] >= 1.3:
            ds_predicted[i] += Pukkala_diameter_growth_by_species(t.species, D_current, G, BAL, BAL_spruces, BAL_spruces_broads, TS, sitetype, landclass)
            height_0 = naslund_height(D_current, t.species)
            height_5 = naslund_height(ds_predicted[i], t.species)
            hs_predicted[i] += (height_5 - height_0)
            stems_predicted[i] = stems[i] * Pukkala_survival_by_species(t.species, D_current, BAL, BAL_pines, BAL_spruces, BAL_spruces_broads, landclass)
        else:
            ds_predicted[i] += Pukkala_diameter_growth_by_species(t.species, 1.0, G, BAL, BAL_spruces, BAL_spruces_broads, TS, sitetype, landclass)
            height_0 = naslund_height(1.0, t.species)
            height_5 = naslund_height(1.0 + ds_predicted[i], t.species)
            hs_predicted[i] += (height_5 - height_0)
            stems_predicted[i] = stems[i] * Pukkala_survival_by_species(t.species, 1.0, BAL, BAL_pines, BAL_spruces, BAL_spruces_broads, landclass)
            
    return ds_predicted, hs_predicted, stems_predicted


def yearly_diameter_growth_by_species(
    spe: TreeSpecies,
    d: float,
    h: float,
    biological_age_aggregate: float,
    d13_aggregate: float,
    height_aggregate: float,
    dominant_height: float,
    basal_area_total: float
) -> float:
    """ Model source: Acta Forestalia Fennica 163 """
    if spe == TreeSpecies.PINE:
        growth_percent = math.exp(5.4625
                                  - 0.6675 * math.log(biological_age_aggregate)
                                  - 0.4758 * math.log(basal_area_total)
                                  + 0.1173 * math.log(d13_aggregate)
                                  - 0.9442 * math.log(dominant_height)
                                  - 0.3631 * math.log(d)
                                  + 0.7762 * math.log(h))
    else:
        growth_percent = math.exp(6.9342
                                  - 0.8808 * math.log(biological_age_aggregate)
                                  - 0.4982 * math.log(basal_area_total)
                                  + 0.4159 * math.log(d13_aggregate)
                                  - 0.3865 * math.log(height_aggregate)
                                  - 0.6267 * math.log(d)
                                  + 0.1287 * math.log(h))
    return growth_percent


def yearly_height_growth_by_species(
    spe: TreeSpecies,
    d: float,
    h: float,
    biological_age_aggregate: float,
    d13_aggregate: float,
    height_aggregate: float,
    basal_area_total: float
) -> float:
    """ Model source: Acta Forestalia Fennica 163 """
    if spe == TreeSpecies.PINE:
        growth_percent = math.exp(5.4636
                                  - 0.9002 * math.log(biological_age_aggregate)
                                  + 0.5475 * math.log(d13_aggregate)
                                  - 1.1339 * math.log(h))
    else:
        growth_percent = (12.7402
                          - 1.1786 * math.log(biological_age_aggregate)
                          - 0.0937 * math.log(basal_area_total)
                          - 0.1434 * math.log(d13_aggregate)
                          - 0.8070 * math.log(height_aggregate)
                          + 0.7563 * math.log(d)
                          - 2.0522 * math.log(h))
    return growth_percent


# def grow_diameter_and_height(
#     trees: list[ReferenceTree],
#     step: int = 5
# ) -> tuple[list[float], list[float]]:
#     """ Diameter and height growth for trees with height > 1.3 meters. Based on Acta Forestalia Fennica 163. """
#     if not trees:
#         return [], []
#     group = defaultdict(list)
#     for i, t in enumerate(trees):
#         group[t.species].append(i)
#     ds = [t.breast_height_diameter or 0 for t in trees]
#     hs = [t.height for t in trees]
#     for s in range(step):
#         bigh = [h for h in hs if h >= 1.3]
#         if bigh:
#             hdom = median(bigh)
#             gs = [t.stems_per_ha * math.pi * (0.01 * 0.5 * d)**2 for t, d in zip(trees, ds)]
#             g = sum(gs)
#             for spe, idx in group.items():
#                 gg = sum(gs[i] for i in idx)
#                 ag = sum((trees[i].biological_age + s) * gs[i] for i in idx) / gg
#                 dg = sum(ds[i] * gs[i] for i in idx) / gg
#                 hg = sum(hs[i] * gs[i] for i in idx) / gg
#                 for i in idx:
#                     if hs[i] >= 1.3:
#                         pd = yearly_diameter_growth_by_species(spe, ds[i], hs[i], ag, dg, hg, hdom, g) / 100
#                         ph = yearly_height_growth_by_species(spe, ds[i], hs[i], ag, dg, hg, g) / 100
#                         ds[i] *= 1 + pd
#                         hs[i] *= 1 + ph
#         for i, h in enumerate(hs):
#             if h < 1.3:
#                 hs[i] += 0.3
#                 if hs[i] >= 1.3 and not ds[i]:
#                     ds[i] = 1.0
#     return ds, hs

yearly_diameter_growth_by_species_vectorized = np.vectorize(yearly_diameter_growth_by_species)
yearly_height_growth_by_species_vectorized = np.vectorize(yearly_height_growth_by_species)

def grow_diameter_and_height_vectorized(trees: ReferenceTrees,
                                        step: int = 5) -> tuple[npt.NDArray[np.float64],
                                                                npt.NDArray[np.float64]]:
    """
    Diameter and height growth for trees with height > 1.3 meters. Based on Acta Forestalia Fennica 163.
    Vector data implementation.
    """
    if trees.size == 0:
        return np.array([]), np.array([])
    ds = trees.breast_height_diameter
    hs = trees.height
    for s in range(step):
        bigh = np.extract(hs >= 1.3, hs)
        if bigh.size > 0:
            hdom = np.median(bigh)
            gs = trees.stems_per_ha * np.pi * (0.01 * 0.5 * ds)**2
            g = np.sum(gs)
            species = np.unique(trees.species)
            for spe in species:
                gg = np.sum(gs, where=trees.species == spe)
                ag = np.sum((trees.biological_age + s) * gs, where=trees.species == spe) / gg
                dg = np.sum(ds * gs, where=trees.species == spe) / gg
                hg = np.sum(hs * gs, where=trees.species == spe) / gg

                pd = np.where(
                    trees.species == spe,
                    yearly_diameter_growth_by_species_vectorized(
                        spe,
                        ds,
                        hs,
                        ag,
                        dg,
                        hg,
                        hdom,
                        g) / 100,
                    0)
                ph = np.where(
                    trees.species == spe,
                    yearly_height_growth_by_species_vectorized(
                        spe,
                        ds,
                        hs,
                        ag,
                        dg,
                        hg,
                        g) / 100,
                    0)
                ds *= 1 + pd
                hs *= 1 + ph
    return ds, hs
