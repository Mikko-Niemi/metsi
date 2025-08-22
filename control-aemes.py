import sys
sys.path.append('/content/metsi/')    # Dynamic import in Google Colab

from examples.declarations.export_prepro import csv_and_json
from lukefi.metsi.domain.pre_ops import *
from lukefi.metsi.domain.sim_ops import *
from lukefi.metsi.sim.generators import *


control_structure = {
    "app_configuration": {
        "state_format": "xml",  # options: fdm, vmi12, vmi13, xml, gpkg
        # "state_input_container": "csv",  # Only relevant with fdm state_format. Options: pickle, json
        # "state_output_container": "csv",  # options: pickle, json, csv, null
        # "derived_data_output_container": "pickle",  # options: pickle, json, null
        "formation_strategy": "partial",
        "evaluation_strategy": "depth",
        "run_modes": ["preprocess", "export_prepro", "simulate", "postprocess", "export"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        preproc_filter
        # "supplement_missing_tree_heights",
        # "supplement_missing_tree_ages",
        # "generate_sapling_trees_from_sapling_strata"
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ],
        preproc_filter: [
            {
                "remove trees": "sapling or stems_per_ha == 0",
                "remove stands": "site_type_category == 0",  # not reference_trees
                "remove stands": "site_type_category == None"
            }
        ]
    },
    "simulation_events": [
        {
            "time_points": [0, 5, 10, 15, 20, 25, 30, 35, 40],
            "generators": [
                {sequence: [
                    cross_cut_standing_trees,
                    collect_standing_tree_properties,
                    calculate_npv,
                    calculate_biomass,
                    report_state
                ]}
            ]
        },
        {
            "time_points": [0, 5, 10, 15],
            "generators": [
                {
                    alternatives: [
                        do_nothing,
                        # "thinning_from_below",
                        # "thinning_from_above",
                        first_thinning,
                        even_thinning,
                        {
                            sequence: [
                                clearcutting,
                                planting
                                # operations for renewal after clearcutting go here
                            ]
                        }
                    ]
                },
                {
                    sequence: [
                        cross_cut_felled_trees,
                        collect_felled_tree_properties
                    ]
                }
            ]
        },
        {
            "time_points": [0, 5, 10, 15, 20, 25, 30, 35, 40],
            "generators": [
                {sequence: [report_period]}
            ]
        },
        {
            "time_points": [40],
            "generators": [
                {sequence: [report_collectives]}
            ]
        },
        {
            "time_points": [0, 5, 10, 15, 20, 25, 30, 35, 40],
            "generators": [
                {sequence: [grow_acta]}
                # "grow_motti"
            ]
        }
    ],
    "operation_params": {
        first_thinning: [
            {
                "thinning_factor": 0.97,
                "e": 0.2,
                "dominant_height_lower_bound": 11,
                "dominant_height_upper_bound": 16
            }
        ],
        thinning_from_below: [
            {
                "thinning_factor": 0.97,
                "e": 0.2
            }
        ],
        thinning_from_above: [
            {
                "thinning_factor": 0.98,
                "e": 0.2
            }
        ],
        even_thinning: [
            {
                "thinning_factor": 0.9,
                "e": 0.2
            }
        ],
        calculate_biomass: [
            {"model_set": 1}
        ],
        report_collectives: [
            {
                "identifier": "identifier",
                "npv_1_percent": "net_present_value.value[(net_present_value.interest_rate==1) & "
                "(net_present_value.time_point == 40)]",
                "npv_2_percent": "net_present_value.value[(net_present_value.interest_rate==2) & "
                "(net_present_value.time_point == 40)]",
                "npv_3_percent": "net_present_value.value[(net_present_value.interest_rate==3) & "
                "(net_present_value.time_point == 40)]",
                "npv_4_percent": "net_present_value.value[(net_present_value.interest_rate==4) & "
                "(net_present_value.time_point == 40)]",
                "npv_5_percent": "net_present_value.value[(net_present_value.interest_rate==5) & "
                "(net_present_value.time_point == 40)]",
                "stock_0": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 0)]",
                "stock_5": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 5)]",
                "stock_10": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 10)]",
                "stock_15": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 15)]",
                "stock_20": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 20)]",
                "stock_25": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 25)]",
                "stock_30": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 30)]",
                "stock_35": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 35)]",
                "stock_40": "cross_cutting.volume_per_ha[(cross_cutting.source == 'standing') & "
                "(cross_cutting.time_point == 40)]",
                "harvest_0": "cross_cutting.volume_per_ha[(cross_cutting.source == 'harvested') & "
                "(cross_cutting.time_point == 0)]",
                "harvest_5": "cross_cutting.volume_per_ha[(cross_cutting.source == 'harvested') & "
                "(cross_cutting.time_point == 5)]",
                "harvest_10": "cross_cutting.volume_per_ha[(cross_cutting.source == 'harvested') & "
                "(cross_cutting.time_point == 10)]",
                "harvest_15": "cross_cutting.operation[(cross_cutting.source == 'harvested') & "
                "(cross_cutting.time_point == 15)]",
                "bm_stem_wood_0": "calculate_biomass.stem_wood[calculate_biomass.time_point == 0]",
                "bm_stem_bark_0": "calculate_biomass.stem_bark[calculate_biomass.time_point == 0]",
                "bm_stem_waste_0": "calculate_biomass.stem_waste[calculate_biomass.time_point == 0]",
                "bm_living_branches_0": "calculate_biomass.living_branches[calculate_biomass.time_point == 0]",
                "bm_dead_branches_0": "calculate_biomass.dead_branches[calculate_biomass.time_point == 0]",
                "bm_foliage_0": "calculate_biomass.foliage[calculate_biomass.time_point == 0]",
                "bm_stumps_0": "calculate_biomass.stumps[calculate_biomass.time_point == 0]",
                "bm_roots_0": "calculate_biomass.roots[calculate_biomass.time_point == 0]",
                "bm_stem_wood_5": "calculate_biomass.stem_wood[calculate_biomass.time_point == 5]",
                "bm_stem_bark_5": "calculate_biomass.stem_bark[calculate_biomass.time_point == 5]",
                "bm_stem_waste_5": "calculate_biomass.stem_waste[calculate_biomass.time_point == 5]",
                "bm_living_branches_5": "calculate_biomass.living_branches[calculate_biomass.time_point == 5]",
                "bm_dead_branches_5": "calculate_biomass.dead_branches[calculate_biomass.time_point == 5]",
                "bm_foliage_5": "calculate_biomass.foliage[calculate_biomass.time_point == 5]",
                "bm_stumps_5": "calculate_biomass.stumps[calculate_biomass.time_point == 5]",
                "bm_roots_5": "calculate_biomass.roots[calculate_biomass.time_point == 5]",
                "bm_stem_wood_10": "calculate_biomass.stem_wood[calculate_biomass.time_point == 10]",
                "bm_stem_bark_10": "calculate_biomass.stem_bark[calculate_biomass.time_point == 10]",
                "bm_stem_waste_10": "calculate_biomass.stem_waste[calculate_biomass.time_point == 10]",
                "bm_living_branches_10": "calculate_biomass.living_branches[calculate_biomass.time_point == 10]",
                "bm_dead_branches_10": "calculate_biomass.dead_branches[calculate_biomass.time_point == 10]",
                "bm_foliage_10": "calculate_biomass.foliage[calculate_biomass.time_point == 10]",
                "bm_stumps_10": "calculate_biomass.stumps[calculate_biomass.time_point == 10]",
                "bm_roots_10": "calculate_biomass.roots[calculate_biomass.time_point == 10]",
                "bm_stem_wood_15": "calculate_biomass.stem_wood[calculate_biomass.time_point == 15]",
                "bm_stem_bark_15": "calculate_biomass.stem_bark[calculate_biomass.time_point == 15]",
                "bm_stem_waste_15": "calculate_biomass.stem_waste[calculate_biomass.time_point == 15]",
                "bm_living_branches_15": "calculate_biomass.living_branches[calculate_biomass.time_point == 15]",
                "bm_dead_branches_15": "calculate_biomass.dead_branches[calculate_biomass.time_point == 15]",
                "bm_foliage_15": "calculate_biomass.foliage[calculate_biomass.time_point == 15]",
                "bm_stumps_15": "calculate_biomass.stumps[calculate_biomass.time_point == 15]",
                "bm_roots_15": "calculate_biomass.roots[calculate_biomass.time_point == 15]",
                "bm_stem_wood_20": "calculate_biomass.stem_wood[calculate_biomass.time_point == 20]",
                "bm_stem_bark_20": "calculate_biomass.stem_bark[calculate_biomass.time_point == 20]",
                "bm_stem_waste_20": "calculate_biomass.stem_waste[calculate_biomass.time_point == 20]",
                "bm_living_branches_20": "calculate_biomass.living_branches[calculate_biomass.time_point == 20]",
                "bm_dead_branches_20": "calculate_biomass.dead_branches[calculate_biomass.time_point == 20]",
                "bm_foliage_20": "calculate_biomass.foliage[calculate_biomass.time_point == 20]",
                "bm_stumps_20": "calculate_biomass.stumps[calculate_biomass.time_point == 20]",
                "bm_roots_20": "calculate_biomass.roots[calculate_biomass.time_point == 20]",
                "bm_stem_wood_25": "calculate_biomass.stem_wood[calculate_biomass.time_point == 25]",
                "bm_stem_bark_25": "calculate_biomass.stem_bark[calculate_biomass.time_point == 25]",
                "bm_stem_waste_25": "calculate_biomass.stem_waste[calculate_biomass.time_point == 25]",
                "bm_living_branches_25": "calculate_biomass.living_branches[calculate_biomass.time_point == 25]",
                "bm_dead_branches_25": "calculate_biomass.dead_branches[calculate_biomass.time_point == 25]",
                "bm_foliage_25": "calculate_biomass.foliage[calculate_biomass.time_point == 25]",
                "bm_stumps_25": "calculate_biomass.stumps[calculate_biomass.time_point == 25]",
                "bm_roots_25": "calculate_biomass.roots[calculate_biomass.time_point == 25]",
                "bm_stem_wood_30": "calculate_biomass.stem_wood[calculate_biomass.time_point == 30]",
                "bm_stem_bark_30": "calculate_biomass.stem_bark[calculate_biomass.time_point == 30]",
                "bm_stem_waste_30": "calculate_biomass.stem_waste[calculate_biomass.time_point == 30]",
                "bm_living_branches_30": "calculate_biomass.living_branches[calculate_biomass.time_point == 30]",
                "bm_dead_branches_30": "calculate_biomass.dead_branches[calculate_biomass.time_point == 30]",
                "bm_foliage_30": "calculate_biomass.foliage[calculate_biomass.time_point == 30]",
                "bm_stumps_30": "calculate_biomass.stumps[calculate_biomass.time_point == 30]",
                "bm_roots_30": "calculate_biomass.roots[calculate_biomass.time_point == 30]",
                "bm_stem_wood_35": "calculate_biomass.stem_wood[calculate_biomass.time_point == 35]",
                "bm_stem_bark_35": "calculate_biomass.stem_bark[calculate_biomass.time_point == 35]",
                "bm_stem_waste_35": "calculate_biomass.stem_waste[calculate_biomass.time_point == 35]",
                "bm_living_branches_35": "calculate_biomass.living_branches[calculate_biomass.time_point == 35]",
                "bm_dead_branches_35": "calculate_biomass.dead_branches[calculate_biomass.time_point == 35]",
                "bm_foliage_35": "calculate_biomass.foliage[calculate_biomass.time_point == 35]",
                "bm_stumps_35": "calculate_biomass.stumps[calculate_biomass.time_point == 35]",
                "bm_roots_35": "calculate_biomass.roots[calculate_biomass.time_point == 35]",
                "bm_stem_wood_40": "calculate_biomass.stem_wood[calculate_biomass.time_point == 40]",
                "bm_stem_bark_40": "calculate_biomass.stem_bark[calculate_biomass.time_point == 40]",
                "bm_stem_waste_40": "calculate_biomass.stem_waste[calculate_biomass.time_point == 40]",
                "bm_living_branches_40": "calculate_biomass.living_branches[calculate_biomass.time_point == 40]",
                "bm_dead_branches_40": "calculate_biomass.dead_branches[calculate_biomass.time_point == 40]",
                "bm_foliage_40": "calculate_biomass.foliage[calculate_biomass.time_point == 40]",
                "bm_stumps_40": "calculate_biomass.stumps[calculate_biomass.time_point == 40]",
                "bm_roots_40": "calculate_biomass.roots[calculate_biomass.time_point == 40]"
            }
        ],
        report_period: [
            {"overall_volume": "cross_cutting.volume_per_ha"}
        ],
        calculate_npv: [
            {"interest_rates": [1, 2, 3, 4, 5]}
        ],
        collect_standing_tree_properties: [
            {"properties": ["stems_per_ha", "species", "breast_height_diameter", "height",
                            "breast_height_age", "biological_age", "saw_log_volume_reduction_factor"]}
        ],
        collect_felled_tree_properties: [
            {"properties": ["stems_per_ha", "species", "breast_height_diameter", "height"]}
        ],
        planting: [
            {"tree_count": 10}
        ]
    },
    "operation_file_params": {
        thinning_from_above: {
            "thinning_limits": "data/parameter_files/Thin_Keskinen_Suomi.txt"
        },
        cross_cut_felled_trees: {
            "timber_price_table": "data/parameter_files/timber_price_table.csv"
        },
        cross_cut_standing_trees: {
            "timber_price_table": "data/parameter_files/timber_price_table.csv"
        },
        clearcutting: {
            "clearcutting_limits_ages": "data/parameter_files/renewal_ages_centralFI.txt",
            "clearcutting_limits_diameters": "data/parameter_files/renewal_diameters_centralFI.txt"
        },
        planting: {
            "planting_instructions": "data/parameter_files/planting_instructions.txt"
        },
        calculate_npv: {
            "land_values": "data/parameter_files/land_values_per_site_type_and_interest_rate_TS1200.json",
            "renewal_costs": "data/parameter_files/renewal_operation_pricing.csv"
        }
    },
    "run_constraints": {
        first_thinning: {
            "minimum_time_interval": 50
        },
        even_thinning: {
            "minimum_time_interval": 15
        },
        clearcutting: {
            "minimum_time_interval": 50
        }
    },
    "post_processing": {
        "operation_params": {
            do_nothing: [
                {"param": "value"}
            ]
        },
        "post_processing": [
            do_nothing
        ]
    },
    "export": [
        {
            "format": "J",
            "cvariables": [
                "identifier", "site_type_category", "land_use_category", "soil_peatland_category"
            ],
            "xvariables": [
                "identifier", "area", "npv_1_percent", "npv_2_percent", "npv_3_percent", "npv_4_percent", "npv_5_percent",
                "stock_0", "stock_5", "stock_10", "stock_15","stock_20", "stock_25", "stock_30", "stock_35", "stock_40",
                "harvest_0", "harvest_5", "harvest_10", "harvest_15",
                "bm_stem_wood_0", "bm_stem_bark_0", "bm_stem_waste_0", "bm_living_branches_0",
                "bm_dead_branches_0", "bm_foliage_0", "bm_stumps_0", "bm_roots_0",
                "bm_stem_wood_5", "bm_stem_bark_5", "bm_stem_waste_5", "bm_living_branches_5",
                "bm_dead_branches_5", "bm_foliage_5", "bm_stumps_5", "bm_roots_5",
                "bm_stem_wood_10", "bm_stem_bark_10", "bm_stem_waste_10", "bm_living_branches_10",
                "bm_dead_branches_10", "bm_foliage_10", "bm_stumps_10", "bm_roots_10",
                "bm_stem_wood_15", "bm_stem_bark_15", "bm_stem_waste_15", "bm_living_branches_15",
                "bm_dead_branches_15", "bm_foliage_15", "bm_stumps_15", "bm_roots_15",
                "bm_stem_wood_20", "bm_stem_bark_20", "bm_stem_waste_20", "bm_living_branches_20",
                "bm_dead_branches_20", "bm_foliage_20", "bm_stumps_20", "bm_roots_20",
                "bm_stem_wood_25", "bm_stem_bark_25", "bm_stem_waste_25", "bm_living_branches_25",
                "bm_dead_branches_25", "bm_foliage_25", "bm_stumps_25", "bm_roots_25",
                "bm_stem_wood_30", "bm_stem_bark_30", "bm_stem_waste_30", "bm_living_branches_30",
                "bm_dead_branches_30", "bm_foliage_30", "bm_stumps_30", "bm_roots_30",
                "bm_stem_wood_35", "bm_stem_bark_35", "bm_stem_waste_35", "bm_living_branches_35",
                "bm_dead_branches_35", "bm_foliage_35", "bm_stumps_35", "bm_roots_35",
                "bm_stem_wood_40", "bm_stem_bark_40", "bm_stem_waste_40", "bm_living_branches_40",
                "bm_dead_branches_40", "bm_foliage_40", "bm_stumps_40", "bm_roots_40"
            ]
        },
        {
            "format": "rm_schedules_events_timber",
            "filename": "timber_sums.txt"
        },
        {
            "format": "rm_schedules_events_trees",
            "filename": "trees.txt"
        }
    ]
}

# The preprocessing export format is added as an external module
control_structure['export_prepro'] = csv_and_json

__all__ = ['control_structure']
