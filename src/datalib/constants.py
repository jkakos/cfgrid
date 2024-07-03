"""
Used in __init__.py to lookup catalogs and datasets to import them
directly using, e.g., 'datalib.MPAJHU()'. These should be in the form
CATALOGS[class_name] = file_name.

"""

CATALOGS = {
    "MPAJHU": "mpajhu",
    "BPSnapshot": "bpsnapshot",
    "BPSnapshotZSpace": "bpsnapshot",
    "BPSnapshotScatter": "bpsnapshot",
    "CuiModel": "bpsnapshot",
    "InverseCuiModel": "bpsnapshot",
    "Empire": "empire",
}
DATASETS = [
    "DataSet",
    "SimSnapshotDataSet",
    "SimSnapshotZSpaceDataSet",
    "ObsLightConeDataSet",
    "DensityDataSet",
]
