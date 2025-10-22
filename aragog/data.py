import logging
import os
from pathlib import Path
from time import sleep
import subprocess as sp

import platformdirs
from osfclient.api import OSF

logger: logging.Logger = logging.getLogger(__name__)

FWL_DATA_DIR = Path(os.environ.get("FWL_DATA", platformdirs.user_data_dir("fwl_data")))

logger.info(f"FWL data location: {FWL_DATA_DIR}")

# project ID of the lookup data folder in the OSF
project_id = "phsxf"

basic_list = (
    "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018",
    "Melting_curves/Wolf_Bower+2018",
    )

full_list = (
    "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018",
    "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_400GPa",
    "Melting_curves/Monteux+600",
    "Melting_curves/Monteux-600",
    "Melting_curves/Wolf_Bower+2018",
    )

def get_zenodo_record(folder: str) -> str | None:
    """
    Get Zenodo record ID for a given folder.

    Inputs :
        - folder : str
            Folder name to get the Zenodo record ID for

    Returns :
        - str | None : Zenodo record ID or None if not found
    """
    zenodo_map = {
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018': '15877374',
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_400GPa': '15877424',
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa': '17413837',
        'Melting_curves/Monteux+600': '15728091',
        'Melting_curves/Monteux-600': '15728138',
        'Melting_curves/Wolf_Bower+2018': '15728072',
    }    return zenodo_map.get(folder, None)

def download_zenodo_folder(folder: str, data_dir: Path):
    """
    Download a specific Zenodo record into specified folder

    Inputs :
        - folder : str
            Folder name to download
        - folder_dir : Path
            local repository where data are saved
    """

    folder_dir = data_dir / folder
    folder_dir.mkdir(parents=True)
    zenodo_id = get_zenodo_record(folder)
    cmd = [
            "zenodo_get", zenodo_id,
            "-o", folder_dir
        ]
    out = os.path.join(GetFWLData(), "zenodo.log")
    logger.debug("    logging to %s"%out)
    with open(out,'w') as hdl:
        sp.run(cmd, check=True, stdout=hdl, stderr=hdl)

def download_OSF_folder(*, storage, folders: list[str], data_dir: Path):
    """
    Download a specific folder in the OSF repository

    Inputs :
        - storage : OSF storage name
        - folders : folder names to download
        - data_dir : local repository where data are saved
    """
    for file in storage.files:
        for folder in folders:
            if not file.path[1:].startswith(folder):
                continue
            parts = file.path.split("/")[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {file.path}...")
            with open(target, "wb") as f:
                file.write_to(f)
            break

def GetFWLData() -> Path:
    """
    Get path to FWL data directory on the disk
    """
    return Path(FWL_DATA_DIR).absolute()

def DownloadLookupTableData(fname: str = ""):
    """
    Download lookup table data

    Inputs :
        - fname (optional) :    folder name, i.e. "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018"
                                if not provided download all the folder list
    """

    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage("osfstorage")

    data_dir = GetFWLData() / "interior_lookup_tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    # If no folder specified download all basic list
    if not fname:
        folder_list = basic_list
    elif fname in full_list:
        folder_list = [fname]
    else:
        raise ValueError(f"Unrecognised folder name: {fname}")

    for folder in folder_list:
        folder_dir = data_dir / folder
        max_tries = 2 # Maximum download attempts, could be a function argument

        if not folder_dir.exists():
            logger.info(f"Downloading interior lookup table data to {data_dir}")
            for i in range(max_tries):
                logger.info(f"Attempt {i + 1} of {max_tries}")
                success = False

                try:
                    download_zenodo_folder(folder = folder, data_dir=data_dir)
                    success = True
                except RuntimeError as e:
                    logger.error(f"Zenodo download failed: {e}")
                    folder_dir.rmdir()

                if not success:
                    try:
                        download_OSF_folder(storage=storage, folders=folder, data_dir=data_dir)
                        success = True
                    except RuntimeError as e:
                        logger.error(f"OSF download failed: {e}")

                if success:
                    break

                if i < max_tries - 1:
                    logger.info("Retrying download...")
                    sleep(5) # Wait 5 seconds before retrying
                else:
                    logger.error("Max retries reached. Download failed.")

    return
