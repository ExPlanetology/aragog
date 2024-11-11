import logging
import os
from pathlib import Path

import platformdirs
from osfclient.api import OSF

logger: logging.Logger = logging.getLogger(__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

logger.info(f'FWL data location: {FWL_DATA_DIR}')

#project ID of the lookup data folder in the OSF
project_id = 'phsxf'

basic_list = (
        "1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018",
        )

def download_folder(*, storage, folders: list[str], data_dir: Path):
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
            parts = file.path.split('/')[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Downloading {file.path}...')
            with open(target, 'wb') as f:
                file.write_to(f)
            break


def GetFWLData() -> Path:
    """
    Get path to FWL data directory on the disk
    """
    return Path(FWL_DATA_DIR).absolute()

def DownloadLookupTableData(fname: str=""):
    """
    Download lookup table data

    Inputs :
        - fname (optional) :    folder name, i.e. "1TPa-dK09-elec-free/temperature"
                                if not provided download all the folder list
    """

    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')

    data_dir = GetFWLData() / "interior_lookup_tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    #If no folder specified download all basic list
    if not fname:
        folder_list = basic_list
    elif fname in basic_list:
        folder_list = [fname]
    else:
        raise ValueError(f"Unrecognised folder name: {fname}")

    folders = [folder for folder in folder_list if not (data_dir / folder).exists()]

    if folders:
        logger.info(f"Downloading interior lookup table data to {data_dir}")
        download_folder(storage=storage, folders=folders, data_dir=data_dir)

    return
