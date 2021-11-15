"""
Download all experiment models from:

https://drive.google.com/file/d/17d-wM8C7c-XJt1vKFs1-Kw6KgKwO9ZfU/view?usp=sharing

The move the jov_data.zip file into the repository folder and run this module.
Once the code ran successfully, you can remove the folder 'saved_models'. 

"""
import shutil
from DLBio.helpers import load_json, search_in_all_subfolders
from os.path import dirname, splitext, join, isfile, isdir

ARCHIVE_PATH = 'jov_data.zip'
EXTRACT_DIR = 'saved_models'
RGX = r'file_paths.json'
ZFILL = 4


def run():
    shutil.unpack_archive(ARCHIVE_PATH, EXTRACT_DIR)
    files_ = search_in_all_subfolders(RGX, EXTRACT_DIR)
    assert files_

    for file in files_:
        file_paths = load_json(file)
        base_folder = dirname(file)
        _move_files_to_destination(file_paths, base_folder)


def _move_files_to_destination(file_paths, base_folder):
    for ctr, dst in file_paths.items():
        ext = splitext(dst)[-1]
        src = join(base_folder, str(ctr).zfill(ZFILL) + ext)
        assert isfile(src), f'No file found at: {src}'
        assert isdir(dirname(dst)), f'Directory does not exist: {dst}'

        print(f'Moving {src}->{dst}')
        shutil.move(src, dst)


if __name__ == '__main__':
    run()
