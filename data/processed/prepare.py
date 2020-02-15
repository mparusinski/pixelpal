#!/usr/bin/python
import os
import os.path
import logging
import tempfile
import tarfile
import pprint
from zipfile import ZipFile

RAW_DATA_DIR = './data/raw/'
TEST_DIRS = ['clearlooks', 'tango']

pplog = logging.getLogger('pixel pal')
pplog.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4, width=41, compact=True) 

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logging.info('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            pplog.info('{}{}'.format(subindent, f))


def extract(fp, target_dir):
    pplog.info("Extracting file {}".format(fp))
    abs_fp = os.path.abspath(fp)
    if fp.endswith('.zip'):
        opener, mode = ZipFile, 'r'
    if fp.endswith('.tar.gz'):
        opener, mode = tarfile.open, 'r:*'

    cwd = os.getcwd()
    os.chdir(target_dir)

    try:
        file = opener(abs_fp, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)
    list_files(target_dir)
    return target_dir


def find_svgs(directory):
    svg_files = []
    for root, dirs, files in os.walk(directory):
        svg_files.extend([os.path.join(root, f) for f in files if
            f.endswith('.svg')])
    return svg_files


def is_theme_file(fp):
    return fp.endswith('.tar.gz') or fp.endswith('.zip')


with tempfile.TemporaryDirectory() as tmpdir:
    pplog.info('Created temporary directory {}'.format(tmpdir))
    # Step 1 extract all the folder to a temporary directory
    dirs = os.listdir(RAW_DATA_DIR)
    extracted_folders = {}
    for dirname in filter(is_theme_file, dirs):
        basename = dirname.split('.')[0]
        src_zip_fp = os.path.join(RAW_DATA_DIR, dirname)
        dst_dir = os.path.join(tmpdir, basename)
        os.makedirs(dst_dir, exist_ok=True)
        extracted_folders[basename] = extract(src_zip_fp, dst_dir) 

    # Step 2 find all the 
    test_dirs = TEST_DIRS
    training_dirs = [d for d in extracted_folders.keys() if d not in TEST_DIRS]

    for td in training_dirs:
        svgs = find_svgs(extracted_folders[td])
        pplog.info(pp.pprint(find_svgs(svgs))

    for td in test_dirs:
        svgs = find_svgs(extracted_folders[td])
        pplog.info(pp.pprint(find_svgs(svgs))
