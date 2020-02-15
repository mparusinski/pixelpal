#!/usr/bin/python
import os
import os.path
import logging
import tempfile
import tarfile
from zipfile import ZipFile

RAW_DATA_DIR = './data/raw/'


pplog = logging.getLogger('pixel pal')
pplog.setLevel(logging.INFO)

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

def is_theme_file(fp):
    return fp.endswith('.tar.gz') or fp.endswith('.zip')


with tempfile.TemporaryDirectory() as tmpdir:
    pplog.info('Created temporary directory {}'.format(tmpdir))
    # Step 1 extract all the folder to a temporary directory
    dirs = os.listdir(RAW_DATA_DIR)
    for dirname in filter(is_theme_file, dirs):
        basename = dirname.split('.')[0]
        src_zip_fp = os.path.join(RAW_DATA_DIR, dirname)
        dst_dir = os.path.join(tmpdir, basename)
        os.makedirs(dst_dir, exist_ok=True)
        extract(src_zip_fp, dst_dir)
