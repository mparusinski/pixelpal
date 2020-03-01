#!/usr/bin/python
import os
import os.path
import logging
import tempfile
import tarfile
import pprint
import xml
import defusedxml
import cairosvg
import urllib
from shutil import rmtree
from uuid import uuid4
from zipfile import ZipFile

RAW_DATA_DIR = './data/raw/'
DST_DATA_DIR = './data/processed'
TEST_DIRS = ['clearlooks', 'tango']

pplog = logging.getLogger('pixel pal')
pplog.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4, width=41) 

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


def produce_pngs(svg_fp, dst_dir):
    filename = str(uuid4())
    for size in [32, 64]:
        sub_dir = os.path.join(dst_dir, '{}x{}'.format(size, size))
        os.makedirs(sub_dir, exist_ok=True)
        dst_png_fp = os.path.abspath(os.path.join(sub_dir, filename + '.png'))
        pplog.info(pp.pprint(
            "Converting {} to {}".format(svg_fp, dst_png_fp))
        )
        # Skipping problematic files isn't bad; we only want enough files
        try:
            cairosvg.svg2png(
                url=svg_fp,
                write_to=dst_png_fp,
                output_width=size,
                output_height=size
            )
        except defusedxml.common.EntitiesForbidden:
            pplog.info(pp.pprint("Error in conversion"))
            return
        except TypeError:
            pplog.info(pp.pprint("Error in conversion"))
            return
        except xml.etree.ElementTree.ParseError:
            pplog.info(pp.pprint("Error in conversion"))
            return
        except xml.parsers.expat.ExpatError:
            pplog.info(pp.pprint("Error in conversion"))
            return
        except urllib.error.URLError:
            pplog.info(pp.pprint("Error in conversion"))
            return


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

    target_training_dir = os.path.join(DST_DATA_DIR, 'training')
    if os.path.exists(target_training_dir):
        rmtree(target_training_dir)
    for td in training_dirs:
        svgs = find_svgs(extracted_folders[td])
        pplog.info(pp.pprint(svgs))
        for svg_fp in {x for x in svgs}:
            produce_pngs(svg_fp, target_training_dir)

    test_training_dir = os.path.join(DST_DATA_DIR, 'testing')
    if os.path.exists(test_training_dir):
        rmtree(test_training_dir)
    for td in test_dirs:
        svgs = find_svgs(extracted_folders[td])
        pplog.info(pp.pprint(svgs))
        for svg_fp in svgs:
            produce_pngs(svg_fp, test_training_dir)
