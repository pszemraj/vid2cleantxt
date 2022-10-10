"""
    v2ct_utils.py - a bunch of general utilities like loading all files of a certain extension in a directory, etc.

    Note: Functions for the actual processing of the files are in audio2text_functions.py
"""
import logging
import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
import pprint as pp
import shutil
from datetime import datetime
from io import StringIO
from os import listdir
from os.path import basename, dirname, isfile, join

import GPUtil as GPU
import humanize
import psutil
import spacy
import torch
import wordninja
from cleantext import clean
from natsort import natsorted
from spacy.cli import download

# basics
def get_timestamp(exact=False):
    """
    get_timestamp - return a timestamp in the format YYYY-MM-DD_HH-MM-SS (exact=False)
        or YYYY-MM-DD_HH-MM-SS-MS (exact=True)
    exact : bool, optional, by default False,  if True, return a timestamp with seconds
    """
    ts = (
        datetime.now().strftime("%b-%d-%Y_-%H-%M-%S")
        if exact
        else datetime.now().strftime("%b-%d-%Y_-%H")
    )
    return ts


def print_spacer(n=1):
    """print_spacer - print a spacer line"""
    print("\n   --------    " * n)


def find_ext_local(
    src_dir, req_ext=".txt", return_type="list", full_path=True, verbose=False
):
    """
    find_ext_local - return all files that match extension in a list, either either the full filepath or just the filename relative to the ONLY in the immediate directory below src_dir

    Parameters
    ----------
    src_dir : [type]
        [description]
    req_ext : str, optional
        [description], by default ".txt"
    return_type : str, optional
        [description], by default "list"
    full_path : bool, optional
        [description], by default True
    verbose : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if full_path:
        appr_files = [
            join(src_dir, f)
            for f in listdir(src_dir)
            if isfile(join(src_dir, f)) and f.endswith(req_ext)
        ]
    else:
        appr_files = [
            f
            for f in listdir(src_dir)
            if isfile(join(src_dir, f)) and f.endswith(req_ext)
        ]
    appr_files = natsorted(appr_files)  # sort

    if verbose:
        print(f"A list of files in the {src_dir} directory is: \n")
        pp.pprint(appr_files) if len(appr_files) < 10 else pp.pprint(appr_files[:10])
        if len(appr_files) > 10:
            print(f"NOTE: there are {len(appr_files)} total matching files\n")

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose:
            print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


def find_ext_recursive(src_dir, req_ext=".txt", full_path=True, verbose=False):
    """
    load_all_dir_files - return all files that match extension in a list, either either the full filepath or just the filename relative to the src_dir recursively

    returns appr_files - a list of all files in the src_dir and sub-dirs that match the req_extension
    """
    appr_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(src_dir):
        for prefile in f:
            if prefile.endswith(req_ext):
                this_path = os.path.join(r, prefile)
                appr_files.append(this_path) if full_path else appr_files.append(
                    prefile
                )

    appr_files = natsorted(appr_files)  # sort

    if verbose:
        print(f"A list of files in the {src_dir} directory is: \n")
        pp.pprint(appr_files) if len(appr_files) < 10 else pp.pprint(appr_files[:10])
        if len(appr_files) > 10:
            print(f"NOTE: there are {len(appr_files)} total matching files\n")

    return appr_files


def shorten_title(title_text, max_no=20):
    """
    shorten_title - shorten a title to a max length of max_no characters

    Parameters
    ----------
    title_text : str, required, the title to shorten, e.g. "This is a very long title"
    max_no : int, optional, the max length of the title, e.g. 20

    Returns
    -------
    str, the shortened title
    """

    if len(title_text) < max_no:
        return title_text
    else:
        return title_text[:max_no] + "..."


def create_folder(new_path):
    os.makedirs(new_path, exist_ok=True)


def load_spacy_models():
    """downloads spaCy models if not installed on local machine."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        print(
            "INFO: Downloading language model for the spaCy POS tagger\n"
            "(don't worry, this will only happen once)",
        )
        logging.info(f"downloading the spacy model en_core_web_sm due to:\t{e}")
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")


class NullIO(StringIO):
    """NullIO - used to redirect system output for things that print a lot to console"""

    def write(self, txt):
        pass


def move2completed(from_dir, filename, new_folder="completed", verbose=False):
    """
    move2completed - move a file to a new folder

    Parameters
    ----------
    from_dir : str, the directory to move the file from
    filename : str, the filename to move
    new_folder : str, optional,     the directory to move the file to, by default "completed"
    verbose : bool, optional
    """
    old_filepath = join(from_dir, filename)

    new_filedirectory = join(from_dir, new_folder)

    if not os.path.isdir(new_filedirectory):
        os.mkdir(new_filedirectory)
        if verbose:
            print(f"created new directory {new_filedirectory}")

    new_filepath = join(new_filedirectory, filename)

    try:
        shutil.move(old_filepath, new_filepath)
        print(f"moved {old_filepath} to {new_filepath}")
    except Exception as e:
        print(f"ERROR: could not move {old_filepath} to {new_filepath}")
        print(e)


def cleantxt_wrap(
    ugly_text,
    lang="en",
):
    """
    cleantxt_wrap - a wrapper for the clean() function from the cleantext module

    Parameters
    ----------
    ugly_text : str, the text to clean
    lang : str, the language of the text, by default "en", set to 'de' for German special handling

    Returns
    -------
    str, the cleaned text
    """
    # a wrapper for clean text with options different than default. This is used for the audio2text_functions.py

    # https://pypi.org/project/clean-text/
    cleaned_text = clean(
        ugly_text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=True,  # lowercase text
        no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=True,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=True,  # replace all currency symbols with a special token
        no_punct=True,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUM>",
        replace_with_digit="0",
        lang=lang,
    )

    return cleaned_text


def trim_fname(filename, num_words=20, start_rev=False, word_separator="_"):
    """
    trim_fname - trim a filename to a specified number of words

    Parameters
    ----------
    filename : str
    num_words : int, optional, default=20
    start_reverse : bool, optional, default=False
    word_separator : str, optional, default="_", the character to use to join the words

    Returns
    -------
    str, the trimmed filename
    """

    filename = str(filename)
    index_file_Ext = filename.rfind(".")
    current_name = str(filename)[:index_file_Ext]  # get rid of extension
    clean_name = cleantxt_wrap(current_name)  # helper fn to clean up text
    file_words = wordninja.split(clean_name)  # split into words
    num_words = len(file_words) if len(file_words) <= num_words else num_words

    t_file_words = file_words[:num_words] if not start_rev else file_words[-num_words:]
    new_name = word_separator.join(t_file_words)
    return new_name.strip()


# Hardware


def torch_validate_cuda(verbose=False):
    """
    torch_validate_cuda - checks if CUDA is available and if it is, it checks if the GPU is available.

    """
    try:
        GPUs = GPU.getGPUs()
        if GPUs is not None and len(GPUs) > 0:
            torch.cuda.init()
            print(f"Cuda availability (PyTorch) is {torch.cuda.is_available()}\n")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if verbose:
                print(
                    f"Active GPU device: {torch.cuda.get_device_name(device=device)}\n"
                )
        else:
            print("No GPU being used by this machine :(\n")
    except Exception as e:
        print(f"\nERROR upon attempting to validate CUDA: {e}\n")
        print("\nNo GPU being used by this machine :(\n")


def check_runhardware(verbose=False):
    """
    check_runhardware - checks if CUDA is available and if it is, it checks if the GPU is available.

    """
    try:
        GPUs = GPU.getGPUs()
        if GPUs is not None and len(GPUs) > 0:
            gpu = GPUs[0]
        else:
            print("No GPU being used\n")
            GPUs = gpu = None
    except Exception as e:
        print(f"\nERROR: {e}\n")
        print("\nNo GPU being used\n")
        GPUs = gpu = None

    process = psutil.Process(os.getpid())

    CPU_load = psutil.cpu_percent()
    if CPU_load > 0:
        cpu_load_string = f"loaded at {CPU_load} % |\n"
    else:
        # the first time process.cpu_percent() is called it returns 0 which can be confusing
        cpu_load_string = "|\n"
    print(
        "\nGen RAM Free: {ram} | Proc size: {proc} | {n_cpu} CPUs ".format(
            ram=humanize.naturalsize(psutil.virtual_memory().available),
            proc=humanize.naturalsize(process.memory_info().rss),
            n_cpu=psutil.cpu_count(),
        ),
        cpu_load_string,
    )
    if verbose:
        # prints out load on each core vs time
        cpu_trend = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        print(
            "CPU load vs. time: 5 mins - {}% | 10 mins - {}% | 15 mins - {}% |\n".format(
                cpu_trend[0], cpu_trend[1], cpu_trend[2]
            )
        )

    if GPUs is not None:
        # display GPU name, memory, etc
        if len(GPUs) > 0:
            print(
                "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB\n".format(
                    gpu.memoryFree,
                    gpu.memoryUsed,
                    gpu.memoryUtil * 100,
                    gpu.memoryTotal,
                )
            )
    # no need to check for CUDA AGAIN they are sad and will not let you use them


def digest_txt_directory(file_dir, iden: str = None, verbose=False, make_folder=True):
    """
    digest_txt_directory - digest a directory of text files into a single file

    Parameters
    ----------
    file_dir : str, the directory to digest
    iden : str, optional, default=None, the identifier to use for the digest
    verbose : bool, optional
    make_folder : bool, optional

    Returns
    -------
    str, the path to the digest file
    """

    run_date = get_timestamp()
    iden = str(shorten_title(trim_fname(dirname(file_dir)))) if iden is None else iden
    if make_folder:
        os.makedirs(f"{file_dir}/{iden}", exist_ok=True)
    merged_loc = (
        f"{file_dir}/{iden}/mrg_{iden}_{run_date}.txt"
        if make_folder
        else f"{file_dir}/mrg_{iden}_{run_date}.txt"
    )
    files = [
        f for f in listdir(file_dir) if isfile(join(file_dir, f) and f.endswith(".txt"))
    ]

    outfile = open(merged_loc, "w", encoding="utf-8", errors="ignore")
    for file in files:
        with open(f"{file_dir}/{file}", "r", encoding="utf-8", errors="ignore") as f:
            text = f.readlines()
        outfile.write(f"\n\nStart of {file}\n")
        outfile.writelines(text)

    if verbose:
        print(f"{len(files)} files processed")
    return merged_loc  # return the location of the merged file
