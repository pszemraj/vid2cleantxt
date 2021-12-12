"""
    v2ct_utils.py - a bunch of general utilities like loading all files of a certain extension in a directory, etc. FUnctions for the actual processing of the files are in audio2text_functions.py
"""

import os
import pprint as pp
import shutil
from datetime import datetime
from io import StringIO
from os import listdir
from os.path import basename
from os.path import dirname, isfile, join

import GPUtil as GPU
import humanize
import psutil
import torch
import wordninja
from cleantext import clean
from natsort import natsorted


# basics
def get_timestamp():
    return datetime.now().strftime("%b-%d-%Y_t-%H")


def print_spacer(n=1):
    """print_spacer - print a spacer line"""
    print("\n   --------    " * n)


def find_ext_local(
    src_dir, req_ext=".txt", return_type="list", full_path=True, verbose=False
):
    # returns the full path for every file with extension req_ext ONLY in the immediate directory specified
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
            print(
               f"NOTE: there are {len(appr_files)} total matching files\n")

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose:
            print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


def find_ext_recursive(
    src_dir, req_ext=".txt", full_path=True, verbose=False
):
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
                appr_files.append(this_path) if full_path else appr_files.append(prefile)

    appr_files = natsorted(appr_files)  # sort

    if verbose:
        print(f"A list of files in the {src_dir} directory is: \n")
        pp.pprint(appr_files) if len(appr_files) < 10 else pp.pprint(appr_files[:10])
        if len(appr_files) > 10:
            print(
               f"NOTE: there are {len(appr_files)} total matching files\n")

    return appr_files



def shorten_title(title_text, max_no=20):
    if len(title_text) < max_no:
        return title_text
    else:
        return title_text[:max_no] + "..."


def create_folder(new_path):
    os.makedirs(new_path, exist_ok=True)


class NullIO(StringIO):
    # used to redirect system output for things that print a lot to console
    def write(self, txt):
        pass


def move2completed(from_dir, filename, new_folder="completed", verbose=False):
    # this is the better version
    old_filepath = join(from_dir, filename)

    new_filedirectory = join(from_dir, new_folder)

    if not os.path.isdir(new_filedirectory):
        os.mkdir(new_filedirectory)
        if verbose:
            print("created new directory for files at: \n", new_filedirectory)

    new_filepath = join(new_filedirectory, filename)

    try:
        shutil.move(old_filepath, new_filepath)
        print("successfully moved the file {} to */completed.".format(filename))
    except:
        print(
            "ERROR! unable to move file to \n{}. Please investigate".format(
                new_filepath
            )
        )


def cleantxt_wrap(ugly_text):
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
        lang="en",  # set to 'de' for German special handling
    )

    return cleaned_text


def trim_fname(filename, num_words=20, start_reverse=False, word_separator="_"):
    # takes a filename stored as text, removes extension, separates into X words ...
    # and returns a nice filename with the words separateed by
    # useful for when you are reading files, doing things to them, and making new files

    filename = str(filename)
    index_file_Ext = filename.rfind(".")
    current_name = str(filename)[:index_file_Ext]  # get rid of extension
    clean_name = cleantxt_wrap(current_name)  # wrapper with custom defs
    file_words = wordninja.split(clean_name)
    # splits concatenated text into a list of words based on common word freq
    if len(file_words) <= num_words:
        num_words = len(file_words)

    if start_reverse:
        t_file_words = file_words[-num_words:]
    else:
        t_file_words = file_words[:num_words]

    pretty_name = word_separator.join(t_file_words)  # see function argument

    # NOTE IT DOES NOT RETURN THE EXTENSION
    return pretty_name[
        : (len(pretty_name) - 1)
    ]  # there is a always space at the end, so -1


# Hardware


def check_runhardware_torch(verbose=False):
    # https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu/

    GPUs = GPU.getGPUs()

    if len(GPUs) > 0:
        if verbose:
            print("\n ------------------------------")
            print("Checking CUDA status for PyTorch")

        torch.cuda.init()

        print("Cuda availability (PyTorch): ", torch.cuda.is_available())

        # Get Id of default device
        torch.cuda.current_device()
        if verbose:
            print(
                "Name of GPU: ", torch.cuda.get_device_name(device=0)
            )  # '0' is the id of your GPU
            print("------------------------------\n")
        return True

    else:
        print("No GPU being used :(")
        return False


def torch_validate_cuda(verbose=False):
    """
    torch_validate_cuda - checks if CUDA is available and if it is, it checks if the GPU is available.

    """
    GPUs = GPU.getGPUs()
    num_gpus = len(GPUs)
    try:
        torch.cuda.init()
        if not torch.cuda.is_available():
            print(
                "WARNING - CUDA is not being used in processing - expect longer runtime"
            )
            if verbose:
                print("GPU util detects {} GPUs on your system".format(num_gpus))
    except Exception as e:
        print(
            "WARNING - CUDA is not being used in processing - expect longer runtime"
        )
        print(e)


def check_runhardware(verbose=False):
    # ML package agnostic hardware check
    GPUs = GPU.getGPUs()

    if verbose:
        print("\n ------------------------------")
        print("Checking hardware with psutil")
    try:
        gpu = GPUs[0]
    except:
        if verbose:
            print("GPU not available - ", datetime.now())
        gpu = None
    process = psutil.Process(os.getpid())

    CPU_load = psutil.cpu_percent()
    if CPU_load > 0:
        cpu_load_string = "loaded at {} % |".format(CPU_load)
    else:
        # the first time process.cpu_percent() is called it returns 0 which can be confusing
        cpu_load_string = "|"
    print(
        "\nGen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
        " | Proc size: " + humanize.naturalsize(process.memory_info().rss),
        " | {} CPUs ".format(psutil.cpu_count()),
        cpu_load_string,
    )
    if verbose:
        cpu_trend = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        print(
            "CPU load vs. time: 5 mins - {}% | 10 mins - {}% | 15 mins - {}% |".format(
                cpu_trend[0], cpu_trend[1], cpu_trend[2]
            )
        )

    if len(GPUs) > 0 and GPUs is not None:
        print(
            "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB\n".format(
                gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal
            )
        )
    else:
        print("No GPU being used :(", "\n-----------------\n")


def digest_txt_directory(file_dir, identifer="", verbose=False, make_folder=True):
    run_date = datetime.now()
    if len(identifer) < 1:
        identifer = str(shorten_title(trim_fname(dirname(file_dir))))
    files_to_merge = natsorted(
        [f for f in listdir(file_dir) if isfile(join(file_dir, f)) & f.endswith(".txt")]
    )
    outfilename = (
        "[All-Merged-Text]" + identifer + run_date.strftime("_%d%m%Y_%H") + ".txt"
    )

    og_wd = os.getcwd()
    os.chdir(file_dir)

    if make_folder:
        folder_name = "merged_txt_files"
        output_loc = join(file_dir, folder_name)
        create_folder(output_loc)
        outfilename = join(folder_name, outfilename)
        if verbose:
            print("created new folder. new full path is: \n", output_loc)

    count = 0
    with open(outfilename, "w") as outfile:

        for names in files_to_merge:
            with open(names) as infile:
                count += 1
                outfile.write("Start of: " + names + "\n")
                outfile.writelines(infile.readlines())

            outfile.write("\n")

    print("Merged {} text files together in {}".format(count, dirname(file_dir)))
    if verbose:
        print("the merged file is located at: \n", os.getcwd())
    os.chdir(og_wd)
