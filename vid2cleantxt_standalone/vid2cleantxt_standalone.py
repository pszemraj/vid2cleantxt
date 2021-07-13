# this script is the same as the others, but has all of the functions, etc. in one .py script in case there are issues
# using the other versions. Primary use case for this would be debugging what is going on, or understanding
# the overall pipeline.

"""
Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
This script is the 'single-file' edition
Peter Szemraj

large model link / doc from host website (huggingface)
https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

sections in this file:
- define model parameters (pretrained model)
- basic user inputs (vid file name / directory)
- convert video to audio chunks of duration X*
- pass all X audio chunks through wav2vec2model, store results in a list
- write all results of the list into a text file, store various runtime metrics
- pass created text file through a spell checker and autocorrect spelling. save as new file
- run basic keyword extraction from (via YAKE) on spell-corrected file, save in the same directory as other results
- cleanup tasks (delete the X .wav files created for audio transcription, etc), report runtime, and exit

* (where X is some duration that does not overload your computer or crash your IDE)
"""

import math
import os
import pprint as pp
import re
import shutil
import sys
import time
from datetime import datetime
from io import StringIO
from os import listdir
from os.path import basename, dirname, isfile, join

import GPUtil as GPU
import humanize
import librosa
import moviepy.editor as mp
import neuspell
import pandas as pd
import pkg_resources
import plotly.express as px
import psutil
import pysbd
import torch
import wordninja
import yake
from cleantext import clean
from natsort import natsorted
from symspellpy import SymSpell
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


# --------------------------------------------------------------------------
# Function Definitions
# --------------------------------------------------------------------------

# General Utilities

def corr(s):
    # adds space after period if there isn't one
    # removes extra spaces
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s))


def shorten_title(title_text, max_no=20):
    if len(title_text) < max_no:
        return title_text
    else:
        return title_text[:max_no] + "..."


class NullIO(StringIO):
    # used to redirect system output for things that print a lot to console
    def write(self, txt):
        pass


def cleantxt_wrap(ugly_text):
    # a wrapper for clean text with options different than default

    # https://pypi.org/project/clean-text/
    cleaned_text = clean(ugly_text,
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
                         lang="en"  # set to 'de' for German special handling
                         )

    return cleaned_text


def beautify_filename(filename, num_words=20, start_reverse=False,
                      word_separator="_"):
    # takes a filename stored as text, removes extension, separates into X words ...
    # and returns a nice filename with the words separateed by
    # useful for when you are reading files, doing things to them, and making new files

    filename = str(filename)
    index_file_Ext = filename.rfind('.')
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
    return pretty_name[: (len(pretty_name) - 1)]  # there is a space always at the end, so -1


def quick_keys(filename, filepath, max_ngrams=3, num_keywords=20, save_db=False,
               verbose=False, txt_lang='en', ddup_thresh=0.3):
    # uses YAKE to quickly determine keywords in a text file. Saves Keywords and YAKE score (0 means very important) in
    with open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore') as file:
        text = file.read()

    custom_kw_extractor = yake.KeywordExtractor(lan=txt_lang, n=max_ngrams, dedupLim=ddup_thresh,
                                                top=num_keywords, features=None)
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    phrase_db = pd.DataFrame(yake_keywords)
    if len(phrase_db) == 0:
        print("warning - no phrases were able to be extracted... ")
        return None

    if verbose:
        print("YAKE keywords are: \n", yake_keywords)
        print("dataframe structure: \n")
        pp.pprint(phrase_db.head())

    phrase_db.columns = ['key_phrase', 'YAKE_score']

    # add a column for how many words the phrases contain
    yake_kw_len = []
    yake_kw_freq = []
    for entry in yake_keywords:
        entry_wordcount = len(str(entry).split(" ")) - 1
        yake_kw_len.append(entry_wordcount)

    for index, row in phrase_db.iterrows():
        search_term = row["key_phrase"]
        entry_freq = text.count(str(search_term))
        yake_kw_freq.append(entry_freq)

    word_len_series = pd.Series(yake_kw_len, name='No. Words in Phrase')
    word_freq_series = pd.Series(yake_kw_freq, name='Phrase Freq. in Text')
    phrase_db2 = pd.concat([phrase_db, word_len_series, word_freq_series], axis=1)
    # add column names and save file as excel because CSVs suck
    phrase_db2.columns = ['key_phrase', 'YAKE Score (Lower = More Important)',
                          'num_words', 'freq_in_text']
    if save_db:
        # saves individual file if user asks
        yake_fname = beautify_filename(filename=filename,
                                       start_reverse=False) + "_top_phrases_YAKE.xlsx"
        phrase_db2.to_excel(join(filepath, yake_fname), index=False)

    # print out top 10 keywords, or if desired num keywords less than 10, all of them
    max_no_disp = 10
    if num_keywords > max_no_disp:
        num_phrases_disp = max_no_disp
    else:
        num_phrases_disp = num_keywords

    if verbose:
        print("Top Key Phrases from YAKE, with max n-gram length: ", max_ngrams, "\n")
        pp.pprint(phrase_db2.head(n=num_phrases_disp))
    else:
        list_o_words = phrase_db2["key_phrase"].to_list()
        print("top 5 phrases are: \n")
        if len(list_o_words) < 5:
            pp.pprint(list_o_words)
        else:
            pp.pprint(list_o_words[:5])

    return phrase_db2


def digest_txt_directory(file_dir, identifer="", verbose=False,
                         make_folder=True):
    run_date = datetime.now()
    if len(identifer) < 1:
        identifer = str(shorten_title(beautify_filename(dirname(file_dir))))
    files_to_merge = natsorted([f for f in listdir(file_dir) if isfile(join(file_dir, f)) & f.endswith('.txt')])
    outfilename = '[All-Merged-Text]' + identifer + run_date.strftime("_%d%m%Y_%H") + ".txt"

    og_wd = os.getcwd()
    os.chdir(file_dir)

    if make_folder:
        folder_name = "merged_txt_files"
        output_loc = join(file_dir, folder_name)
        create_folder(output_loc)
        outfilename = join(folder_name, outfilename)
        if verbose: print("created new folder. new full path is: \n", output_loc)

    count = 0
    with open(outfilename, 'w') as outfile:

        for names in files_to_merge:
            with open(names) as infile:
                count += 1
                outfile.write("Start of: " + names + '\n')
                outfile.writelines(infile.readlines())

            outfile.write("\n")

    print("Merged {} text files together in {}".format(count, dirname(file_dir)))
    if verbose:
        print("the merged file is located at: \n", os.getcwd())
    os.chdir(og_wd)


def create_folder(new_path):
    os.makedirs(new_path, exist_ok=True)


def validate_output_directories(directory):
    t_folder_name = "v2txt_video_transcriptions"
    m_folder_name = "v2txt_transcription_metadata"

    t_path_full = join(directory, t_folder_name)
    create_folder(t_path_full)

    m_path_full = join(directory, m_folder_name)
    create_folder(m_path_full)

    output_locs = {
        "t_out": t_path_full,
        "m_out": m_path_full
    }
    return output_locs


def move2completed(from_dir, filename, new_folder='completed', verbose=False):
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
        print("ERROR! unable to move file to \n{}. Please investigate".format(new_filepath))


def load_imm_dir_files(src_dir, req_ext=".txt", return_type="list", full_path=True,
                       verbose=False):
    # returns the full path for every file with extension req_ext ONLY in the immediate directory specified
    if full_path:
        appr_files = [join(src_dir, f) for f in listdir(src_dir) if isfile(join(src_dir, f)) and f.endswith(req_ext)]
    else:
        appr_files = [f for f in listdir(src_dir) if isfile(join(src_dir, f)) and f.endswith(req_ext)]
    appr_files = natsorted(appr_files)  # sort

    if verbose:
        print("A list of files in the {} directory + sub-dirs is: \n".format(src_dir))
        if len(appr_files) < 10:
            pp.pprint(appr_files)
        else:
            pp.pprint(appr_files[:10])
            print("\n and more. There are a total of {} files".format(len(appr_files)))

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose: print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


def load_all_dir_files(src_dir, req_ext=".txt", return_type="list", verbose=False):
    # returns the full path for every file with extension req_ext in the directory (and its sub-directories)
    appr_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(src_dir):
        for prefile in f:
            if prefile.endswith(req_ext):
                fullpath = join(r, prefile)
                appr_files.append(fullpath)

    appr_files = natsorted(appr_files)

    if verbose:
        print("A list of files in the {} directory + sub-dirs is: \n".format(src_dir))
        if len(appr_files) < 10:
            pp.pprint(appr_files)
        else:
            pp.pprint(appr_files[:10])
            print("\n and more. There are a total of {} files".format(len(appr_files)))

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose: print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


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
            print("Name of GPU: ", torch.cuda.get_device_name(device=0))  # '0' is the id of your GPU
            print("------------------------------\n")
        return True

    else:
        print("No GPU being used :(")
        return False


def torch_validate_cuda(verbose=False):
    GPUs = GPU.getGPUs()
    num_gpus = len(GPUs)
    try:
        torch.cuda.init()
        if not torch.cuda.is_available():
            print("WARNING - CUDA is not being used in processing - expect longer runtime")
            if verbose: print("GPU util detects {} GPUs on your system".format(num_gpus))
    except:
        print("WARNING - unable to start CUDA. If you wanted to use a GPU, exit and check hardware.")


def check_runhardware(verbose=False):
    # ML package agnostic hardware check
    GPUs = GPU.getGPUs()

    if verbose:
        print("\n ------------------------------")
        print("Checking hardware with psutil")
    try:
        gpu = GPUs[0]
    except:
        if verbose: print("GPU not available - ", datetime.now())
        gpu = None
    process = psutil.Process(os.getpid())

    CPU_load = psutil.cpu_percent()
    if CPU_load > 0:
        cpu_load_string = "loaded at {} % |".format(CPU_load)
    else:
        # the first time process.cpu_percent() is called it returns 0 which can be confusing
        cpu_load_string = "|"
    print("\nGen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " | Proc size: " + humanize.naturalsize(process.memory_info().rss),
          " | {} CPUs ".format(psutil.cpu_count()), cpu_load_string)
    if verbose:
        cpu_trend = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        print("CPU load vs. time: 5 mins - {}% | 10 mins - {}% | 15 mins - {}% |".format(cpu_trend[0], cpu_trend[1],
                                                                                         cpu_trend[2]))

    if len(GPUs) > 0 and GPUs is not None:
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB\n".format(gpu.memoryFree,
                                                                                                      gpu.memoryUsed,
                                                                                                      gpu.memoryUtil * 100,
                                                                                                      gpu.memoryTotal))
    else:
        print("No GPU being used :(", "\n-----------------\n")


# Spelling

def symspell_file(filepath, filename, dist=2, keep_numb_words=True, want_folder=True, verbose=False):
    # original spell-checking method pre SBD (before neuspell. Here for reference / if Neuspell is hard to use on the
    # user's machine/ https://github.com/mammothb/symspellpy

    script_start_time = time.time()
    if verbose: print("\nPySymSpell - Starting to correct the file: ", filename)
    # ------------------------------------
    sym_spell = init_symspell()
    with open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore') as file:
        textlines = file.readlines()  # return a list

    if want_folder:
        # create a folder
        output_folder_name = "auto-corrected"
        filepath = join(filepath, output_folder_name)
        create_folder(filepath)

    if verbose: print("loaded text with {} lines ".format(len(textlines)))

    corrected_list = []

    # iterate through list of lines. Pass each line to be corrected.
    # Append / sum results from each line till done
    for line in textlines:
        if line == "": continue  # blank line, skip to next run

        # correct the line of text using spellcorrect_line() which returns a dictionary
        suggestions = sym_spell.lookup_compound(phrase=line, max_edit_distance=dist,
                                                ignore_non_words=keep_numb_words,
                                                ignore_term_with_digits=keep_numb_words)
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(suggestion.term)  # append / sum / log results from correcting the line

        corrected_list.append(' '.join(all_sugg_for_line) + "\n")

    # finished iterating through lines. Now sum total metrics

    corrected_doc = "".join(corrected_list)
    corrected_fname = "[corr_symsp]" + beautify_filename(filename, num_words=15, start_reverse=False) + ".txt"

    # proceed to saving
    with open(join(filepath, corrected_fname), 'w', encoding="utf-8", errors='ignore') as file_out:
        file_out.writelines(corrected_doc)

    if verbose:
        script_rt_m = (time.time() - script_start_time) / 60
        print("RT for this file was {0:5f} minutes".format(script_rt_m))
        print("output folder for this transcription is: \n", filepath)

    print("Done correcting {} -".format(filename), datetime.now().strftime("%H:%M:%S"), "\n")

    corr_file_Data = {
        "corrected_ssp_text": corrected_doc,
        "corrected_ssp_fname": corrected_fname,
        "output_path": filepath,
    }
    return corr_file_Data


def init_symspell(max_dist=3, pref_len=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max_dist, prefix_length=pref_len)

    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell


def symspell_freetext(textlines, dist=3, keep_numb_words=True, verbose=False,
                      speller=None):
    # https://github.com/mammothb/symspellpy
    if speller is None:
        if verbose: print("Warning - symspell object not passed in, creating one. - ", datetime.now())
        sym_spell = init_symspell()
    else:
        sym_spell = speller

    corrected_list = []

    if type(textlines) == str: textlines = [textlines]  # put in a list if a string

    if verbose:
        print("\nStarting to correct text with {0:6d} lines ".format(len(textlines)))
        print("the type of textlines var is ", type(textlines))

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line_obj in textlines:
        line = ''.join(line_obj)
        if verbose:
            print("line {} in the text is: ".format(textlines.index(line_obj)))
            pp.pprint(line)
        if line == "": continue  # blank line, skip to next run

        suggestions = sym_spell.lookup_compound(phrase=line, max_edit_distance=dist, ignore_non_words=keep_numb_words,
                                                ignore_term_with_digits=keep_numb_words)
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(suggestion.term)  # append / sum / log results from correcting the line

        corrected_list.append(' '.join(all_sugg_for_line) + "\n")

    corrected_text = "".join(corrected_list)  # join corrected text

    if verbose: print("Finished correcting w/ symspell at time: ", datetime.now(), "\n")

    return corrected_text


def init_neuspell(verbose=False):
    # TODO check if options for diferent languages with Neuspell
    if verbose:
        checker = neuspell.SclstmbertChecker()
        checker.from_pretrained()
    else:
        sys.stdout = NullIO()  # hide printing to console for initializations
        checker = neuspell.SclstmbertChecker()
        checker.from_pretrained()
        sys.stdout = sys.__stdout__  # return to default of print-to-console

    return checker


def neuspell_freetext(textlines, ns_checker=None, verbose=False):
    if ns_checker is None:
        print("Warning - neuspell object not passed in, creating one. - ", datetime.now())
        ns_checker = init_neuspell()

    if type(textlines) == str: textlines = [textlines]  # put in a list if a string

    corrected_list = []

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line_obj in textlines:
        line = ''.join(line_obj)

        if verbose:
            print("line {} in the text is: ".format(textlines.index(line_obj)))
            pp.pprint(line)
        if line == "" or (len(line) <= 5): continue  # blank line

        line = line.lower()
        corrected_text = ns_checker.correct_strings([line])
        corrected_text_f = " ".join(corrected_text)

        corrected_list.append(corrected_text_f + "\n")

    corrected_text = " ".join(corrected_list)  # join corrected text

    if verbose: print("Finished correcting w/ neuspell at time: ", datetime.now(), "\n")

    return corrected_text


def SBD_freetext(text, verbose=False, lang="en"):
    # use pysbd to segment

    if isinstance(text, list):
        print("Warning, input ~text~ has type {}. Will convert to str".format(type(text)))
        text = " ".join(text)

    seg = pysbd.Segmenter(language=lang, clean=True)
    sentences = seg.segment(text)

    if verbose:
        print("input text of {} words was split into ".format(len(text.split(" "))), len(sentences), "sentences")

    capitalized = []
    for sentence in sentences:
        if sentence and sentence.strip():
            # ensure that the line is not all spaces
            first_letter = sentence[0].upper()
            rest = sentence[1:]
            capitalized.append(first_letter + rest)

    seg_and_capital = ". ".join(capitalized)  # take segments and make them sentences

    return seg_and_capital


def spellcorrect_pipeline(filepath, filename, ns_checker=None, verbose=False):
    # uses two functions (neuspell_freetext, SBD_freetext) in a pipeline

    with open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore') as file:
        textlines = file.readlines()  # return a list

    sc_textlines = neuspell_freetext(textlines, ns_checker=ns_checker, verbose=verbose)

    loc_SC = "neuspell_sc"
    create_folder(join(filepath, loc_SC))

    sc_outname = "NSC_" + beautify_filename(filename, num_words=15, start_reverse=False) + ".txt"

    with open(join(filepath, loc_SC, sc_outname), 'w', encoding="utf-8", errors='replace') as file_sc:
        file_sc.writelines(sc_textlines)  # save spell-corrected text

    if isinstance(sc_textlines, list):
        SBD_sc_textlines = []
        for line in sc_textlines:
            if isinstance(line, list):
                # handles weird corner cases
                line = " ".join(line)

            sentenced = SBD_freetext(line, verbose=verbose)
            SBD_sc_textlines.append(sentenced)
    else:
        SBD_sc_textlines = SBD_freetext(sc_textlines, verbose=verbose)

    # SBD_text = " ".join(SBD_sc_textlines)

    loc_SBD = "NSC + SBD"
    create_folder(join(filepath, loc_SBD))

    SBD_outname = "FIN_" + beautify_filename(filename, num_words=15, start_reverse=False) + ".txt"

    with open(join(filepath, loc_SBD, SBD_outname), 'w', encoding="utf-8", errors='replace') as file_sc:
        file_sc.writelines(SBD_sc_textlines)  # save spell-corrected AND sentence-boundary disambig text

    pipelineout = {
        "origi_tscript_text": " ".join(textlines),
        "spellcorrected_text": " ".join(sc_textlines),
        "final_text": " ".join(SBD_sc_textlines),
        "spell_corrected_dir": join(filepath, loc_SC),
        "sc_filename": sc_outname,
        "SBD_dir": join(filepath, loc_SBD),
        "SBD_filename": SBD_outname
    }

    return pipelineout


# vid2cleantext specific

def create_metadata_df():
    md_colnames = ["orig_file", "num_audio_chunks", "chunk_len_sec", "input_dur_mins", "date_of_transc",
                   "full_text", "num_chars", "word_count"]

    md_df = pd.DataFrame(columns=md_colnames)
    return md_df


def convert_vidfile(vidfilename, start_time=0, end_time=6969,
                    input_directory="", output_directory="", new_filename=""):
    # takes a video file and creates an audiofile with various parameters

    if len(input_directory) < 1:
        my_clip = mp.VideoFileClip(vidfilename)
    else:
        my_clip = mp.VideoFileClip(join(input_directory, vidfilename))

    if end_time == 6969:
        modified_clip = my_clip.subclip(t_start=int(start_time * 60))
    else:
        modified_clip = my_clip.subclip(t_start=int(start_time * 60), t_end=int(end_time * 60))

    converted_filename = vidfilename[: (len(vidfilename) - 4)] + "-converted_" + \
                         datetime.now().strftime("day_%d_time_%H-%M-%S_") + ".wav"
    # update_filename
    if len(new_filename) > 0:
        converted_filename = new_filename

    if len(output_directory) < 1:
        modified_clip.audio.write_audiofile(converted_filename)
    else:
        # removed 'verbose=False,' from argument of function (removed from Dev)
        modified_clip.audio.write_audiofile(join(output_directory, converted_filename), logger=None)

    audio_conv_results = {
        "output_filename": converted_filename,
        "output_folder": output_directory,
        "clip_length": modified_clip.duration
    }

    return audio_conv_results


def convert_vid_for_transcription(vid2beconv, len_chunks, input_directory, output_directory, verbose=False):
    # takes a video file, turns it into .wav audio chunks of length <input> and stores them in a specific location
    # TODO add function that is run instead of user already has .WAV files or other audio to be converted

    my_clip = mp.VideoFileClip(join(input_directory, vid2beconv))
    number_of_chunks = math.ceil(my_clip.duration / len_chunks)  # to get in minutes
    if verbose: print('\nconverting into ' + str(number_of_chunks) + ' audio chunks')
    preamble = beautify_filename(vid2beconv)
    outfilename_storage = []

    for i in tqdm(range(number_of_chunks), desc="Converting Video to Audio Chunks",
                  total=number_of_chunks):

        start_time = i * len_chunks
        if i == number_of_chunks - 1:
            this_clip = my_clip.subclip(t_start=start_time)
        else:
            this_clip = my_clip.subclip(t_start=start_time, t_end=(start_time + len_chunks))
        this_filename = preamble + '_run_' + str(i) + '.wav'
        outfilename_storage.append(this_filename)

        if this_clip.audio is not None:
            # removed 'verbose=False,' from argument of function (removed from Dev)
            this_clip.audio.write_audiofile(join(output_directory, this_filename), logger=None)
        else:
            print("\n WARNING: chunk {} is empty / has no audio".format(i))

    print('Created all audio chunks - ', datetime.now().strftime("_%H.%M.%S"))
    if verbose: print('Files are located in ', output_directory)
    return outfilename_storage


def transcribe_video_wav2vec(transcription_model, directory, vid_clip_name, chunk_length_seconds,
                             verbose=False):
    # this is the same process as used in the single video transcription, now as a function. Note that spell
    # correction and keyword extraction are now done separately in the script  user needs to pass in: the model,
    # the folder the video is in, and the name of the video

    # Split Video into Audio Chunks-----------------------------------------------
    if verbose: print("Starting to transcribe {} @ {}".format(vid_clip_name, datetime.now()))
    # create audio chunk folder
    output_folder_name = "audio_chunks"
    path2audiochunks = join(directory, output_folder_name)
    create_folder(path2audiochunks)
    chunk_directory = convert_vid_for_transcription(vid2beconv=vid_clip_name, input_directory=directory,
                                                    len_chunks=chunk_length_seconds, output_directory=path2audiochunks)
    torch_validate_cuda()
    check_runhardware()
    time_log.append(time.time())
    time_log_desc.append("converted video to audio")
    full_transcription = []
    GPU_update_incr = math.ceil(len(chunk_directory) / 2)

    # Load audio chunks by name, pass into model, append output text-----------------------------------------------

    for audio_chunk in tqdm(chunk_directory, total=len(chunk_directory),
                            desc="Transcribing {} ".format(shorten_title(basename(vid_clip_name)))):

        current_loc = chunk_directory.index(audio_chunk)

        if (current_loc % GPU_update_incr == 0) and (GPU_update_incr != 0):
            # provide update on GPU usage

            check_runhardware()

        # load dat chunk
        audio_input, rate = librosa.load(join(path2audiochunks, audio_chunk), sr=16000)
        # MODEL
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_values = tokenizer(audio_input, return_tensors="pt", padding="longest", truncation=True).input_values.to(
            device)
        transcription_model = transcription_model.to(device)
        logits = transcription_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = str(tokenizer.batch_decode(predicted_ids)[0])
        full_transcription.append(transcription + "\n")
        # empty memory so you don't overload the GPU
        del input_values
        del logits
        del predicted_ids
        torch.cuda.empty_cache()

    if verbose: print("\nFinished audio transcription of " + vid_clip_name + " and now saving metrics.")

    # build metadata log -------------------------------------------------
    md_df = create_metadata_df()  # makes a blank df with column names
    approx_input_len = (len(chunk_directory) * chunk_length_seconds) / 60
    transc_dt = datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S")
    full_text = corr(' '.join(full_transcription))
    md_df.loc[len(md_df), :] = [vid_clip_name, len(chunk_directory), chunk_length_seconds, approx_input_len,
                                transc_dt, full_text, len(full_text), len(full_text.split(' '))]
    md_df.transpose(copy=False)
    # delete audio chunks in folder -------------------------------------------------
    try:
        shutil.rmtree(path2audiochunks)
        if verbose: print("\nDeleted Audio Chunk Folder + Files")
    except:
        print("WARNING - unable to clean up + delete the audio_chunks folder for {}".format(vid_clip_name))

    # compile results -------------------------------------------------
    transcription_results = {
        "audio_transcription": full_transcription,
        "metadata": md_df
    }

    if verbose: print("\nFinished transcription successfully for " + vid_clip_name + " at "
                      + datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S"))

    return transcription_results


def timelog_analytics(time_log, time_log_desc, save_path=None, show_plot=False, save_plot=True):
    # takes the two lists used everywhere in this file and makes a relatively useful report
    if save_path is None: save_path = os.getcwd()
    time_records_db = pd.DataFrame(list(zip(time_log_desc, time_log)), columns=['Event', 'Time (sec)'])
    start_time_block = time_records_db.loc[0, 'Time (sec)']
    time_records_db["time_mins"] = 0
    time_records_db["diff_time_mins"] = 0
    prior_time = 0
    for index, row in time_records_db.iterrows():
        normalized_time = row['Time (sec)'] - start_time_block
        time_records_db.loc[index, 'Time (sec)'] = normalized_time
        time_records_db.loc[index, 'time_mins'] = normalized_time / 60

        if prior_time == 0:
            time_records_db.loc[index, 'diff_time_mins'] = normalized_time / 60
            prior_time = normalized_time / 60
        else:
            # conditional check mostly for safety reasons
            time_records_db.loc[index, 'diff_time_mins'] = (normalized_time / 60) - prior_time
            prior_time = normalized_time / 60

    time_records_db.to_excel(join(save_path, "vid2cleantxt_" + datetime.now().strftime(
        "%d%m%Y") + "transc_time_log.xlsx"))
    print("total runtime was {} minutes".format(round((time_log[-1] - time_log[0]) / 60)))

    total_time = time_records_db["diff_time_mins"].sum()

    def get_time_frac(section_time):
        return section_time / total_time

    time_records_db["duration_frac"] = time_records_db["diff_time_mins"].apply(get_time_frac)
    time_records_db.loc[time_records_db['duration_frac'] < 0.03, 'Event'] = 'Misc.'  # Represent only large events
    figtitle = 'Run Time Viz - transc of {} video files on {}'.format(len(approved_files),
                                                                      datetime.now().strftime("%d.%m.%Y"))
    fig = px.pie(time_records_db, values='diff_time_mins', names='Event', title=figtitle, template='seaborn')
    if show_plot: fig.show()
    if save_plot: fig.to_html(join(out_p_metadata, "transcriptions {} run time viz.html".format(tag_date)),
                              include_plotlyjs=True, default_width=1280, default_height=720)


# Main

if __name__ == "__main__":
    run_start = datetime.now()
    tag_date = "started_" + run_start.strftime("%m/%d/%Y, %H-%M")

    time_log = []
    time_log_desc = []
    time_log.append(time.time())
    time_log_desc.append("start")

    run_default_examples = True

    if run_default_examples:
        # because the .py files are in their own sub-folder, need to back out
        directory = join(os.path.dirname(os.getcwd()), "initial_test_examples")
    else:
        directory = str(input("\n\n INPUT - enter full filepath to the folder containing video files -->"))

    print('\nWill use the following as directory/file: \n', directory, "\n")

    sys.stdout = NullIO()  # hide printing to console for initializations below:
    checker = init_neuspell()
    sym_spell = init_symspell()
    sys.stdout = sys.__stdout__  # return to default of print-to-console

    # load video files in immediate folder
    vid_extensions = [".mp4", ".mov", ".avi"]
    approved_files = []
    for ext in vid_extensions:
        approved_files.extend(load_imm_dir_files(directory, req_ext=ext, full_path=False))

    # load huggingface model
    time_log.append(time.time())
    time_log_desc.append("starting to load model")

    sys.stdout = NullIO()  # hide printing to console for initializations below:
    wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"  # "facebook/wav2vec2-base-960h" # faster+smaller,    # less accurate
    print("\nPreparing to load model: " + wav2vec2_model)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(wav2vec2_model)
    # Need to switch ^ eventually due to:    # Please use `Wav2Vec2Processor` or `Wav2Vec2CTCTokenizer` instead.
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
    # model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
    chunk_length = 30  # (in seconds) if model fails to work or errors out reduce this number. 30 is a good start.
    sys.stdout = sys.__stdout__  # return to default of print-to-console
    print("loaded the following model:", wav2vec2_model, " at ", datetime.now())
    time_log.append(time.time())
    time_log_desc.append("loaded model")

    # load videos, run through the model
    t_script_start_time = time.time()
    time_log.append(t_script_start_time)
    time_log_desc.append("starting transcription process")

    storage_locs = validate_output_directories(directory)  # create and get output folders
    out_p_tscript = storage_locs.get('t_out')
    out_p_metadata = storage_locs.get('m_out')
    # iterate through list of video files, transcribing one at a time --------------------------------------------------
    for filename in tqdm(approved_files, total=len(approved_files), desc="Total Transcription Progress"):
        t_script_start_time = time.time()
        time_log.append(t_script_start_time)
        time_log_desc.append("start transc. - " + filename)
        # transcribe the video file
        t_results = transcribe_video_wav2vec(transcription_model=model, directory=directory,
                                             vid_clip_name=filename, chunk_length_seconds=chunk_length)
        full_transcription = t_results.get('audio_transcription')
        metadata = t_results.get('metadata')

        # label and store this transcription
        vid_preamble = beautify_filename(filename, num_words=15, start_reverse=False)
        # transcription
        transcribed_filename = vid_preamble + '_tscript_' + datetime.now().strftime("_%H.%M.%S") + '.txt'
        transcribed_file = open(join(out_p_tscript, transcribed_filename), 'w', encoding="utf-8", errors='ignore')
        with open(join(out_p_tscript, transcribed_filename), 'w', encoding="utf-8", errors='ignore') as tf:
            tf.writelines(full_transcription)  # save transcription

        metadata_filename = 'metadata for ' + vid_preamble + " transcription.csv"
        metadata.to_csv(join(out_p_metadata, metadata_filename), index=True)

        t_script_end_time = time.time()
        time_log.append(t_script_end_time)
        time_log_desc.append("transc. file #{}".format(approved_files.index(filename)) + filename[:5] + "...")

        this_rt = (t_script_end_time - t_script_start_time) / 60
        print("transcribed {} in {} minutes".format(shorten_title(filename), round(this_rt, 2)))

        move2completed(directory, filename=filename)

        # save runtime database after each run (useful if the next video causes it to crash)
        time_records_db = pd.DataFrame(list(zip(time_log_desc, time_log)), columns=['Event', 'Time (sec)'])
        time_records_db.to_excel(join(out_p_metadata, "mid_loop_runtime_database.xlsx"))

    time_log.append(time.time())
    time_log_desc.append("all transcriptions completed")
    print("\ntranscription process completed.\n")

    # Post Model: Save files, Spell Check, SBD, Keywords

    merge_transc = True
    if merge_transc:
        # merges all outputs of the transcription, making it easier to check and see if it did things correctly
        t_merge = datetime.now().strftime("date_%d_%m_%Y_time_%H")
        digest_txt_directory(out_p_tscript, identifer="original_tscripts" + t_merge)
        digest_txt_directory(out_p_metadata, identifer="_metadata_for_tscript_run" + t_merge, make_folder=False)
    else:
        print("skipping merging generated .txt files")

    time_log.append(time.time())
    time_log_desc.append("merge files")

    # Validate text files to spell-check (in case there was a change)
    total_files = len(load_imm_dir_files(out_p_tscript, req_ext=".txt"))
    approved_txt_files = load_imm_dir_files(out_p_tscript, req_ext=".txt", verbose=True, full_path=False)
    print("from {} file(s) in dir, loading {} .txt files".format(total_files, len(approved_txt_files)))

    # Spellcorrect Pipeline
    transcript_run_qk = pd.DataFrame()  # empty df to hold all the keywords
    max_item = len(approved_txt_files)

    for origi_tscript in tqdm(approved_txt_files, total=len(approved_txt_files),
                              desc="SC_pipeline - transcribed audio"):
        current_loc = approved_txt_files.index(origi_tscript) + 1  # add 1 bc start at 0
        PL_out = spellcorrect_pipeline(out_p_tscript, origi_tscript, ns_checker=checker, verbose=False)
        # get locations of where corrected files were saved
        directory_for_keywords = PL_out.get("spell_corrected_dir")
        filename_for_keywords = PL_out.get("sc_filename")
        # extract keywords from the saved file
        qk_df = quick_keys(filepath=directory_for_keywords, filename=filename_for_keywords,
                           num_keywords=25, max_ngrams=3, save_db=False, verbose=False)

        transcript_run_qk = pd.concat([transcript_run_qk, qk_df], axis=1)

    # save overall transcription file
    keyword_db_name = "YAKE - keywords for all transcripts in run.csv"
    transcript_run_qk.to_csv(join(out_p_tscript, "YAKE - keywords for all transcripts in run.csv"),
                             index=True)
    time_log.append(time.time())
    time_log_desc.append("SC, keywords")

    # noinspection PyUnboundLocalVariable
    print("Transcription files used to extract KW can be found in: \n ", directory_for_keywords)
    print("A file with keyword results is in {} \ntitled {}".format(out_p_tscript, keyword_db_name))

    zip_dir = join(directory, "zipped_outputs")
    create_folder(zip_dir)
    transcript_header = "transcript_file_archive_" + datetime.now().strftime("%d%m%Y")
    metadata_header = "transcript_metadata_archive" + datetime.now().strftime("%d%m%Y")
    shutil.make_archive(join(zip_dir, transcript_header), "zip", out_p_tscript)
    shutil.make_archive(join(zip_dir, metadata_header), "zip", out_p_metadata)

    time_log.append(time.time())
    time_log_desc.append("crt zip archive")

    # exit block
    print("\n\n----------------------------------- Script Complete -------------------------------")
    print("Transcription file + more can be found here: \n", out_p_tscript)
    print("Metadata for each transcription is located: \n", out_p_metadata)
    time_log.append(time.time())
    time_log_desc.append("End")
    timelog_analytics(time_log, time_log_desc, save_path=out_p_metadata)  # save runtime database & plot
