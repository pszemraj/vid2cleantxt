"""
    audio2text_functions.py - functions for vid2cleantxt project, these functions are used to convert audio files to text.
    general helper functions are in v2ct_utils.py
"""
import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import math
import pprint as pp
import re
import sys
from datetime import datetime

import neuspell
import pandas as pd
import pkg_resources
import pysbd
import yake
from natsort import natsorted
from pydub import AudioSegment
from symspellpy import SymSpell
from tqdm.auto import tqdm

from vid2cleantxt.v2ct_utils import NullIO, create_folder, get_timestamp, trim_fname


def get_av_fmts():
    """
    get_audio_video_fmts - returns the audio and video formats supported by the system
    """
    audio_fmt = [".wav", ".mp3", ".m4a", ".flac"]
    video_fmt = [".mp4", ".mov", ".avi", ".mkv", ".ogg", ".webm"]
    supported_fmts = audio_fmt + video_fmt
    return supported_fmts


def setup_out_dirs(
    directory,
    t_folder_name="v2clntxt_transcriptions",
    m_folder_name="v2clntxt_transc_metadata",
):
    """
    setup_out_dirs - creates output directories for audio2text project, if they do not exist
    """

    t_path_full = join(directory, t_folder_name)
    create_folder(t_path_full)

    m_path_full = join(directory, m_folder_name)
    create_folder(m_path_full)

    output_locs = {"t_out": t_path_full, "m_out": m_path_full}
    return output_locs


def check_if_audio(filename):
    """
    check_if_audio - check if a file is an audio file (True/False)

    Parameters
    ----------
    filename : str, the name of the file to be checked
    """
    return (
        filename.endswith(".wav")
        or filename.endswith(".mp3")
        or filename.endswith(".m4a")
    )


def create_metadata_df():
    """
    create_metadata_df - creates an empty dataframe to store metadata
    """
    md_colnames = [
        "orig_file",
        "num_audio_chunks",
        "chunk_len_sec",
        "input_dur_mins",
        "date_of_transc",
        "full_text",
        "num_chars",
        "word_count",
    ]

    return pd.DataFrame(columns=md_colnames)


def prep_transc_pydub(
    _vid2beconv,
    in_dir,
    out_dir,
    len_chunks=15,
    verbose=False,
):
    """
    prep_transc_pydub - prepares audio files for transcription using pydub

    Parameters
    ----------
    _vid2beconv : str, the name of the video file to be converted
    in_dir : str or Path, the path to the video file directory
    out_dir : str or Path, the path to the output audio file directory
    len_chunks : int, optional, by default 15, the length of the audio chunks in seconds
    verbose : bool, optional, by default False

    Returns
    -------
    list, the list of audio filepaths created
    """

    load_path = join(in_dir, _vid2beconv) if in_dir is not None else _vid2beconv
    vid_audio = AudioSegment.from_file(load_path)
    sound = AudioSegment.set_channels(vid_audio, 1)

    create_folder(out_dir)  # create the output directory if it doesn't exist
    dur_seconds = len(sound) / 1000
    n_chunks = math.ceil(dur_seconds / len_chunks)  # to get in minutes, round up
    pbar = tqdm(total=n_chunks, desc="Creating .wav audio clips")
    preamble = trim_fname(_vid2beconv)
    chunk_fnames = []
    slicer = 1000 * len_chunks  # in milliseconds. slicer = length of each chunk
    for i, chunk in enumerate(sound[::slicer]):
        chunk_name = f"{preamble}_clipaudio_{i}.wav"
        with open(join(out_dir, chunk_name), "wb") as f:
            chunk.export(f, format="wav")
        chunk_fnames.append(chunk_name)
        pbar.update(1)
    pbar.close()

    print(f"\ncreated audio chunks - {get_timestamp()}")
    if verbose:
        print(f" files saved to {out_dir}")

    return natsorted(chunk_fnames)


# ------------------------------------------------------------------------

# KEYWORD EXTRACTION


def quick_keys(
    filename,
    filepath,
    max_ngrams: int = 3,
    num_kw: int = 20,
    disp_max: int = 10,
    save_db=False,
    verbose=False,
    txt_lang="en",
    ddup_thresh=0.3,
):
    """
    quick_keys - extracts keywords from a text file using ngrams and a TF-IDF model (Yake). The keywords are returned as a list of strings and then turned into a dataframe.

    Parameters
    ----------
    filename : str, required, the name of the text file to be processed
    filepath : str, required, the path to the text file to be processed
    max_ngrams : int, optional, the maximum number of ngrams to use, by default 3
    num_kw : int, optional, the number of keywords to extract, by default 20
    disp_max : int, optional, the maximum number of keywords to display, by default 10
    save_db : bool, optional, whether to save the keyword extraction results to excel, by default False
    verbose : bool, optiona,
    txt_lang : str, optional, the language of the text file, by default "en"
    ddup_thresh : float, optional, the threshold for duplicate keywords, by default 0.3

    Returns
    -------
    kw_df : pd.DataFrame, the results of the keyword extraction
    """

    with open(join(filepath, filename), "r", encoding="utf-8", errors="ignore") as fi:
        text = fi.read()

    kw_extractor = yake.KeywordExtractor(
        lan=txt_lang,
        n=max_ngrams,
        dedupLim=ddup_thresh,
        top=num_kw,
        features=None,
    )
    kw_result = kw_extractor.extract_keywords(text)
    phrase_db = pd.DataFrame(kw_result)
    if len(phrase_db) == 0:
        print("warning - no phrases were able to be extracted... ")
        return None

    if verbose:
        print(f"YAKE keywords are: {kw_result}\n")
        print("dataframe structure: \n")
        pp.pprint(phrase_db.head())

    phrase_db.columns = ["key_phrase", "YAKE_score"]

    # add a column for how many words the phrases contain
    yake_kw_len = []
    yake_kw_freq = []
    for entry in kw_result:
        entry_wordcount = len(str(entry).split(" ")) - 1
        yake_kw_len.append(entry_wordcount)

    for index, row in phrase_db.iterrows():
        search_term = row["key_phrase"]
        entry_freq = text.count(str(search_term))
        yake_kw_freq.append(entry_freq)

    word_len_series = pd.Series(yake_kw_len, name="word_count")
    word_freq_series = pd.Series(yake_kw_freq, name="phrase_freq")
    kw_report = pd.concat([phrase_db, word_len_series, word_freq_series], axis=1)
    kw_report.columns = [
        "key_phrase",
        "YAKE_score",
        "word_count",
        "phrase_freq",
    ]
    if save_db:  # saves individual file if user asks
        yake_fname = (
            f"{trim_fname(filename=filename, start_rev=False)}_YAKE_keywords.xlsx"
        )
        kw_report.to_excel(join(filepath, yake_fname), index=False)

    num_phrases_disp = min(num_kw, disp_max)  # number of phrases to display

    if verbose:
        print(f"\nTop Key Phrases from YAKE, with max n-gram length {max_ngrams}")
        pp.pprint(kw_report.head(n=num_phrases_disp))
    else:
        kw_list = kw_report["key_phrase"].to_list()
        print(
            f"\nTop {num_phrases_disp} Key Phrases from YAKE, with max n-gram length {max_ngrams}"
        )
        pp.pprint(kw_list[:num_phrases_disp])

    return kw_report


# ------------------------------------------------------------------------
# Spelling
# TODO: move to spelling module


def avg_word(sentence):
    """
    avg_word - calculates the average word length of a sentence
    """
    words = sentence.split()
    num_words = len(words)
    if num_words == 0:
        num_words = 1
    return sum(len(word) for word in words) / num_words


def num_numeric_chars(free_text):
    """
    returns number of numeric "words" (i.e., digits that are surrounded by spaces)
    """
    num_numeric_words = len(
        [free_text for free_text in free_text.split() if free_text.isdigit()]
    )
    return num_numeric_words


def corr(s: str):
    """
    corr - adds space after period if there isn't one. removes extra spaces

    Parameters
    ----------
    s : str, text to be corrected

    Returns
    -------
    str
    """
    return re.sub(r"\.(?! )", ". ", re.sub(r" +", " ", s))


def init_symspell(max_dist=3, pref_len=7):
    """
    init_symspell - initialize the SymSpell object. This is used to correct misspelled words in the text (interchangeable with NeuSpell)

    Parameters
    ----------
    max_dist : int, optional, by default 3, max distance between words to be considered a match
    pref_len : int, optional, by default 7, minimum length of word to be considered a valid word

    Returns
    -------
    symspell : SymSpell object
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=max_dist, prefix_length=pref_len)

    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )
    # term_index is the column of the term and count_index is the column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell


def symspell_freetext(
    textlines, dist=3, keep_numb_words=True, verbose=False, speller=None
):
    """
    symspell_freetext - spell check a text file using SymSpell. This is used to correct misspelled words in the text (interchangeable with NeuSpell)
    https://github.com/mammothb/symspellpy
    Parameters
    ----------

    textlines : list, list of strings, or string, text to be spell checked
    dist : int, optional, by default 3, max distance between words to be considered a match
    keep_numb_words : bool, optional, by default True, whether to keep numbers in the text
    verbose : bool, optional, by default False, whether to print out the results
    speller : SymSpell object, optional, by default None, if None, will initialize a new SymSpell object

    Returns
    -------
    corrected_text : list, list of strings, or string, the corrected text
    """

    sym_spell = speller or init_symspell(
        max_dist=dist
    )  # initialize a new SymSpell object if none is provided
    corrected_list = []
    textlines = list(textlines) if isinstance(textlines, str) else textlines
    if verbose:
        print(f"{len(textlines)} lines to be spell checked")

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for i, line_obj in enumerate(textlines):
        line = "".join(line_obj)
        if verbose:
            print(f"line {i} in the source is: ")
            pp.pprint(line)
        if line == "":
            continue  # blank line, skip to next run

        suggestions = sym_spell.lookup_compound(
            phrase=line,
            max_edit_distance=dist,
            ignore_non_words=keep_numb_words,
            ignore_term_with_digits=keep_numb_words,
        )
        line_suggests = [
            s.term for s in suggestions
        ]  # list of suggested words for the line

        corrected_list.append(
            " ".join(line_suggests) + "\n"
        )  # add newline to end of each line appended

    corrected_text = corr(
        " ".join(corrected_list)
    )  # join list of lines into a single string and correct spaces

    if verbose:
        print(
            f"fineshed spelling correction on {len(textlines)} lines at {datetime.now()}"
        )

    return corrected_text


def init_neuspell(verbose=False):
    """
    init_neuspell - initialize the neuspell object. This is used to correct misspelled words in the text (interchangeable with SymSpell)

    Returns
    -------
    checker : neuspell.SpellChecker object
    """
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
    """
    neuspell_freetext - spell check a text object using Neuspell. This is used to correct misspelled words in the text (interchangeable with SymSpell)

    Parameters
    ----------
    textlines : list, list of strings, or string, text to be spell checked
    ns_checker : neuspell.SpellChecker object, optional, by default None, if None, will initialize a new Neuspell object
    verbose : bool, optional

    Returns
    -------
    corrected_text : list, list of strings, or string, the corrected text
    """
    ns_checker = ns_checker or init_neuspell(
        verbose=verbose
    )  # initialize a new Neuspell object if none is provided
    textlines = list(textlines) if isinstance(textlines, str) else textlines
    corrected_list = []

    for line_obj in textlines:
        line = "".join(line_obj)

        if verbose:
            print("line {} in the text is: ".format(textlines.index(line_obj)))
            pp.pprint(line)
        if line == "" or (len(line) <= 5):
            continue  # blank line

        corrected_text = ns_checker.correct_strings(
            [line.lower()]
        )  # spell check the lowercase line
        corrected_text_f = " ".join(corrected_text)

        corrected_list.append(corrected_text_f + "\n")

    corrected_text = corr(
        " ".join(corrected_list)
    )  # join list of lines into a single string and correct spaces

    if verbose:
        print("Finished correcting w/ neuspell at time: ", datetime.now(), "\n")
    assert isinstance(corrected_text, str), "corrected text is not a string"
    return corrected_text


def SBD_freetext(text, verbose=False, lang="en"):
    """
    SBD_freetext - spell check a text object using pySBD, a python implementation of the Sentence Boundary Detection algorithm

    Parameters
    ----------
    text : list, list of strings, or string, text to be spell checked
    verbose : bool, optional
    lang : str, optional, by default "en", language of the text

    Returns
    -------
    seg_and_capital = list, list of strings, or string, the corrected text
    """

    if isinstance(text, list):
        text = " ".join(text)
        if verbose:
            print("text is a list, converting to string")

    seg = pysbd.Segmenter(language=lang, clean=True)
    sentences = seg.segment(text)

    if verbose:
        print("Finished sentence boundary detection at time: ", datetime.now(), "\n")
        print("Number of sentences: ", len(sentences))

    capitalized = []
    for sentence in sentences:
        if sentence and sentence.strip():
            # ensure that the line is not all spaces
            first_letter = sentence[0].upper()
            rest = sentence[1:]
            capitalized.append(first_letter + rest)

    seg_and_capital = ". ".join(capitalized)  # take segments and make them sentences

    return seg_and_capital


# TODO: add alternatives for non-English
def spellcorrect_pipeline(
    filepath,
    filename: str,
    method: str = "neuspell",
    spell_checker=None,
    linebyline=True,
    verbose=False,
):
    """
    spellcorrect_pipeline - takes a filepath and filename and returns a corrected version of the text. It uses both the PySBD and Neuspell algorithms to correct the text. Note that the Neuspell algorithm is more accurate than the SymSpell algorithm, but it is slower - it is recommended to use the SymSpell algorithm if you are dealing with a large corpus of text or see runtime issues.

    Parameters
    ----------
    filepath : [type], optional,    the filepath to the file to be corrected
    filename : [type], optional,    the filename of the file to be corrected
    method : str, optional, by default "neuspell", the method to use for spell checking. Options are "neuspell" or "symspell"
    spell_checker : [type], optional, by default None, if None, will initialize a new object for the spell checker
    linebyline : bool, optional,    whether to save the corrected text as a list of lines or a single string in the output file, by default True
    verbose : bool, optional,      whether to print out the progress of the spellchecking process, by default False

    Returns
    -------
    pipelineoutput : dict, the corrected text and other data
    """
    accepted_methods = ["neuspell", "symspell"]
    assert method in accepted_methods, "method must be one of {}".format(
        accepted_methods
    )
    with open(join(filepath, filename), "r", encoding="utf-8", errors="ignore") as file:
        textlines = file.readlines()  # return a list

    # lowercase the textlines
    textlines = [line.lower() for line in textlines]
    # step 1: spellcheck using specified method
    if method.lower() == "neuspell":
        corrected_text = neuspell_freetext(
            textlines, ns_checker=spell_checker, verbose=verbose
        )
    else:
        # symspell fallback
        corrected_text = symspell_freetext(
            textlines,
            verbose=verbose,
            speller=spell_checker,
        )
    loc_SC = f"{method}_corrected"

    create_folder(join(filepath, loc_SC))

    sc_outname = f"{trim_fname(filename)}_NSC_results.txt"
    _corr_out = join(filepath, loc_SC, sc_outname)
    with open(_corr_out, "w", encoding="utf-8", errors="replace") as fo:
        fo.writelines(corrected_text)  # save spell-corrected text
    # step 2: sentence boundary detection & misc punctuation removal
    misc_fixes = {
        " ' ": "'",
        " - ": "-",
        " . ": ".",
    }  # dictionary of miscellaneous fixes, mostly for punctuation

    corrected_text = (
        corrected_text if isinstance(corrected_text, list) else [corrected_text]
    )
    punc_txtlines = []
    for line in corrected_text:
        line = (
            " ".join(line) if isinstance(line, list) else line
        )  # check for list of lists/strings

        for key, value in misc_fixes.items():
            line = line.replace(key, value)
        sentenced = SBD_freetext(line, verbose=verbose)
        assert isinstance(
            sentenced, str
        ), f"sentenced, with type {type(sentenced)}  and valye {sentenced} is not a string.. fix it"
        for key, value in misc_fixes.items():
            sentenced = sentenced.replace(key, value)  # fix punctuation
        punc_txtlines.append(sentenced)

    punc_txtlines = [line.strip() for line in punc_txtlines if line.strip()]
    punc_txtlines = (
        punc_txtlines[0] if linebyline and len(punc_txtlines) == 1 else punc_txtlines
    )
    if linebyline and isinstance(punc_txtlines, str):
        # if the corrected text is a single string, convert it to a list of lines
        punc_txtlines = (
            punc_txtlines.split(". ")
            if isinstance(punc_txtlines, str)
            else punc_txtlines
        )
        punc_txtlines = [
            line + ".\n" for line in punc_txtlines
        ]  # add periods to the end of each line
    loc_FIN = "results_SC_pipeline"
    create_folder(join(filepath, loc_FIN))
    final_outname = f"{trim_fname(filename)}_NSC_SBD.txt"
    SBD_out_path = join(filepath, loc_FIN, final_outname)
    with open(SBD_out_path, "w", encoding="utf-8", errors="replace") as fo2:
        fo2.writelines(punc_txtlines)

    pipelineout = {
        "origi_tscript_text": " ".join(textlines),
        "spellcorrected_text": " ".join(corrected_text),
        "final_text": " ".join(punc_txtlines),
        "spell_corrected_dir": join(filepath, loc_SC),
        "sc_filename": sc_outname,
        "SBD_dir": join(filepath, loc_FIN),
        "SBD_filename": final_outname,
    }

    return pipelineout  # return the corrected text and other data
