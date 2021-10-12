"""

a list of common functions used repeatedly in audio 2 text project, so I don't have to copy and paste them all each time,
and update each individually

"""
import math
import pprint as pp
import re
import sys
import time
from datetime import datetime
from os.path import join

import moviepy.editor as mp
import neuspell
import pandas as pd
import pkg_resources
import pysbd
import yake
from symspellpy import SymSpell
from tqdm.auto import tqdm

from v2ct_utils import beautify_filename, create_folder, NullIO


# ------------------------------------------------------------------------


def avg_word(sentence):
    # returns average word length as a float
    words = sentence.split()
    num_words = len(words)
    if num_words == 0:
        num_words = 1
    return sum(len(word) for word in words) / num_words


# returns number of numeric "words" (i.e., digits that are surrounded by spaces)
def num_numeric_chars(free_text):
    # returns number of numeric words in your text "I love 202 memes" --> 202 is one numeric word
    num_numeric_words = len(
        [free_text for free_text in free_text.split() if free_text.isdigit()]
    )
    return num_numeric_words


def corr(s):
    # adds space after period if there isn't one
    # removes extra spaces
    return re.sub(r"\.(?! )", ". ", re.sub(r" +", " ", s))


def validate_output_directories(
    directory,
    t_folder_name="v2clntxt_transcriptions",
    m_folder_name="v2clntxt_transc_metadata",
):
    t_path_full = join(directory, t_folder_name)
    create_folder(t_path_full)

    m_path_full = join(directory, m_folder_name)
    create_folder(m_path_full)

    output_locs = {"t_out": t_path_full, "m_out": m_path_full}
    return output_locs


def create_metadata_df():
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

    md_df = pd.DataFrame(columns=md_colnames)
    return md_df


def convert_vidfile(
    vidfilename,
    start_time=0,
    end_time=6969,
    input_directory="",
    output_directory="",
    new_filename="",
):
    # takes a video file and creates an audiofile with various parameters
    # NOTE video filename is required
    if len(input_directory) < 1:
        my_clip = mp.VideoFileClip(vidfilename)
    else:
        my_clip = mp.VideoFileClip(join(input_directory, vidfilename))

    if end_time == 6969:
        modified_clip = my_clip.subclip(t_start=int(start_time * 60))
    else:
        modified_clip = my_clip.subclip(
            t_start=int(start_time * 60), t_end=int(end_time * 60)
        )

    converted_filename = (
        vidfilename[: (len(vidfilename) - 4)]
        + "-converted_"
        + datetime.now().strftime("day_%d_time_%H-%M-%S_")
        + ".wav"
    )
    # update_filename
    if len(new_filename) > 0:
        converted_filename = new_filename

    if len(output_directory) < 1:
        modified_clip.audio.write_audiofile(converted_filename)
    else:
        modified_clip.audio.write_audiofile(join(output_directory, converted_filename))

    audio_conv_results = {
        "output_filename": converted_filename,
        "output_folder": output_directory,
        "clip_length": modified_clip.duration,
    }

    return audio_conv_results


def convert_vid_for_transcription(
    vid2beconv, len_chunks, input_directory, output_directory, verbose=False
):
    # Oriented specifically for the "wav2vec2" model speech to text transcription
    # takes a video file, turns it into .wav audio chunks of length <input> and stores them in a specific location
    # TODO add function that is run instead of user already has .WAV files or other audio to be converted
    my_clip = mp.VideoFileClip(join(input_directory, vid2beconv))
    number_of_chunks = math.ceil(my_clip.duration / len_chunks)  # to get in minutes
    if verbose:
        print("converting into " + str(number_of_chunks) + " audio chunks")
    preamble = beautify_filename(vid2beconv)
    outfilename_storage = []
    if verbose:
        print(
            "separating audio into chunks starting at ",
            datetime.now().strftime("_%H.%M.%S"),
        )
    update_incr = math.ceil(number_of_chunks / 10)

    for i in tqdm(
        range(number_of_chunks),
        total=number_of_chunks,
        desc="Converting Video to Audio",
    ):
        start_time = i * len_chunks
        if i == number_of_chunks - 1:
            this_clip = my_clip.subclip(t_start=start_time)
        else:
            this_clip = my_clip.subclip(
                t_start=start_time, t_end=(start_time + len_chunks)
            )
        this_filename = preamble + "_run_" + str(i) + ".wav"
        outfilename_storage.append(this_filename)
        this_clip.audio.write_audiofile(
            join(output_directory, this_filename), logger=None
        )

    print("Finished creating audio chunks at ", datetime.now().strftime("_%H.%M.%S"))
    if verbose:
        print("Files are located in ", output_directory)

    return outfilename_storage


# ------------------------------------------------------------------------

# KEYWORD EXTRACTION


def quick_keys(
    filename,
    filepath,
    max_ngrams=3,
    num_keywords=20,
    save_db=False,
    verbose=False,
    txt_lang="en",
    ddup_thresh=0.3,
):
    # uses YAKE to quickly determine keywords in a text file. Saves Keywords and YAKE score (0 means very important) in
    with open(join(filepath, filename), "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    custom_kw_extractor = yake.KeywordExtractor(
        lan=txt_lang,
        n=max_ngrams,
        dedupLim=ddup_thresh,
        top=num_keywords,
        features=None,
    )
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    phrase_db = pd.DataFrame(yake_keywords)
    if len(phrase_db) == 0:
        print("warning - no phrases were able to be extracted... ")
        return None

    if verbose:
        print("YAKE keywords are: \n", yake_keywords)
        print("dataframe structure: \n")
        pp.pprint(phrase_db.head())

    phrase_db.columns = ["key_phrase", "YAKE_score"]

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

    word_len_series = pd.Series(yake_kw_len, name="No. Words in Phrase")
    word_freq_series = pd.Series(yake_kw_freq, name="Phrase Freq. in Text")
    phrase_db2 = pd.concat([phrase_db, word_len_series, word_freq_series], axis=1)
    # add column names and save file as excel because CSVs suck
    phrase_db2.columns = [
        "key_phrase",
        "YAKE Score (Lower = More Important)",
        "num_words",
        "freq_in_text",
    ]
    if save_db:  # saves individual file if user asks
        yake_fname = (
            beautify_filename(filename=filename, start_reverse=False)
            + "_top_phrases_YAKE.xlsx"
        )
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


# ------------------------------------------------------------------------
# Spelling


def symspell_file(
    filepath, filename, dist=2, keep_numb_words=True, want_folder=True, verbose=False
):
    # original spell-checking method pre SBD (before neuspell. Here for reference / if Neuspell is hard to use on the
    # user's machine/ https://github.com/mammothb/symspellpy

    script_start_time = time.time()
    if verbose:
        print("\nPySymSpell - Starting to correct the file: ", filename)
    # ------------------------------------
    sym_spell = init_symspell()
    with open(join(filepath, filename), "r", encoding="utf-8", errors="ignore") as file:
        textlines = file.readlines()  # return a list

    if want_folder:
        # create a folder
        output_folder_name = "auto-corrected"
        filepath = join(filepath, output_folder_name)
        create_folder(filepath)

    if verbose:
        print("loaded text with {} lines ".format(len(textlines)))

    corrected_list = []

    # iterate through list of lines. Pass each line to be corrected.
    # Append / sum results from each line till done
    for line in textlines:
        if line == "":
            continue  # blank line, skip to next run

        # correct the line of text using spellcorrect_line() which returns a dictionary
        suggestions = sym_spell.lookup_compound(
            phrase=line,
            max_edit_distance=dist,
            ignore_non_words=keep_numb_words,
            ignore_term_with_digits=keep_numb_words,
        )
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(
                suggestion.term
            )  # append / sum / log results from correcting the line

        corrected_list.append(" ".join(all_sugg_for_line) + "\n")

    # finished iterating through lines. Now sum total metrics

    corrected_doc = "".join(corrected_list)
    corrected_fname = (
        "[corr_symsp]"
        + beautify_filename(filename, num_words=15, start_reverse=False)
        + ".txt"
    )

    # proceed to saving
    with open(
        join(filepath, corrected_fname), "w", encoding="utf-8", errors="ignore"
    ) as file_out:
        file_out.writelines(corrected_doc)

    if verbose:
        script_rt_m = (time.time() - script_start_time) / 60
        print("RT for this file was {0:5f} minutes".format(script_rt_m))
        print("output folder for this transcription is: \n", filepath)

    print(
        "Done correcting {} -".format(filename),
        datetime.now().strftime("%H:%M:%S"),
        "\n",
    )

    corr_file_Data = {
        "corrected_ssp_text": corrected_doc,
        "corrected_ssp_fname": corrected_fname,
        "output_path": filepath,
    }
    return corr_file_Data


def init_symspell(max_dist=3, pref_len=7):
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
    # https://github.com/mammothb/symspellpy
    if speller is None:
        if verbose:
            print(
                "Warning - symspell object not passed in, creating one. - ",
                datetime.now(),
            )
        sym_spell = init_symspell()
    else:
        sym_spell = speller

    corrected_list = []

    if type(textlines) == str:
        textlines = [textlines]  # put in a list if a string

    if verbose:
        print("\nStarting to correct text with {0:6d} lines ".format(len(textlines)))
        print("the type of textlines var is ", type(textlines))

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line_obj in textlines:
        line = "".join(line_obj)
        if verbose:
            print("line {} in the text is: ".format(textlines.index(line_obj)))
            pp.pprint(line)
        if line == "":
            continue  # blank line, skip to next run

        suggestions = sym_spell.lookup_compound(
            phrase=line,
            max_edit_distance=dist,
            ignore_non_words=keep_numb_words,
            ignore_term_with_digits=keep_numb_words,
        )
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(
                suggestion.term
            )  # append / sum / log results from correcting the line

        corrected_list.append(" ".join(all_sugg_for_line) + "\n")

    corrected_text = "".join(corrected_list)  # join corrected text

    if verbose:
        print("Finished correcting w/ symspell at time: ", datetime.now(), "\n")

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
        print(
            "Warning - neuspell object not passed in, creating one. - ", datetime.now()
        )
        ns_checker = init_neuspell()

    if type(textlines) == str:
        textlines = [textlines]  # put in a list if a string

    corrected_list = []

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line_obj in textlines:
        line = "".join(line_obj)

        if verbose:
            print("line {} in the text is: ".format(textlines.index(line_obj)))
            pp.pprint(line)
        if line == "" or (len(line) <= 5):
            continue  # blank line

        line = line.lower()
        corrected_text = ns_checker.correct_strings([line])
        corrected_text_f = " ".join(corrected_text)

        corrected_list.append(corrected_text_f + "\n")

    corrected_text = " ".join(corrected_list)  # join corrected text

    if verbose:
        print("Finished correcting w/ neuspell at time: ", datetime.now(), "\n")

    return corrected_text


def SBD_freetext(text, verbose=False, lang="en"):
    # use pysbd to segment

    if isinstance(text, list):
        print(
            "Warning, input ~text~ has type {}. Will convert to str".format(type(text))
        )
        text = " ".join(text)

    seg = pysbd.Segmenter(language=lang, clean=True)
    sentences = seg.segment(text)

    if verbose:
        print(
            "input text of {} words was split into ".format(len(text.split(" "))),
            len(sentences),
            "sentences",
        )

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

    with open(join(filepath, filename), "r", encoding="utf-8", errors="ignore") as file:
        textlines = file.readlines()  # return a list

    sc_textlines = neuspell_freetext(textlines, ns_checker=ns_checker, verbose=verbose)

    loc_SC = "neuspell_sc"
    create_folder(join(filepath, loc_SC))

    sc_outname = (
        "NSC_" + beautify_filename(filename, num_words=15, start_reverse=False) + ".txt"
    )

    with open(
        join(filepath, loc_SC, sc_outname), "w", encoding="utf-8", errors="replace"
    ) as file_sc:
        file_sc.writelines(sc_textlines)  # save spell-corrected text

    # TODO update logic in the respective functions instead of using quick_sc_fixes to fix recurring small issues
    quick_sc_fixes = {
        " ' ": "'",
    }
    if isinstance(sc_textlines, list):
        SBD_sc_textlines = []
        for line in sc_textlines:
            if isinstance(line, list):
                # handles weird corner cases
                line = " ".join(line)

            sentenced = SBD_freetext(line, verbose=verbose)
            for key, value in quick_sc_fixes.items():
                sentenced = sentenced.replace(key, value)
            SBD_sc_textlines.append(sentenced)
    else:
        SBD_sc_textlines = SBD_freetext(sc_textlines, verbose=verbose)
        for key, value in quick_sc_fixes.items():
            SBD_sc_textlines = SBD_sc_textlines.replace(key, value)

    # SBD_text = " ".join(SBD_sc_textlines)

    loc_SBD = "NSC + SBD"
    create_folder(join(filepath, loc_SBD))

    SBD_outname = (
        "FIN_" + beautify_filename(filename, num_words=15, start_reverse=False) + ".txt"
    )

    with open(
        join(filepath, loc_SBD, SBD_outname), "w", encoding="utf-8", errors="replace"
    ) as file_sc:
        file_sc.writelines(
            SBD_sc_textlines
        )  # save spell-corrected AND sentence-boundary disambig text

    pipelineout = {
        "origi_tscript_text": " ".join(textlines),
        "spellcorrected_text": " ".join(sc_textlines),
        "final_text": " ".join(SBD_sc_textlines),
        "spell_corrected_dir": join(filepath, loc_SC),
        "sc_filename": sc_outname,
        "SBD_dir": join(filepath, loc_SBD),
        "SBD_filename": SBD_outname,
    }

    return pipelineout
