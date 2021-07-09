"""

a list of common functions used repeatedly in audio 2 text project so I don't have to copy and paste them all each time,
and update each individually

by Peter Szemraj

"""
import math
import os
import pprint as pp
import re
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import moviepy.editor as mp
import pandas as pd
import pkg_resources
import texthero as hero
import wordninja
import yake
from natsort import natsorted
from spellchecker import SpellChecker
from symspellpy import SymSpell
from tqdm.auto import tqdm


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
    num_numeric_words = len([free_text for free_text in free_text.split() if free_text.isdigit()])
    return num_numeric_words


def shorten_title(title_text, max_no=20):
    if len(title_text) < max_no:
        return title_text
    else:
        return title_text[:max_no] + "..."


def corr(s):
    # adds space after period if there isn't one
    # removes extra spaces
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s))


def create_folder(new_path):
    os.makedirs(new_path, exist_ok=True)


def beautify_filename(filename, num_words=5, start_reverse=True):
    # takes a filename stored as text, removes extension, separates into X words, and returns
    # useful for when you are reading files, doing things to them, and making new files - you want to preserve some
    # semblance of the prior file, but not get infinitely long crap filenames
    filename = str(filename)
    index_file_Ext = filename.rfind('.')
    current_name = str(filename)[:index_file_Ext]  # get rid of extension
    s = pd.Series(current_name)
    s = hero.remove_stopwords(s)
    hero.clean(s)
    clean_name = s.loc[0]
    file_words = wordninja.split(clean_name)
    # splits concatenated text into a list of words based on common word freq
    if len(file_words) <= num_words:
        num_words = len(file_words)

    if start_reverse:
        t_file_words = file_words[-num_words:]
    else:
        t_file_words = file_words[:num_words]

    pretty_name = " ".join(t_file_words)
    # NOTE IT DOES NOT RETURN THE EXTENSION
    return pretty_name[: (len(pretty_name) - 1)]  # there is a space always at the end, so -1


def convert_vidfile(vidfilename, start_time=0, end_time=6969, input_directory="", output_directory="",
                    new_filename=""):
    # takes a video file and creates an audiofile with various parameters
    # NOTE video filename is required
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
        modified_clip.audio.write_audiofile(join(output_directory, converted_filename))

    audio_conv_results = {
        "output_filename": converted_filename,
        "output_folder": output_directory,
        "clip_length": modified_clip.duration
    }

    return audio_conv_results


def convert_vid_for_transcription(vid2beconv, len_chunks, input_directory, output_directory, verbose=False):
    # Oriented specifically for the "wav2vec2" model speech to text transcription
    # takes a video file, turns it into .wav audio chunks of length <input> and stores them in a specific location
    # TODO add try/except clause in case the user already has an audio file the want to transcribe
    my_clip = mp.VideoFileClip(join(input_directory, vid2beconv))
    number_of_chunks = math.ceil(my_clip.duration / len_chunks)  # to get in minutes
    if verbose: print('converting into ' + str(number_of_chunks) + ' audio chunks')
    preamble = beautify_filename(vid2beconv)
    outfilename_storage = []
    if verbose: print('separating audio into chunks starting at ', datetime.now().strftime("_%H.%M.%S"))
    update_incr = math.ceil(number_of_chunks / 10)

    for i in tqdm(range(number_of_chunks), total=number_of_chunks, desc="Converting Video to Audio"):
        start_time = i * len_chunks
        if i == number_of_chunks - 1:
            this_clip = my_clip.subclip(t_start=start_time)
        else:
            this_clip = my_clip.subclip(t_start=start_time, t_end=(start_time + len_chunks))
        this_filename = preamble + '_run_' + str(i) + '.wav'
        outfilename_storage.append(this_filename)
        this_clip.audio.write_audiofile(join(output_directory, this_filename), logger=None)

    print('Finished creating audio chunks at ', datetime.now().strftime("_%H.%M.%S"))
    if verbose: print('Files are located in ', output_directory)

    return outfilename_storage


def spellcorrect_freetext(input_text, dist=2, del_single_char=False, keep_nw=True, save_results=False,
                          verbose=False):
    spell = SpellChecker(distance=dist, case_sensitive=False)

    # because it's a huge issue, extremely specific misunderstandings are (currently) manually replaced here
    input_text.replace("colonel", "kernel")

    # split text into individual words
    wordlist = spell.split_words(text=input_text)
    # get unknown words from PySpellChecker
    misspelled = spell.unknown(wordlist)
    if keep_nw:
        # if user wants to keep words with number values as-is, remove them from the list of "misspelled" items
        upd_misspelled = []
        for unknown_w in misspelled:
            contains_digit = any(map(str.isdigit, unknown_w))  # check if contains digit using map
            if not contains_digit:
                upd_misspelled.append(unknown_w)
        misspelled = upd_misspelled  # override misspelled (because .remove() somehow gave set size change error)

    corrected_list = []
    num_misspelled = 0
    num_corrected = 0
    for phoneme in wordlist:
        if phoneme in misspelled:
            num_misspelled += 1

            if del_single_char is True and (len(phoneme) == 1):
                # input parameter (default false) can remove any misspelled word with len() = 1
                # you would want to do this sometimes because there really aren't too many single-letter words and there
                # are a lot of random artifacts from pdfs, math equations, and things from word splits that pop up
                corrected_list.append('')

            else:
                to_rebuild = spell.correction(phoneme)
                if to_rebuild != phoneme:
                    # if the spell corrector can't find anything, it returns the same word. So you only fix it if
                    # different
                    num_corrected += 1
                corrected_list.append(to_rebuild)
        else:
            # word is spellec correctly
            to_rebuild = phoneme
            corrected_list.append(to_rebuild)

    corrected_doc = " ".join(corrected_list)

    corrected_dict = {
        "autocorr_text": corrected_doc,
        "input_wordcount": len(wordlist),
        "num_misspelled": num_misspelled,
        "num_corrected": num_corrected
    }
    if num_misspelled == 0:
        perc_corr = 100
    else:
        perc_corr = 100 * num_corrected / num_misspelled
    if verbose:
        print("Spell correct finished.")
        print("Total input wordcount was {0:6d} words.".format(corrected_dict.get("input_wordcount")))
        print("Out of {0:6d} misspelled words, ".format(num_misspelled),
              "a total of {0:6d} words were corrected.".format(num_corrected))
        print("Percent corrected is {0:4f} %\n".format(float(perc_corr)))

    if save_results:
        main_path = os.getcwd()
        prefix = max(4, len(wordlist))
        c_out_f_name = "autocorrected text " + wordlist[0:prefix] + ".txt"
        file_out = open(join(main_path, c_out_f_name), 'w', encoding="utf-8", errors='ignore')
        file_out.write(corrected_doc)
        file_out.close()

    return corrected_dict


def spellcorrect_line(line_o_text, search_dist=2, k_n_w=True, r_s_c=False):
    # this function is meant to be used by spellcorrect_file. This function ASSUMES the input is a single line of text
    # i.e. if you put text that should be multiple lines in here, you are only gonna get 1 long line out

    spell = SpellChecker(distance=search_dist, case_sensitive=False)  # create spellchecker instance

    # because it's a huge issue, extremely specific misunderstandings are (currently) manually replaced here
    # I currently have no use for word colonel as none of my use cases have to do with the military or KFC
    line_o_text.replace("colonel", "kernel")
    # If you do, feel free to delete the above

    # split line_o_text into individual words
    wordlist = spell.split_words(text=line_o_text)
    # get unknown words from PySpellChecker
    misspelled = spell.unknown(wordlist)

    if k_n_w:
        # if user wants to keep words with number values as-is, remove them from the list of "misspelled" items
        upd_misspelled = []
        for unknown_w in misspelled:
            contains_digit = any(map(str.isdigit, unknown_w))  # check if contains digit using map
            if not contains_digit:
                upd_misspelled.append(unknown_w)
        misspelled = upd_misspelled  # override misspelled (because .remove() somehow gave set size change error)

    corrected_list = []
    misspelled_word_log = []
    misspelled_word_len_log = []
    misspelled_substitution_log = []
    num_corrected_fp = 0  # first pass
    num_corrected_sp = 0  # second pass
    num_misspelled = 0
    input_len = len(wordlist)
    # iterate through list of words, check if spelled correctly, apply hierarchy of methods if word is spelled incorrect
    for phoneme in wordlist:
        if phoneme in misspelled:
            # the word is misspelled
            num_misspelled += 1

            if r_s_c is True and (len(phoneme) == 1):
                # input parameter (default false) can remove any misspelled word with len() = 1
                # you would want to do this sometimes because there really aren't too many single-letter words and there
                # are a lot of random artifacts from pdfs, math equations, and things from word splits that pop up
                corrected_list.append('')
                misspelled_word_log.append(phoneme)
                misspelled_substitution_log.append('<set to NULL>')
                misspelled_word_len_log.append(len(phoneme))
            else:
                to_rebuild = spell.correction(phoneme)  # get first correction
                # check if the correction is actually different than word that is misspelled
                if to_rebuild == phoneme:
                    # you did not get a different word for the correction the first time. Try splitting into sub-words
                    separated_amalgam = wordninja.split(phoneme)
                    correct_the_split = " ".join(separated_amalgam)
                    out_split_dict = spellcorrect_freetext(correct_the_split, dist=search_dist, keep_nw=k_n_w,
                                                           del_single_char=r_s_c)
                    to_rebuild = out_split_dict.get("autocorr_text")

                    # see if it is different the second time (after splitting and re-running). If not, too bad
                    if to_rebuild != phoneme:
                        # you did successfully change something on the second pass
                        num_corrected_sp += 1
                else:
                    # you did successfully change something first pass
                    num_corrected_fp += 1
                misspelled_word_log.append(phoneme)
                misspelled_word_len_log.append(len(phoneme))
                misspelled_substitution_log.append(to_rebuild)
                corrected_list.append(to_rebuild)
        else:
            # word is 'valid' so just use that
            to_rebuild = phoneme
            corrected_list.append(to_rebuild)

    corrected_doc = " ".join(corrected_list)

    korrekt_line = {
        "original_text": line_o_text + "\n",
        "corrected_text": corrected_doc + "\n",
        "num_words_in": input_len,
        "num_words_misspelled": num_misspelled,
        "num_fp": num_corrected_fp,
        "num_sp": num_corrected_sp,
        "list_mspws": misspelled_word_log,
        "list_mspws_len": misspelled_word_len_log,
        "list_mspws_changes": misspelled_substitution_log,
    }

    return korrekt_line


def spellcorrect_file(filepath, filename, dist=2, remove_single_chars=False, keep_numb_words=True,
                      create_folder=False, save_metrics=False, verbose=False):
    # given a text (has to be text) file, reads the file, autocorrects any words it deems misspelled, saves as new file
    # it can store the new file in a sub-folder it creates as needed
    # distance represents how far it searches for a better spelling. higher dist = higher RT. See PySpellChecker docs

    script_start_time = time.time()
    if verbose: print("Starting to check and correct the file: ", filename)
    spell = SpellChecker(distance=dist, case_sensitive=False)  # create spellchecker instance

    # delete the three lines below if you do not have a custom frequency dict
    db_fname = 'en.json'
    db_folder = r"C:\Users\peter\PycharmProjects\directory_txt_editing\data"
    try:
        spell.word_frequency.load_dictionary(join(db_folder, db_fname))
    except:
        print("\n\nWARNING - unable to load spelling frequency dictionary from {}".format(db_folder))

    if save_metrics:
        # adjust for weird case
        verbose = True

    # ------------------------------------
    # v3 updates so it reads lines
    file = open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore')
    textlines = file.readlines()  # return a list
    file.close()

    if create_folder:
        # create a folder
        output_folder_name = "spell_corrected_v2_docs"
        if not os.path.isdir(join(filepath, output_folder_name)):
            os.mkdir(
                join(filepath, output_folder_name))  # make a place to store outputs if one does not exist
        filepath = join(filepath, output_folder_name)

    corrected_list = []
    misspelled_word_log = []
    misspelled_word_len_log = []
    misspelled_substitution_log = []
    num_corrected_fp = 0  # first pass
    num_corrected_sp = 0  # second pass
    num_misspelled = 0
    input_len = 0
    print("loaded text with {0:6d} lines ".format(len(textlines)))

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line in tqdm(textlines, total=len(textlines),
                     desc="spell correcting - {}".format(shorten_title(filename))):
        if line == "":
            # blank line, skip to next run
            continue

        # correct the line of text using spellcorrect_line() which returns a dictionary
        correct_dict = spellcorrect_line(line_o_text=line, search_dist=dist, k_n_w=keep_numb_words,
                                         r_s_c=remove_single_chars)

        # append / sum / log results from correcting the line

        corrected_list.append(correct_dict.get("corrected_text"))
        misspelled_word_log.extend(correct_dict.get("list_mspws"))
        misspelled_word_len_log.extend(correct_dict.get("list_mspws_len"))
        misspelled_substitution_log.extend(correct_dict.get("list_mspws_changes"))

        num_misspelled += correct_dict.get("num_words_misspelled")
        num_corrected_fp += correct_dict.get("num_fp")
        num_corrected_sp += correct_dict.get("num_sp")
        input_len += correct_dict.get("num_words_in")

    # finished iterating through lines. Now sum total metrics

    corrected_doc = " ".join(corrected_list)
    total_corrected = num_corrected_fp + num_corrected_sp
    total_rem = num_misspelled - total_corrected
    corrected_fname = "Corrected-v3" + beautify_filename(filename, num_words=9, start_reverse=False) + ".txt"

    # compute % corrected:
    if num_misspelled == 0:
        perc_corr = 100
    else:
        perc_corr = 100 * total_corrected / num_misspelled

    # print results depending on user pref
    if verbose:
        print("Finished. Found {0:6d} misspellings in the file. ".format(num_misspelled),
              "A total of {0:6d} were able to be corrected".format(
                  total_corrected))
        print("Therefore {0:5f} Percent Corrected \n".format(perc_corr))
        metric_list_names = ['# of Words (Total)', '# of Words (Misspelled)', '# Corrected 1st Pass',
                             '# Corrected 2nd Pass', '# Corrected Total', '# of Words Not Fixed']
        metric_list_vals = [input_len, num_misspelled, num_corrected_fp, num_corrected_sp, total_corrected,
                            total_rem]
        print("==========Metrics: ==========")
        sc_metric_db = pd.DataFrame(list(zip(metric_list_names, metric_list_vals)),
                                    columns=['Name', 'Value (int)'])
        pp.pp(sc_metric_db)
        print("==========First 5 Corrections: ==========")
        corr_log_dict = {
            'Misspelled Word (Orig.)': misspelled_word_log,
            'Misspelled Word Length (Char)': misspelled_word_len_log,
            'Correction (PySpellChecker)': misspelled_substitution_log
        }
        corr_log_db = pd.DataFrame(corr_log_dict)
        pp.pp(corr_log_db.head())
        if save_metrics:
            metric_db_fname = "Metrics for " + beautify_filename(filename, num_words=9,
                                                                 start_reverse=False) + " spell correction.xlsx"
            sc_metric_db.to_excel(join(filepath, metric_db_fname), index=False)
            corr_log_db_fname = "Corrections used in " + beautify_filename(filename, num_words=9,
                                                                           start_reverse=False) + " sc.xlsx"
            corr_log_db.to_excel(join(filepath, corr_log_db_fname), index=False)
    else:
        print("Finished. Words input: {0:6d} |".format(input_len),
              "Found # incorrect: {0:6d} words |".format(num_misspelled),
              "Final # corrected: {0:6d} |".format(total_corrected))
        print("Therefore {0:5f} Percent Corrected \n".format(perc_corr))

    # proceed to saving
    with open(join(filepath, corrected_fname), 'w', encoding="utf-8", errors='ignore') as file_out:
        file_out.writelines(corrected_doc)

    # report RT
    script_rt_m = (time.time() - script_start_time) / 60
    if verbose: print("RT for this file was {0:5f} minutes".format(script_rt_m))
    print("\nFinished correcting ", filename, " at time: ", datetime.now().strftime("%H:%M:%S"), "\n")

    corr_file_Data = {
        "corrected_text": corrected_doc,
        "corrected_fname": corrected_fname,
        "output_path": filepath,
        "percent_corrected": perc_corr,
        "num_corrected": total_corrected
    }
    return corr_file_Data


def quick_keys(filename, filepath, max_ngrams=3, num_keywords=20, save_db=False):
    # uses YAKE to quickly determine keywords in a text file. Saves Keywords and YAKE score (0 means very important) in
    # an excel file (from a dataframe)
    # yes, the double entendre is intended.
    file = open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore')
    text = file.read()
    file.close()

    language = "en"
    deduplication_threshold = 0.3  # technically a hyperparameter
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngrams, dedupLim=deduplication_threshold,
                                                top=num_keywords, features=None)
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    phrase_db = pd.DataFrame(yake_keywords)
    phrase_db.columns = ['key_phrase', 'YAKE_sore']

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
    phrase_db2.columns = ['key_phrase', 'YAKE Score (Lower = More Important)', 'num_words', 'freq_in_text']

    if save_db:
        # saves individual file if user asks
        yake_fname = beautify_filename(filename=filename, start_reverse=False) + "_top_phrases_YAKE.xlsx"
        phrase_db2.to_excel(join(filepath, yake_fname), index=False)

    # print out top 10 keywords, or if desired num keywords less than 10, all of them
    max_no_disp = 10
    if num_keywords > max_no_disp:
        num_phrases_disp = max_no_disp
    else:
        num_phrases_disp = num_keywords

    print("Top Key Phrases from YAKE, with max n-gram length: ", max_ngrams, "\n")
    pp.pp(phrase_db2.head(n=num_phrases_disp))

    return phrase_db2


def digest_text_fn(direct, iden='', w_folder=False):
    directory_1 = direct
    id = iden
    run_date = datetime.now()
    if id == "":
        id = "document" + run_date.strftime("_%d%m%Y_")  # run date
    os.chdir(directory_1)
    main_path = os.getcwd()

    if w_folder:
        # create a sub-folder
        output_folder_name = "mergedf_" + run_date.strftime("_%d%m%Y_")
        if not os.path.isdir(output_folder_name):
            os.mkdir(output_folder_name)  # make a place to store outputs if one does not exist
        output_path_full = os.path.join(main_path, output_folder_name)
    else:
        # do not create a folder
        print("not creating folder, file will be @:", direct)
        output_path_full = main_path

    # Load Files from the Directory-----------------------------------------------

    files_to_munch_1 = natsorted([f for f in listdir(directory_1) if isfile(join(directory_1, f))])
    total_files_1 = len(files_to_munch_1)
    removed_count_1 = 0
    # remove non-.txt files
    for prefile in files_to_munch_1:
        if prefile.endswith(".txt"):
            continue
        else:
            files_to_munch_1.remove(prefile)
            removed_count_1 += 1

    print("out of {0:3d} file(s) originally in the folder, ".format(total_files_1),
          "{0:3d} non-.txt files were removed".format(removed_count_1))
    print('\n {0:3d} .txt file(s) in folder will be joined.'.format(len(files_to_munch_1)))

    stitched_masterpiece = []
    announcement_top = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    announcement_middle = "\nTHE NEXT FILE BEGINS HERE!!!!!!!!!"
    announcement_bottom = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"

    for i in range(len(files_to_munch_1)):

        # add header thing for a break between files
        stitched_masterpiece.append(announcement_top)
        stitched_masterpiece.append(announcement_middle)
        stitched_masterpiece.append(announcement_bottom)
        stitched_masterpiece.append(beautify_filename(files_to_munch_1[i], start_reverse=False, num_words=10))
        stitched_masterpiece.append("\n")
        stitched_masterpiece.append("\n")

        # open and append file text
        file = open(join(directory_1, files_to_munch_1[i]), 'rt', encoding="utf-8", errors='ignore')
        f1_text = file.readlines()
        file.close()
        for j in range(len(f1_text)):
            this_line = corr(f1_text[j])
            stitched_masterpiece.append(this_line)
        stitched_masterpiece.append("\n")

    # write file with everything appended to it
    out_filename = id + " [all_text_merged]" + ".txt"
    outfile = open(join(output_path_full, out_filename), 'w', encoding="utf-8", errors='ignore')
    outfile.writelines(stitched_masterpiece)
    outfile.close()
    print("\nDone. Files are located here: ")
    pp.pprint(output_path_full)


def symspell_file(filepath, filename, dist=2, keep_numb_words=True, want_folder=True, save_metrics=False,
                  verbose=False):
    # given a text (has to be text) file, reads the file, autocorrects any words it deems misspelled, saves as new file
    # it can store the new file in a sub-folder it creates as needed
    # distance represents how far it searches for a better spelling. higher dist = higher RT.
    # https://github.com/mammothb/symspellpy

    script_start_time = time.time()
    sym_spell = SymSpell(max_dictionary_edit_distance=dist, prefix_length=7)
    print("PySymSpell - Starting to check and correct the file: ", filename)

    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    # if save_metrics:
    #     adjust for weird case
    # verbose = True

    # ------------------------------------
    with open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore') as file:
        textlines = file.readlines()  # return a list

    if want_folder:
        # create a folder
        output_folder_name = "pyymspell_corrections_SD=" + str(dist)
        out_folder: str = join(filepath, output_folder_name)
        create_folder(out_folder)
        filepath = out_folder  # override

    corrected_list = []
    if verbose: print("loaded text with {0:6d} lines ".format(len(textlines)))

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line in textlines:
        if line == "":
            # blank line, skip to next run
            continue

        # correct the line of text using spellcorrect_line() which returns a dictionary
        suggestions = sym_spell.lookup_compound(phrase=line, max_edit_distance=dist,
                                                ignore_non_words=keep_numb_words,
                                                ignore_term_with_digits=keep_numb_words)
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(suggestion.term)

        # append / sum / log results from correcting the line

        corrected_list.append(' '.join(all_sugg_for_line) + "\n")

    # finished iterating through lines. Now sum total metrics

    corrected_doc = "".join(corrected_list)
    corrected_fname = "Corrected_SSP_" + beautify_filename(filename, num_words=9,
                                                           start_reverse=False) + ".txt"

    # proceed to saving
    with open(join(filepath, corrected_fname), 'w', encoding="utf-8", errors='ignore') as file_out:
        file_out.writelines(corrected_doc)

    # report RT
    script_rt_m = (time.time() - script_start_time) / 60
    print("RT for this file was {0:5f} minutes".format(script_rt_m))
    print("Finished correcting w/ symspell", filename, " at time: ", datetime.now().strftime("%H:%M:%S"),
          "\n")

    corr_file_Data = {
        "corrected_ssp_text": corrected_doc,
        "corrected_ssp_fname": corrected_fname,
        "output_path": filepath,
        # "percent_corrected": perc_corr,
        # "num_corrected": total_corrected
    }
    return corr_file_Data
