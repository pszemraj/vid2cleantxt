"""

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
This script is the 'single-file' edition
Peter Szemraj

currently still in development

large model link / doc from host website (huggingface)
https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

basic article on how to use wav2vec2
https://www.analyticsvidhya.com/blog/2021/02/hugging-face-introduces-the-first-automatic-speech-recognition-model-wav2vec2/

sections in this file:
- define model parameters (pretrained model)
- basic user inputs (vid file name / directory)
- convert video to audio chunks of duration X*
- pass all X audio chunks through wav2vec2model, store results in a list
- write all results of the list into a text file, store various runtime metrics
- pass created textfile through a spell checker and autocorrect spelling. save as new file
- run basic keyword extraction from (via YAKE) on spell-corrected file, save in the same directory as other results
- cleanup tasks (delete the X .wav files created for audio transcription, etc), report runtime, and exit

* (where X is some duration that does not overload your computer or crash your IDE)
"""

import math
import os
import pprint as pp
import shutil
import time
import re
from datetime import datetime
from os import listdir
from os.path import isfile, join

import librosa
import moviepy.editor as mp
import pandas as pd
import pkg_resources
import pysbd
import texthero as hero
import torch
import wordninja
import yake
from natsort import natsorted
from symspellpy import SymSpell
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


# -------------------------------------------------------
# Function Definitions Script
# -------------------------------------------------------


def corr(s):
    # adds space after period if there isn't one
    # removes extra spaces
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', s))

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


def convert_vidfile(vidfilename, start_time=0, end_time=6969, input_directory="", output_directory="", new_filename=""):
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


def convert_vid_for_transcription(vid2beconv, len_chunks, input_directory, output_directory):
    # Oriented specifically for the "wav2vec2" model speech to text transcription
    # takes a video file, turns it into .wav audio chunks of length <input> and stores them in a specific location
    # TODO add try/except clause in case the user already has an audio file the want to transcribe
    my_clip = mp.VideoFileClip(join(input_directory, vid2beconv))
    number_of_chunks = math.ceil(my_clip.duration / len_chunks)  # to get in minutes
    print('converting into ' + str(number_of_chunks) + ' audio chunks')
    preamble = beautify_filename(vid2beconv)
    outfilename_storage = []
    print('separating audio into chunks starting at ', datetime.now().strftime("_%H.%M.%S"))
    update_incr = math.ceil(number_of_chunks / 10)

    for i in range(number_of_chunks):
        if i % update_incr == 0 and i > 0:
            print('Video conversion to Audio - chunks {0:5f} % done'.format(100 * i / number_of_chunks))
        start_time = i * len_chunks
        if i == number_of_chunks - 1:
            this_clip = my_clip.subclip(t_start=start_time)
        else:
            this_clip = my_clip.subclip(t_start=start_time, t_end=(start_time + len_chunks))
        this_filename = preamble + '_run_' + str(i) + '.wav'
        outfilename_storage.append(this_filename)
        this_clip.audio.write_audiofile(join(output_directory, this_filename), logger=None)

    print('Finished creating audio chunks at ', datetime.now().strftime("_%H.%M.%S"))
    print('Files are located in ', output_directory)
    return outfilename_storage


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


def symspell_file(filepath, filename, dist=2, keep_numb_words=True, create_folder=True, save_metrics=False,
                  print_results=False):
    # given a text (has to be text) file, reads the file, autocorrects any words it deems misspelled, saves as new file
    # it can store the new file in a sub-folder it creates as needed
    # distance represents how far it searches for a better spelling. higher dist = higher RT.
    # https://github.com/mammothb/symspellpy

    script_start_time = time.time()
    sym_spell = SymSpell(max_dictionary_edit_distance=dist, prefix_length=7)
    print("PySymSpell - Starting to check and correct the file: ", filename)

    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    # if save_metrics:
    #     adjust for weird case
    # print_results = True

    # ------------------------------------
    file = open(join(filepath, filename), 'r', encoding="utf-8", errors='ignore')
    textlines = file.readlines()  # return a list
    file.close()

    if create_folder:
        # create a folder
        output_folder_name = "pyymspell_corrections_SD=" + str(dist)
        if not os.path.isdir(join(filepath, output_folder_name)):
            os.mkdir(join(filepath, output_folder_name))  # make a place to store outputs if one does not exist
        filepath = join(filepath, output_folder_name)

    corrected_list = []
    print("loaded text with {0:6d} lines ".format(len(textlines)))

    # iterate through list of lines. Pass each line to be corrected. Append / sum results from each line till done
    for line in textlines:
        if line == "":
            # blank line, skip to next run
            continue

        # correct the line of text using spellcorrect_line() which returns a dictionary
        suggestions = sym_spell.lookup_compound(phrase=line, max_edit_distance=dist, ignore_non_words=keep_numb_words,
                                                ignore_term_with_digits=keep_numb_words)
        all_sugg_for_line = []
        for suggestion in suggestions:
            all_sugg_for_line.append(suggestion.term)

        # append / sum / log results from correcting the line

        corrected_list.append(' '.join(all_sugg_for_line) + "\n")

    # finished iterating through lines. Now sum total metrics

    corrected_doc = "".join(corrected_list)
    corrected_fname = "Corrected_SSP_" + beautify_filename(filename, num_words=9, start_reverse=False) + ".txt"

    # proceed to saving
    file_out = open(join(filepath, corrected_fname), 'w', encoding="utf-8", errors='ignore')
    file_out.writelines(corrected_doc)
    file_out.close()

    # report RT
    script_rt_m = (time.time() - script_start_time) / 60
    print("RT for this file was {0:5f} minutes".format(script_rt_m))
    print("Finished correcting w/ symspell", filename, " at time: ", datetime.now().strftime("%H:%M:%S"), "\n")

    corr_file_Data = {
        "corrected_ssp_text": corrected_doc,
        "corrected_ssp_fname": corrected_fname,
        "output_path": filepath,
        # "percent_corrected": perc_corr,
        # "num_corrected": total_corrected
    }
    return corr_file_Data

def transcribe_video_wav2vec(transcription_model, directory, vid_clip_name, chunk_length_seconds):
    # this is the same process as used in the single video transcription, now as a function. Note that spell correction
    # and keyword extraction are now done separately in the script
    # user needs to pass in: the model, the folder the video is in, and the name of the video
    output_path_full = directory
    # Split Video into Audio Chunks-----------------------------------------------

    print("\n============================================================")
    print("Converting video to audio for file: ", vid_clip_name)
    print("============================================================\n")

    # create audio chunk folder
    output_folder_name = "audio_chunks"
    if not os.path.isdir(join(directory, output_folder_name)):
        os.mkdir(join(directory, output_folder_name))  # make a place to store outputs if one does not exist
    path2audiochunks = join(directory, output_folder_name)
    chunk_directory = convert_vid_for_transcription(vid2beconv=vid_clip_name, input_directory=directory,
                                                    len_chunks=chunk_length_seconds, output_directory=path2audiochunks)

    print("\n============================================================")
    print("converted video to audio. About to start transcription loop for file: ", vid_clip_name)
    print("============================================================\n")

    time_log.append(time.time())
    time_log_desc.append("converted video to audio")
    full_transcription = []
    header = "Transcription of " + vid_clip_name + " at: " + \
             datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S")
    full_transcription.append(header + "\n")
    before_loop_st = time.time()
    update_incr = math.ceil(len(chunk_directory) / 10)
    # Load audio chunks by name, pass into model, append output text-----------------------------------------------
    for audio_chunk in chunk_directory:
        current_loc = chunk_directory.index(audio_chunk)
        if (current_loc % update_incr == 0) and (current_loc != 0):
            # provide update so you know the model is still doing shit
            print('\nStarting run {0:3d} out of '.format(current_loc), '{0:3d}'.format(len(chunk_directory)))
            print('Current time for this run is ', datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S"))
            current_runtime = (time.time() - before_loop_st) / 60
            runs_per_sec = current_loc / current_runtime
            ETA = (len(chunk_directory) - 1 - current_loc) / runs_per_sec
            print("Based runtime average, ETA is {0:6.2f}".format(ETA), " minutes")
        audio_input, rate = librosa.load(join(path2audiochunks, audio_chunk), sr=16000)
        input_values = tokenizer(audio_input, return_tensors="pt", padding="longest", truncation=True).input_values
        logits = transcription_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        full_transcription.append(transcription + "\n")

    print("\nFinished audio transcription of " + vid_clip_name + " and now saving metrics.")

    # build metadata log -------------------------------------------------
    mdata = []
    mdata.append('original file name: ' + vid_clip_name + '\n')
    mdata.append(
        'number of recorded audio chunks: ' + str(len(chunk_directory)) + " of lengths seconds each" + str(
            chunk_length_seconds) + '\n')
    approx_input_len = (len(chunk_directory) * chunk_length_seconds) / 60
    mdata.append('approx {0:3f}'.format(approx_input_len) + ' minutes of input audio \n')
    mdata.append('transcription date: ' + datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S") + '\n')
    full_text = ' '.join(full_transcription)
    transcript_length = len(full_text)
    mdata.append("length of transcribed text: " + str(transcript_length) + ' characters \n')
    t_word_count = len(full_text.split(' '))
    mdata.append("total word count: " + str(t_word_count) + ' words (based on spaces) \n')

    # delete audio chunks in folder -------------------------------------------------
    # TODO add try/except for deleting folder as not technically needed to achieve goal
    shutil.rmtree(path2audiochunks)
    print("\nDeleted Audio Chunk Folder + Files")

    # compile results -------------------------------------------------
    transcription_results = {
        "audio_transcription": full_transcription,
        "metadata": mdata
    }
    print("\nFinished transcription successfully for " + vid_clip_name + " at "
          + datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S"))
    return transcription_results


def validate_output_directories(directory):
    t_folder_name = "wav2vec2_sf_transcript"
    m_folder_name = "wav2vec2_sf_metadata"

    # check if transcription folder exists. If not, create it
    if not os.path.isdir(join(directory, t_folder_name)):
        os.mkdir(join(directory, t_folder_name))  # make a place to store outputs if one does not exist
    t_path_full = join(directory, t_folder_name)

    # check if metadata folder exists. If not, create it
    if not os.path.isdir(join(directory, m_folder_name)):
        os.mkdir(join(directory, m_folder_name))  # make a place to store outputs if one does not exist
    m_path_full = join(directory, m_folder_name)

    output_locs = {
        "t_out": t_path_full,
        "m_out": m_path_full
    }

    return output_locs


# -------------------------------------------------------
# Main Script
# -------------------------------------------------------
# start tracking rt
time_log = []
time_log_desc = []
time_log.append(time.time())
time_log_desc.append("start")

# load audio
directory = r'C:\Users\peter\PycharmProjects\vid2cleantxt\example_JFK_speech'
# if file is locally stored in the same folder as the script, you can do this:
# directory = os.getcwd()
# enter video clip name
video_file_name = 'JFK_rice_moon_speech.mp4'

# load pretrained model
# if running for first time on local machine, start with "facebook/wav2vec2-base-960h" for both tokenizer and model
wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"  # tested up to 35 second chunks
# wav2vec2_model = "facebook/wav2vec2-base-960h" # tested up to 90 second chunks. Faster, but less accurate
print("\nPreparing to load model: " + wav2vec2_model)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(wav2vec2_model)
model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
chunk_length = 35  # (in seconds) if model fails to work or errors out (and there isn't some other
# obvious error, reduce this number.

time_log.append(time.time())
time_log_desc.append("loaded model")

# transcribe the video file
t_results = transcribe_video_wav2vec(transcription_model=model, directory=directory,
                                     vid_clip_name=video_file_name, chunk_length_seconds=chunk_length)
time_log.append(time.time())
time_log_desc.append("transcribed video")

# t_results is a dictonary containing the transcript and associated metadata
full_transcription = t_results.get('audio_transcription')
metadata = t_results.get('metadata')
# check if directories for output exist. If not, create them
storage_locs = validate_output_directories(directory)
output_path_transcript = storage_locs.get('t_out')
output_path_metadata = storage_locs.get('m_out')

# label and store this transcription
vid_preamble = beautify_filename(video_file_name, num_words=15, start_reverse=False)  # gets a nice phrase from filename
# transcription
transcribed_filename = vid_preamble + '_tscript_' + datetime.now().strftime("_%H.%M.%S") + '.txt'
transcribed_file = open(join(output_path_transcript, transcribed_filename), 'w', encoding="utf-8", errors='ignore')
transcribed_file.writelines(full_transcription)
transcribed_file.close()
# metadata
metadata_filename = 'metadata for ' + vid_preamble + " transcription.txt"
metadata_file = open(join(output_path_metadata, metadata_filename), 'w', encoding="utf-8", errors='ignore')
metadata_file.writelines(metadata)
metadata_file.close()

time_log.append(time.time())
time_log_desc.append("saved output files")

# Go through base transcription files and spell correct them and get keywords
print('\n Starting to spell-correct and extract keywords\n')
seg = pysbd.Segmenter(language="en", clean=True)
tf_pretty_name = beautify_filename(transcribed_filename, start_reverse=False, num_words=10)
# auto-correct spelling (wav2vec2 doesn't enforce spelling on its output)
corr_results_fl = symspell_file(filepath=output_path_transcript, filename=transcribed_filename, keep_numb_words=True,
                                create_folder=True, dist=2)
output_path_impr = corr_results_fl.get("output_path")

# Write version of transcription with sentences / boundaries inferred with periods. All text in one line
seg_list = seg.segment(corr_results_fl.get("corrected_ssp_text"))
seg_text = '. '.join(seg_list)
seg_outname = "SegTEXT " + tf_pretty_name + ".txt"
file_seg = open(join(output_path_impr, seg_outname), 'w', encoding="utf-8", errors='ignore')
file_seg.write(seg_text)
file_seg.close()

# extract keywords from transcription (once spell-corrected)
key_phr_fl = quick_keys(filepath=output_path_impr, filename=corr_results_fl.get("corrected_ssp_fname"),
                        num_keywords=50, max_ngrams=3, save_db=False)
key_phr_fl.to_excel(join(output_path_transcript, "YAKE_extracted_keywords.xlsx"))

time_log.append(time.time())
time_log_desc.append("transcription spell-corrected + keywords extracted")

# ----------------------------------- END -------------------------------
print("\n\n----------------------------------- Script Complete -------------------------------")
print("Transcription file + more can be found here: ", output_path_transcript)
print("Metadata for each transcription is located: ", output_path_metadata)
time_log.append(time.time())
time_log_desc.append("End")
# save runtime database
time_records_db = pd.DataFrame(list(zip(time_log_desc, time_log)), columns=['Event', 'Time (sec)'])
time_records_db.to_excel(join(output_path_metadata, "transcription_time_log.xlsx"))
# TODO add two columns / functions for adding time difference relative to start in minutes and seconds
print("total runtime was {0:5f}".format((time_log[-1] - time_log[0]) / 60), " minutes")
