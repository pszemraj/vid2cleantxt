"""

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
This script is the 'folder' edition
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
import shutil
import time
from datetime import datetime
from os import listdir
from os.path import isfile
from os.path import join

import librosa
import pandas as pd
import pysbd
import torch
from natsort import natsorted
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

from audio2text_functions import beautify_filename, convert_vid_for_transcription, quick_keys, \
    digest_text_fn, symspell_file


# -------------------------------------------------------
# Function Definitions
# -------------------------------------------------------

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
    t_folder_name = "w2v2_video_transcriptions"
    m_folder_name = "w2v2_transcription_metadata"

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
# TODO create separate files with no import local functions
# start tracking rt
time_log = []
time_log_desc = []
time_log.append(time.time())
time_log_desc.append("start")

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

# Ask user for folder
print('\n')
print(r"Example folder path: C:\Users\peter\PycharmProjects\vid2cleantxt\example_JFK_speech")
directory = str(input("\n Enter path to folder containing videos to be transcribed here ---->"))
tr_db_name = 'runtime data for of ' + ''.join(ch for ch in directory if ch.isalnum()) + " transcription.xlsx"

# iterate through directory and get list of only video files --------------------------------------------------

files_to_munch = natsorted([f for f in listdir(directory) if isfile(join(directory, f))])
total_files_1 = len(files_to_munch)
removed_count_1 = 0
approved_files = []
# remove non-.txt files
for prefile in files_to_munch:
    if prefile.endswith(".mp4") or prefile.endswith(".mov") or prefile.endswith(".avi"):
        approved_files.append(prefile)
    else:
        files_to_munch.remove(prefile)
        removed_count_1 += 1

print("out of {0:3d} file(s) originally in the folder, ".format(total_files_1),
      "{0:3d} non-video files were removed".format(removed_count_1))
print('\n {0:3d} video file(s) in folder will be transcribed.'.format(len(approved_files)))

# iterate through list of video files, transcribing one at a time --------------------------------------------------

for filename in approved_files:
    # transcribe the video file
    t_results = transcribe_video_wav2vec(transcription_model=model, directory=directory,
                                         vid_clip_name=filename, chunk_length_seconds=chunk_length)
    # t_results is a dictonary containing the transcript and associated metadata
    full_transcription = t_results.get('audio_transcription')
    metadata = t_results.get('metadata')
    # check if directories for output exist. If not, create them
    storage_locs = validate_output_directories(directory)
    output_path_transcript = storage_locs.get('t_out')
    output_path_metadata = storage_locs.get('m_out')

    # label and store this transcription
    vid_preamble = beautify_filename(filename, num_words=15, start_reverse=False)  # gets a nice phrase from filename
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

    print("{:3d}".format(files_to_munch.index(filename) + 1), " of {:3d} total files done".format(len(files_to_munch)))
    time_log.append(time.time())
    time_log_desc.append("transcribed the file = " + filename)

    # save runtime database after each run
    time_records_db = pd.DataFrame(list(zip(time_log_desc, time_log)), columns=['Event', 'Time (sec)'])
    time_records_db.to_excel(join(output_path_metadata, tr_db_name))

time_log.append(time.time())
time_log_desc.append("all transcriptions completed")
# ----------------------------------- Merge Text Files  -------------------------------
# makes things easier to review when you only have to open one file instead of N

post_run_now = datetime.now().strftime("date_%d_%m_%Y_time_%H")
print("Creating merged files from original transcriptions")
digest_text_fn(output_path_transcript, iden="original_tscripts" + post_run_now, w_folder=True)
digest_text_fn(output_path_metadata, iden="metadata_for_tscript_run" + post_run_now)

# ----------------------------------- Improve Base Transcriptions  -------------------------------
# iterate through list of transcribed files, correcting spelling and generating keywords

# first, you need to go through the output directory of transcripts and make sure that all those files are gucci
transcripts_to_munch = natsorted(
    [f for f in listdir(output_path_transcript) if isfile(join(output_path_transcript, f))])
t_files = len(transcripts_to_munch)
removed_count_t = 0
# remove non-.txt files
for tfile in transcripts_to_munch:
    if tfile.endswith(".txt"):
        continue
    else:
        transcripts_to_munch.remove(tfile)
        removed_count_t += 1

print("out of {0:3d} file(s) originally in the folder, ".format(t_files),
      "{0:3d} non-txt files were removed".format(removed_count_t))
# Go through base transcription files and spell correct them and get keywords
print('\n Starting to spell-correct and extract keywords\n')
seg = pysbd.Segmenter(language="en", clean=True)
keyphrase_df_transc = pd.DataFrame()
kp_fname = "YAKE_keywords_for_all_transcr.xlsx"
for textfile in transcripts_to_munch:
    tf_pretty_name = beautify_filename(textfile, start_reverse=False, num_words=10)

    # auto-correct spelling (wav2vec2 doesn't enforce spelling on its output)
    corr_results_fl = symspell_file(filepath=output_path_transcript, filename=textfile, keep_numb_words=True,
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

    # edit returned keyword dataframe from quick_keys() and append to total one
    db_col_name = ''.join(list([val for val in tf_pretty_name if val.isalnum()]))
    key_phr_fl.drop(key_phr_fl.columns[[1, 2]], axis=1, inplace=True)
    key_phr_fl.columns = ["YAKE_keywords_from_" + db_col_name, "freq_of_kw_in_" + db_col_name]

    if transcripts_to_munch.index(textfile) == 0:
        # on first iteration set blank df equal to keyword df from first file
        keyphrase_df_transc = key_phr_fl
    else:
        # update the dataframe adding columns on the right (colnames should be unique from manual naming above)
        keyphrase_df_transc = pd.concat([keyphrase_df_transc, key_phr_fl], axis=1)

    # save each iteration
    keyphrase_df_transc.to_excel(join(output_path_transcript, kp_fname))

time_log.append(time.time())
time_log_desc.append("transcriptions spell-corrected + keywords extracted")

# ----------------------------------- END -------------------------------
print("\n\n----------------------------------- Script Complete -------------------------------")
print("Transcription files + more can be found here: ", output_path_transcript)
print("Metadata for each transcription is located: ", output_path_metadata)
time_log.append(time.time())
time_log_desc.append("End")
# save runtime database one last time
time_records_db = pd.DataFrame(list(zip(time_log_desc, time_log)), columns=['Event', 'Time (sec)'])
time_records_db.to_excel(join(output_path_metadata, tr_db_name))
# TODO add two columns / functions for adding time difference relative to start in minutes and seconds
print("total runtime was {0:5f}".format((time_log[-1] - time_log[0]) / 60), " minutes")
