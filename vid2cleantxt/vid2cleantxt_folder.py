# this is the script you should run to transcribe files in 95+% of cases.

"""
vid2clntext by Peter Szemraj
Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
This script is the 'folder' edition

large model link / doc from host website (huggingface)
https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

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
import shutil
import sys
import time
from datetime import datetime
from os.path import join

import librosa
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

from audio2text_functions import (
    beautify_filename,
    convert_vid_for_transcription,
    corr,
    create_metadata_df,
    init_neuspell,
    init_symspell,
    quick_keys,
    spellcorrect_pipeline,
    validate_output_directories,
)
from v2ct_utils import (
    check_runhardware,
    create_folder,
    digest_txt_directory,
    load_imm_dir_files,
    move2completed,
    NullIO,
    shorten_title,
    torch_validate_cuda,
)


# -------------------------------------------------------
# Function Definitions
# -------------------------------------------------------


def transcribe_video_wav2vec(
    transcription_model, directory, vid_clip_name, chunk_length_seconds, verbose=False
):
    # this is the same process as used in the single video transcription, now as a function. Note that spell
    # correction and keyword extraction are now done separately in the script  user needs to pass in: the model,
    # the folder the video is in, and the name of the video

    # Split Video into Audio Chunks-----------------------------------------------
    if verbose:
        print("Starting to transcribe {} @ {}".format(vid_clip_name, datetime.now()))
    # create audio chunk folder
    output_folder_name = "audio_chunks"
    path2audiochunks = join(directory, output_folder_name)
    create_folder(path2audiochunks)
    chunk_directory = convert_vid_for_transcription(
        vid2beconv=vid_clip_name,
        input_directory=directory,
        len_chunks=chunk_length_seconds,
        output_directory=path2audiochunks,
    )
    torch_validate_cuda()
    check_runhardware()
    full_transcription = []
    GPU_update_incr = math.ceil(len(chunk_directory) / 2)

    # Load audio chunks by name, pass into model, append output text-----------------------------------------------

    for audio_chunk in tqdm(
        chunk_directory,
        total=len(chunk_directory),
        desc="Transcribing {} ".format(shorten_title(vid_clip_name)),
    ):

        current_loc = chunk_directory.index(audio_chunk)

        if (current_loc % GPU_update_incr == 0) and (GPU_update_incr != 0):
            # provide update on GPU usage
            check_runhardware()

        # load dat chunk
        audio_input, rate = librosa.load(join(path2audiochunks, audio_chunk), sr=16000)
        # MODEL
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_values = tokenizer(
            audio_input, return_tensors="pt", padding="longest", truncation=True
        ).input_values.to(device)
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

    if verbose:
        print(
            "\nFinished audio transcription of "
            + vid_clip_name
            + " and now saving metrics."
        )

    # build metadata log -------------------------------------------------
    md_df = create_metadata_df()  # makes a blank df with column names
    approx_input_len = (len(chunk_directory) * chunk_length_seconds) / 60
    transc_dt = datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S")
    full_text = corr(" ".join(full_transcription))
    md_df.loc[len(md_df), :] = [
        vid_clip_name,
        len(chunk_directory),
        chunk_length_seconds,
        approx_input_len,
        transc_dt,
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(copy=False)
    # delete audio chunks in folder -------------------------------------------------
    try:
        shutil.rmtree(path2audiochunks)
        if verbose:
            print("\nDeleted Audio Chunk Folder + Files")
    except:
        print(
            "WARNING - unable to clean up + delete the audio_chunks folder for {}".format(
                vid_clip_name
            )
        )

    # compile results -------------------------------------------------
    transcription_results = {
        "audio_transcription": full_transcription,
        "metadata": md_df,
    }

    if verbose:
        print(
            "\nFinished transcription successfully for "
            + vid_clip_name
            + " at "
            + datetime.now().strftime("date_%d_%m_%Y_time_%H-%M-%S")
        )

    return transcription_results


# -------------------------------------------------------
# Main Script
# -------------------------------------------------------

if __name__ == "__main__":
    st = time.time()  # start tracking rt
    directory = str(
        input("\n Enter full path to directory containing videos ---->")
    )  # Ask user for folder

    # load pretrained model
    # if running for first time on local machine, start with "facebook/wav2vec2-base-960h" for both tokenizer and model
    wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"  # load pretrained model
    # wav2vec2_model = "facebook/wav2vec2-base-960h" # tested up to 90 second chunks. Faster, but less accurate
    print("\nPreparing to load model: " + wav2vec2_model + " - ", datetime.now())
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(wav2vec2_model)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
    chunk_length = 30  # (in seconds) if model fails to work or errors out, try reducing this number.

    orig_out = sys.__stdout__
    sys.stdout = NullIO()  # hide printing to console for initializations below:
    checker = init_neuspell()
    sym_spell = init_symspell()
    sys.stdout = orig_out  # return to default of print-to-console

    # iterate through directory and get list of only video files --------------------------------------------------
    vid_extensions = [".mp4", ".mov", ".avi"]
    approved_files = []
    for ext in vid_extensions:
        approved_files.extend(
            load_imm_dir_files(directory, req_ext=ext, full_path=False)
        )

    print(
        "\n# of video files with valid extensions {} in src dir is {}".format(
            vid_extensions, len(approved_files)
        )
    )

    # iterate through list of video files, transcribing one at a time --------------------------------------------------
    storage_locs = validate_output_directories(
        directory
    )  # create and get output folders
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")

    for filename in tqdm(
        approved_files,
        total=len(approved_files),
        desc="transcribing video files from folder",
    ):
        # transcribe the video file
        t_results = transcribe_video_wav2vec(
            transcription_model=model,
            directory=directory,
            vid_clip_name=filename,
            chunk_length_seconds=chunk_length,
        )
        full_transcription = t_results.get("audio_transcription")
        metadata = t_results.get("metadata")

        # label and store this transcription
        vid_preamble = beautify_filename(filename, num_words=15, start_reverse=False)
        # transcription
        transcribed_filename = (
            vid_preamble + "_tscript_" + datetime.now().strftime("_%H.%M.%S") + ".txt"
        )
        transcribed_file = open(
            join(out_p_tscript, transcribed_filename),
            "w",
            encoding="utf-8",
            errors="ignore",
        )
        with open(
            join(out_p_tscript, transcribed_filename),
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as tf:
            tf.writelines(full_transcription)  # save transcription

        metadata_filename = "metadata for " + vid_preamble + " transcription.csv"
        metadata.to_csv(join(out_p_metadata, metadata_filename), index=True)

        move2completed(directory, filename=filename)

    print("\n----------------------------")
    print("transcription process completed - ", datetime.now(), "\n")

    # ----------------------------------- Merge Text Files  -------------------------------
    # makes things easier to review when you only have to open one file instead of N

    pr_stamp = datetime.now().strftime("date_%d_%m_%Y_time_%H")
    digest_txt_directory(out_p_tscript, identifer="original_tscripts" + pr_stamp)
    digest_txt_directory(
        out_p_metadata,
        identifer="metadata_for_tscript_run" + pr_stamp,
        make_folder=False,
    )

    # ----------------------------------- Improve Base Transcriptions  -------------------------------
    # Validate text files to spell-check (in case there was a change)
    total_files = len(load_imm_dir_files(out_p_tscript, req_ext=".txt"))
    approved_txt_files = load_imm_dir_files(
        out_p_tscript, req_ext=".txt", verbose=True, full_path=False
    )
    print(
        "from {} file(s) in dir, loading {} .txt files".format(
            total_files, len(approved_txt_files)
        )
    )

    # Spellcorrect Pipeline
    transcript_run_qk = pd.DataFrame()  # empty df to hold all the keywords

    max_item = len(approved_txt_files)

    for origi_tscript in tqdm(
        approved_txt_files,
        total=len(approved_txt_files),
        desc="SC_pipeline - transcribed audio",
    ):
        current_loc = approved_txt_files.index(origi_tscript) + 1  # add 1 bc start at 0
        PL_out = spellcorrect_pipeline(
            out_p_tscript, origi_tscript, verbose=False, ns_checker=checker
        )
        # get locations of where corrected files were saved
        directory_for_keywords = PL_out.get("spell_corrected_dir")
        filename_for_keywords = PL_out.get("sc_filename")
        # extract keywords from the saved file
        qk_df = quick_keys(
            filepath=directory_for_keywords,
            filename=filename_for_keywords,
            num_keywords=25,
            max_ngrams=3,
            save_db=False,
            verbose=False,
        )

        transcript_run_qk = pd.concat([transcript_run_qk, qk_df], axis=1)

    # save overall transcription file
    keyword_db_name = "YAKE - keywords for all transcripts in run.csv"
    transcript_run_qk.to_csv(
        join(out_p_tscript, "YAKE - keywords for all transcripts in run.csv"),
        index=True,
    )

    print(
        "Transcription files used to extract KW can be found in: \n ",
        directory_for_keywords,
    )
    print(
        "A file with keyword results is in {} \ntitled {}".format(
            out_p_tscript, keyword_db_name
        )
    )

    print(
        "\n\n----------------------------------- Script Complete -------------------------------"
    )
    print(datetime.now())
    print("Transcription files + more in folder: \n", out_p_tscript)
    print("Metadata for each transcription located @ \n", out_p_metadata)
    print("total runtime was {} minutes".format(round((time.time() - st) / 60), 2))
