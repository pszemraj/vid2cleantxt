# this script is typically just a less flexible variant of the folder version. It's easier to follow
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
from os.path import basename, dirname, join

import librosa
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, AutoTokenizer

from audio2text_functions import (
    trim_fname,
    prep_transc_src,
    corr,
    create_metadata_df,
    init_neuspell,
    init_symspell,
    quick_keys,
    spellcorrect_pipeline,
    setup_out_dirs,
)
from v2ct_utils import (
    check_runhardware,
    create_folder,
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
        print(f"Starting to transcribe {vid_clip_name} @ {datetime.now()}")
    # create audio chunk folder
    output_folder_name = "audio_chunks"
    path2audiochunks = join(directory, output_folder_name)
    create_folder(path2audiochunks)
    chunk_directory = prep_transc_src(
        _vid2beconv=vid_clip_name,
        in_dir=directory,
        len_chunks=chunk_length_seconds,
        out_dir=path2audiochunks,
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
    st = time.perf_counter()  # start tracking rt
    filepath = str(
        input("\n Enter full path to the video to be transcribed ---->")
    )  # Ask user for folder

    # load pretrained model
    # if running for first time on local machine, start with "facebook/wav2vec2-base-960h" for both tokenizer and model
    wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"  # load pretrained model
    # wav2vec2_model = "facebook/wav2vec2-base-960h" # tested up to 90 second chunks. Faster, but less accurate
    print("\nPreparing to load model: " + wav2vec2_model + " - ", datetime.now())
    tokenizer = AutoTokenizer.from_pretrained(wav2vec2_model)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
    chunk_length = 30  # (in seconds) if model fails to work or errors out, try reducing this number.

    orig_out = sys.__stdout__
    sys.stdout = NullIO()  # hide printing to console for initializations below:
    checker = init_neuspell()
    sym_spell = init_symspell()
    sys.stdout = orig_out  # return to default of print-to-console
    filename = basename(filepath)
    directory = dirname(filepath)
    # iterate through list of video files, transcribing one at a time --------------------------------------------------
    storage_locs = setup_out_dirs(directory)  # create and get output folders
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")

    t_results = transcribe_video_wav2vec(
        transcription_model=model,
        directory=directory,
        vid_clip_name=filename,
        chunk_length_seconds=chunk_length,
    )
    full_transcription = t_results.get("audio_transcription")
    metadata = t_results.get("metadata")

    # label and store this transcription
    vid_preamble = trim_fname(filename, num_words=15, start_rev=False)
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
    t_out_path = join(out_p_tscript, transcribed_filename)
    with open(t_out_path, "w", encoding="utf-8", errors="ignore") as tf:
        tf.writelines(full_transcription)  # save transcription

    metadata_filename = "metadata for " + vid_preamble + " transcription.csv"
    m_out_path = join(out_p_metadata, metadata_filename)
    metadata.to_csv(m_out_path, index=True)
    move2completed(directory, filename=filename)

    print("\n----------------------------")
    print("transcription process completed - ", datetime.now(), "\n")

    # ----------------------------------- Improve Base Transcriptions  -------------------------------

    # Spellcorrect Pipeline
    PL_out = spellcorrect_pipeline(
        out_p_tscript, transcribed_filename, ns_checker=checker
    )
    # get locations of where corrected files were saved
    directory_for_keywords = PL_out.get("spell_corrected_dir")
    filename_for_keywords = PL_out.get("sc_filename")
    # extract keywords from the saved file
    qk_df = quick_keys(
        filepath=directory_for_keywords,
        filename=filename_for_keywords,
        num_kw=25,
        max_ngrams=3,
        save_db=False,
        verbose=False,
    )
    qk_df.to_csv(
        join(
            out_p_tscript,
            "YAKE - keywords for {}.csv".format(trim_fname(filename_for_keywords)),
        ),
        index=True,
    )

    print(
        "Transcription files used to extract KW can be found in: \n ",
        directory_for_keywords,
    )

    print(
        "\n\n----------------------------------- Script Complete -------------------------------"
    )
    print(datetime.now())
    print("Transcription files + more in folder: \n", out_p_tscript)
    print("Metadata for each transcription located @ \n", out_p_metadata)
    print(
        "total runtime was {} minutes".format(round((time.perf_counter() - st) / 60), 2)
    )
