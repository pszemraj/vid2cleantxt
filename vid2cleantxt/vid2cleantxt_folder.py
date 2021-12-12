# this is the script you should run to transcribe files in 95+% of cases.

"""
vid2clntext by Peter Szemraj

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
This script is the 'folder' edition

Tips for runtime:

start with "facebook/wav2vec2-base-960h" for both tokenizer and model
if model fails to work or errors out, try reducing the chunk length
"""
import math
import shutil
import sys
import time
from datetime import datetime
from os.path import join

import librosa
import pandas as pd
import argparse
from pkg_resources import require
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
    setup_out_dirs,
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
    get_timestamp,
)


# -------------------------------------------------------
# Function Definitions
# -------------------------------------------------------


def transcribe_video_wav2vec(
    ts_model,
    directory,
    vid_clip_name,
    chunk_dur: int,
    verbose=False,
):
    # this is the same process as used in the single video transcription, now as a function. Note that spell
    # correction and keyword extraction are now done separately in the script  user needs to pass in: the model,
    # the folder the video is in, and the name of the video

    # Split Video into Audio Chunks-----------------------------------------------
    if verbose:
        print(f"Starting to transcribe {vid_clip_name} @ {get_timestamp()}")
    # create audio chunk folder
    output_folder_name = "audio_chunks"
    ac_storedir = join(directory, output_folder_name)
    create_folder(ac_storedir)
    chunk_directory = convert_vid_for_transcription(
        vid2beconv=vid_clip_name,
        input_directory=directory,
        len_chunks=chunk_dur,
        output_directory=ac_storedir,
    )
    torch_validate_cuda()
    check_runhardware()
    full_transc = []
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
        audio_input, rate = librosa.load(join(ac_storedir, audio_chunk), sr=16000)
        # MODEL
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_values = tokenizer(
            audio_input, return_tensors="pt", padding="longest", truncation=True
        ).input_values.to(device)
        ts_model = ts_model.to(device)
        logits = ts_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        this_transc = str(tokenizer.batch_decode(predicted_ids)[0])
        full_transc.append(this_transc + "\n")
        # empty memory so you don't overload the GPU
        del input_values
        del logits
        del predicted_ids
        torch.cuda.empty_cache()

    if verbose: print(f"Finished transcribing {vid_clip_name} @ {get_timestamp()}")

    md_df = create_metadata_df()  # makes a blank df with column names
    approx_input_len = (len(chunk_directory) * chunk_dur) / 60
    transc_dt = get_timestamp()
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        vid_clip_name,
        len(chunk_directory),
        chunk_dur,
        approx_input_len,
        transc_dt,
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(copy=False)
    shutil.rmtree(ac_storedir, ignore_errors=True)  # remove audio chunks folder
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print("finished transcription of {vid_clip_name} base folder on {}")

    return transc_res


def get_parser():
    """
    get_parser - a helper function for the argparse module

    Returns: argparse.ArgumentParser object
    """

    parser = argparse.ArgumentParser(
        description="submit a message and have a 774M parameter GPT model respond"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="path to directory containing video files to be transcribed",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        required=False,
        help="folder - where to save the output files. If not specified, will save to the input-dir",
    )
    parser.add_argument(
        "--move-input-vids",
        default=False,
        action="store_true",
        help="if specified, will move the files to the completed folder",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="print out more information",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="huggingface model name as a string, ex facebook/wav2vec2-large-960h-lv60-self",
    )

    parser.add_argument(
        "--chunk-length",
        default=20,
        type=int,
        help="Duration of audio chunks (in seconds) that the transformer model will be fed",
    )

    return parser


# -------------------------------------------------------
# Main Script
# -------------------------------------------------------

if __name__ == "__main__":
    st = time.perf_counter()
    args = get_parser().parse_args()
    directory = str(args.input_dir)
    is_verbose = args.verbose
    chunk_length = int(args.chunk_length)
    model_arg = args.model_name

    # load model
    wav2vec2_model = (
        "facebook/wav2vec2-large-960h-lv60-self" if model_arg is None else model_arg
    )
    if is_verbose:
        print("Loading model: {}".format(wav2vec2_model))
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(wav2vec2_model)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)

    orig_out = sys.__stdout__
    sys.stdout = NullIO()  # hide printing to console for initializations below:
    checker = init_neuspell()
    sym_spell = init_symspell()
    sys.stdout = orig_out  # return to default of print-to-console

    vid_extensions = [".mp4", ".mov", ".avi"]
    approved_files = []
    for ext in vid_extensions:
        approved_files.extend(
            load_imm_dir_files(directory, req_ext=ext, full_path=False)
        )

    print(f"\nFound {len(approved_files)} video files in {directory}")

    # iterate through list of video files, transcribing one at a time --------------------------------------------------
    storage_locs = setup_out_dirs(
        directory
    )  # create and get output folders
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")

    for filename in tqdm(
        approved_files,
        total=len(approved_files),
        desc="transcribing vids",
    ):
        # transcribe video and get results
        t_results = transcribe_video_wav2vec(
            ts_model=model,
            directory=directory,
            vid_clip_name=filename,
            chunk_dur=chunk_length,
        )
        full_transcription = t_results.get("audio_transcription")
        metadata = t_results.get("metadata")

        # label and store this transcription
        v_lbl = beautify_filename(filename, num_words=15, start_reverse=False)
        # transcription
        t_file = f"vid2text_{v_lbl}_tranc_{get_timestamp()}.txt"

        with open(
            join(out_p_tscript, t_file),
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as tf:
            tf.writelines(full_transcription)  # save transcription

        metadata_filename = f"metadata - {v_lbl} - transcription.csv"
        metadata.to_csv(join(out_p_metadata, metadata_filename), index=True)

        move2completed(directory, filename=filename)

    if is_verbose:
        print(f"finished transcribing all files at {get_timestamp()}")

    # ----------------------------------- Merge Text Files  -------------------------------
    # makes things easier to review when you only have to open one file instead of N

    digest_txt_directory(
        out_p_tscript, identifer=f"original_tscripts_{get_timestamp()}"
    )
    digest_txt_directory(
        out_p_metadata,
        identifer=f"trans_metadata_{get_timestamp()}",
        make_folder=False,
    )

    # add spelling correction to output transcriptions
    txt_files = load_imm_dir_files(
        out_p_tscript, req_ext=".txt", verbose=True, full_path=False
    )
    if is_verbose:
        print(f"\nchecking {len(txt_files)} files for spelling errors")
    # Spellcorrect Pipeline
    kw_all_vids = pd.DataFrame()

    max_item = len(txt_files)

    for origi_tscript in tqdm(
        txt_files,
        total=len(txt_files),
        desc="SC_pipeline - transcribed audio",
    ):
        current_loc = txt_files.index(origi_tscript) + 1  # add 1 bc start at 0
        PL_out = spellcorrect_pipeline(
            out_p_tscript, origi_tscript, verbose=False, ns_checker=checker
        )
        # get locations of where corrected files were saved
        kw_dir = PL_out.get("spell_corrected_dir")
        kw_name = PL_out.get("sc_filename")
        # extract keywords from the saved file
        qk_df = quick_keys(
            filepath=kw_dir,
            filename=kw_name,
            num_keywords=25,
            max_ngrams=3,
            save_db=False,
            verbose=is_verbose,
        )

        kw_all_vids = pd.concat([kw_all_vids, qk_df], axis=1)

    # save overall transcription file
    keyword_db_name = f"YAKE - all keys for batch {get_timestamp()}.csv"
    kw_all_vids.to_csv(
        join(out_p_tscript, keyword_db_name),
        index=True,
    )

    print(
        "A file with keyword results is in {} \ntitled {}".format(
            out_p_tscript, keyword_db_name
        )
    )
    print("Finished at: ", get_timestamp())
    print(
        "The relevant files for this run are located in here: \n {out_p_tscript} \n and {out_p_metadata}"
    )
    print("total runtime was {} minutes".format(round((time.time() - st) / 60), 2))
