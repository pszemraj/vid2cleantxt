# this is the script you should run to transcribe files in 95+% of cases.

"""
vid2clntext by Peter Szemraj

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model

Tips for runtime:

start with "facebook/wav2vec2-base-960h" for both tokenizer and model
if model fails to work or errors out, try reducing the chunk length
"""

# TODO: add code to add this file's path to the root path
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
    trim_fname,
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
    find_ext_local,
    move2completed,
    NullIO,
    torch_validate_cuda,
    get_timestamp,
)


# -------------------------------------------------------
# Function Definitions
# -------------------------------------------------------


def transcribe_video_wav2vec(
    ts_model,
    src_dir,
    clip_name,
    chunk_dur: int,
    verbose=False,
    temp_dir: str = "audio_chunks",
):
    """
    transcribe_video_wav2vec - transcribes a video clip using the wav2vec2 model

    Parameters
    ----------
    ts_model : [type]
        [description]
    directory : [type]
        [description]
    vid_clip_name : [type]
        [description]
    chunk_dur : int
        [description]
    verbose : bool, optional
        [description], by default False
    temp_dir : str, optional
        [description], by default "audio_chunks"

    Returns
    -------
    [type]
        [description]
    """

    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    # create audio chunk folder
    ac_storedir = join(src_dir, temp_dir)
    create_folder(ac_storedir)
    chunk_directory = convert_vid_for_transcription(
        vid2beconv=clip_name,
        in_dir=src_dir,
        len_chunks=chunk_dur,
        out_dir=ac_storedir,
    )
    torch_validate_cuda()
    check_runhardware()
    full_transc = []
    GPU_update_incr = math.ceil(len(chunk_directory) / 2)

    for audio_chunk in tqdm(
        chunk_directory,
        total=len(chunk_directory),
        desc="Transcribing video",
    ):

        current_loc = chunk_directory.index(audio_chunk)

        if (current_loc % GPU_update_incr == 0) and (GPU_update_incr != 0):
            # provide update on GPU usage
            check_runhardware()

        audio_input, rate = librosa.load(
            join(ac_storedir, audio_chunk), sr=16000
        )  # 16000 is the sampling rate of the wav2vec model
        # MODEL
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPU or CPU
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

    if verbose:
        print(f"Finished transcribing {clip_name} @ {get_timestamp()}")

    md_df = create_metadata_df()  # makes a blank df with column names
    approx_input_len = (len(chunk_directory) * chunk_dur) / 60
    transc_dt = get_timestamp()
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
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
        description="Transcribe a directory of videos using wav2vec2"
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
        required=False,

        default=False,
        action="store_true",
        help="if specified, will move the files to the completed folder",
    )
    parser.add_argument(
        "--verbose",
        required=False,

        default=False,
        action="store_true",
        help="print out more information",
    )

    parser.add_argument(
        "--model-name",
        required=False,

        default=None,
        help="huggingface model name as a string, ex facebook/wav2vec2-large-960h-lv60-self",
    )

    parser.add_argument(
        "--chunk-length",
        required=False,

        default=20,
        type=int,
        help="Duration of audio chunks (in seconds) that the transformer model will be fed",
    )

    return parser


if __name__ == "__main__":

    st = time.perf_counter()
    # parse the command line arguments
    args = get_parser().parse_args()
    directory = str(args.input_dir)
    is_verbose = args.verbose
    move_comp = args.move_input_vids
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

    vid_extensions = [".mp4", ".mov", ".avi"]  # may add more later
    approved_files = []
    for ext in vid_extensions:
        approved_files.extend(
            find_ext_local(directory, req_ext=ext, full_path=False)
        )

    print(f"\nFound {len(approved_files)} video files in {directory}")

    storage_locs = setup_out_dirs(directory)  # create and get output folders
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
            src_dir=directory,
            clip_name=filename,
            chunk_dur=chunk_length,
        )
        t_finished = t_results.get("audio_transcription")
        metadata = t_results.get("metadata")

        # label and store transcription
        v_lbl = trim_fname(filename, num_words=15, start_reverse=False)
        t_file = f"vid2text_{v_lbl}_tranc_{get_timestamp()}.txt"

        with open(
            join(out_p_tscript, t_file),
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as tf:
            tf.writelines(t_finished)

        metadata_filename = f"metadata - {v_lbl} - transcription.csv"
        metadata.to_csv(join(out_p_metadata, metadata_filename), index=True)

        if move_comp:
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
    txt_files = find_ext_local(
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
            num_kw=25,
            max_ngrams=3,
            save_db=False,
            verbose=is_verbose,
        )

        kw_all_vids = pd.concat([kw_all_vids, qk_df], axis=1)

    # save overall transcription file
    kwdb_fname = f"YAKE - all keys for batch {get_timestamp()}.csv"
    kw_all_vids.to_csv(
        join(out_p_tscript, kwdb_fname),
        index=True,
    )

    print(f"A file with keyword results is in {out_p_tscript} \ntitled {kwdb_fname}")
    print("Finished at: ", get_timestamp())
    print(
        "The relevant files for this run are located in here: \n {out_p_tscript} \n and {out_p_metadata}"
    )
    print("total runtime was {} minutes".format(round((time.perf_counter() - st) / 60), 2))
