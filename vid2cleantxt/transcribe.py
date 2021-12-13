#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vid2clntext by Peter Szemraj

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
this is the primary pipeline for the project

You can access the arguments for this script by running the following command:
    *\vid2cleantxt\vid2cleantxt_folder.py -h (windows)
    */vid2cleantxt/vid2cleantxt_folder.py -h (everything else)

Tips for runtime:

start with "facebook/wav2vec2-base-960h" for both tokenizer and model
if model fails to work or errors out, try reducing the chunk length
"""

import argparse
import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))


import logging

logging.basicConfig(level=logging.WARNING, filename="vid2cleantext_transcriber.log")

import math
import shutil
import time

import librosa
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings

#  filter out warnings that pretend transfer learning does not exist
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="initializing BertModel")
transformers.utils.logging.set_verbosity(40)

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
    get_av_fmts,
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


def save_transc_results(
    out_dir,
    vid_name: str,
    ttext: str,
    mdata: pd.DataFrame,
):
    """
    save_transc_results - save the transcribed text to a file and a metadata file

    Parameters
    ----------
    out_dir : str, path to the output directory
    vid_name : str, name of the video file
    ttext : str, the transcribed text
    mdata : pd.DataFrame, the metadata for the video file
    """
    storage_locs = setup_out_dirs(out_dir)  # create and get output folders
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")
    header = f"{trim_fname(vid_name)}_vid2txt_{get_timestamp()}"  # create header for output file
    # save the text
    with open(
        join(out_p_tscript, f"{header}_full.txt"),
        "w",
        encoding="utf-8",
        errors="ignore",
    ) as fo:
        fo.writelines(ttext)

    mdata.to_csv(join(out_p_metadata, f"{header}_metadata.csv"), index=False)

    # if verbose:
    print(
        "Saved transcript and metadata to {} and {}".format(
            out_p_tscript, out_p_metadata
        )
    )


def transcribe_video_wav2vec(
    ts_model,
    ts_tokenizer,
    src_dir,
    clip_name: str,
    chunk_dur: int,
    verbose=False,
    temp_dir: str = "audio_chunks",
):
    """
    transcribe_video_wav2vec - transcribes a video clip using the wav2vec2 model. Note that results will be saved to the output directory, src_dir

    Parameters
    ----------
    ts_model : torch.nn.Module, the transformer model that was loaded (must be a wav2vec2 model)
    ts_tokenizer : transformers.AutoTokenizer, the tokenizer that was loaded (must be a wav2vec2 tokenizer)
    directory : str, path to the directory containing the video file
    vid_clip_name : str, name of the video clip
    chunk_dur : int, duration of audio chunks (in seconds) that the transformer model will be fed
    verbose : bool, optional
    temp_dir : str, optional, the name of the temporary directory to store the audio chunks

    Returns
    -------
    transc_results : dict, the transcribed text and metadata

    """

    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    # create audio chunk folder
    ac_storedir = join(src_dir, temp_dir)
    create_folder(ac_storedir)
    chunk_directory = prep_transc_src(
        _vid2beconv=clip_name,
        in_dir=src_dir,
        len_chunks=chunk_dur,
        out_dir=ac_storedir,
    )
    torch_validate_cuda()
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")
    for i, audio_chunk in enumerate(chunk_directory):

        if (i % GPU_update_incr == 0) and (GPU_update_incr != 0):
            # provide update on GPU usage
            check_runhardware()
        audio_input, clip_sr = librosa.load(
            join(ac_storedir, audio_chunk), sr=16000
        )  # 16000 is the sampling rate of the wav2vec model
        # convert audio to tensor
        inputs = ts_tokenizer(audio_input, return_tensors="pt", padding="longest")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # set device
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            # run the model
            logits = ts_model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)  # get the predicted ids by argmax
        this_transc = ts_tokenizer.batch_decode(predicted_ids)
        this_transc = (
            "".join(this_transc) if isinstance(this_transc, list) else this_transc
        )
        # double-check if "" should be joined on  or " "
        full_transc.append(f"{this_transc}\n")
        # empty memory so you don't overload the GPU
        del input_values
        del logits
        del predicted_ids
        torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()
    if verbose:
        print(f"Finished transcribing {clip_name} @ {get_timestamp()}")

    md_df = create_metadata_df()  # makes a blank df with column names
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
        len(chunk_directory),
        chunk_dur,
        (len(chunk_directory) * chunk_dur) / 60,  # minutes, the duration of the video
        get_timestamp(),
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(
        copy=False,
    )
    save_transc_results(
        out_dir=src_dir,
        vid_name=clip_name,
        ttext=full_text,
        mdata=md_df,
    )  # save the results here

    shutil.rmtree(ac_storedir, ignore_errors=True)  # remove audio chunks folder
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(
            "finished transcription of {vid_clip_name} base folder on {get_timestamp()}"
        )

    return transc_res


def postprocess_transc(tscript_dir, mdata_dir, merge_files=False, verbose=False):
    """
    postprocess_transc - postprocess the transcribed text by consolidating the text and metadata, and spell checking + sentence splitting

    Parameters
    ----------
    tscript_dir : str, path to the directory containing the transcribed text files
    mdata_dir : str, path to the directory containing the metadata files
    merge_files : bool, optional, by default False, if True, create a new file that contains all text and metadata merged together
    verbose : bool, optional
    """
    if verbose:
        print("Starting to postprocess transcription @ {}".format(get_timestamp()))

    if merge_files:
        digest_txt_directory(tscript_dir, iden=f"orig_transc_{get_timestamp()}")
        digest_txt_directory(
            mdata_dir,
            iden=f"meta_{get_timestamp()}",
            make_folder=True,
        )

    # add spelling correction to output transcriptions
    # reload the metadata and text
    txt_files = find_ext_local(
        tscript_dir, req_ext=".txt", verbose=verbose, full_path=False
    )
    # Spellcorrect Pipeline
    kw_all_vids = pd.DataFrame()

    for this_transc in tqdm(
        txt_files,
        total=len(txt_files),
        desc="SC_pipeline - transcribed audio",
    ):
        PL_out = spellcorrect_pipeline(
            out_p_tscript, this_transc, verbose=False, ns_checker=checker
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
        default=20,  # may need to be adjusted based on hardware and model used
        type=int,
        help="Duration of audio chunks (in seconds) that the transformer model will be fed",
    )

    return parser


# TODO: change to pathlib from os.path

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
    print(f"Loading models @ {get_timestamp(True)} - may take a while...")
    print("If RT seems excessive, try --verbose flag or checking logfile")
    wav2vec2_model = (
        "facebook/wav2vec2-large-960h-lv60-self" if model_arg is None else model_arg
    )
    if is_verbose:
        print("Loading model: {}".format(wav2vec2_model))
    tokenizer = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)

    # load the spellchecker models. suppress outputs as there are way too many
    orig_out = sys.__stdout__
    sys.stdout = NullIO()
    checker = init_neuspell()
    sym_spell = init_symspell()
    sys.stdout = orig_out  # return to default of print-to-console

    approved_files = []
    for ext in get_av_fmts():  # now include audio formats and video formats
        approved_files.extend(find_ext_local(directory, req_ext=ext, full_path=False))

    print(f"\nFound {len(approved_files)} video files in {directory}")

    storage_locs = setup_out_dirs(directory)  # create and get output folders

    for filename in tqdm(
        approved_files,
        total=len(approved_files),
        desc="transcribing vids",
    ):
        # transcribe video and get results
        t_results = transcribe_video_wav2vec(
            ts_model=model,
            ts_tokenizer=tokenizer,
            src_dir=directory,
            clip_name=filename,
            chunk_dur=chunk_length,
        )

        if move_comp:
            move2completed(directory, filename=filename)

    # postprocess the transcriptions
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")
    postprocess_transc(
        tscript_dir=out_p_tscript,
        mdata_dir=out_p_metadata,
        merge_files=False,
        verbose=is_verbose,
    )

    print(
        f"\n\nFinished at: {get_timestamp()} taking a total of {(time.perf_counter() - st)/60} mins"
    )
    print(
        f"The relevant files for this run are in: \n {out_p_tscript} \n and {out_p_metadata}"
    )
