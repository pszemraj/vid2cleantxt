#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vid2cleantxt by Peter Szemraj

Pipeline for Zero-shot transcription of a lecture video file to text using facebook's wav2vec2 model
this is the primary pipeline for the project

You can access the arguments for this script by running the following command:
    *\vid2cleantxt\transcribe.py -h (windows)
    */vid2cleantxt/transcribe.py -h (everything else + Windows bash)

Tips for runtime:

- use the default "facebook/wav2vec2-base-960h" to start out with
- if model fails to work or errors out, try reducing the chunk length with --chunk-length <int>
"""

import argparse
import os
import gc
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))


import logging

logging.basicConfig(level=logging.INFO, filename="LOGFILE_vid2cleantxt_transcriber.log")

import math
import shutil
import time

import librosa
import pandas as pd
import argparse
import torch
from tqdm.auto import tqdm
import transformers
from transformers import (
    HubertForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WavLMForCTC,
)
import warnings

#  filter out warnings that pretend transfer learning does not exist
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="initializing BertModel")
transformers.utils.logging.set_verbosity(40)

from vid2cleantxt.audio2text_functions import (
    trim_fname,
    corr,
    create_metadata_df,
    init_neuspell,
    init_symspell,
    quick_keys,
    spellcorrect_pipeline,
    setup_out_dirs,
    get_av_fmts,
    prep_transc_pydub,
)
from vid2cleantxt.v2ct_utils import (
    check_runhardware,
    create_folder,
    digest_txt_directory,
    find_ext_local,
    move2completed,
    NullIO,
    torch_validate_cuda,
    get_timestamp,
)


def load_transcription_objects(hf_id: str):
    """
    load_transcription_objects - load the transcription objects from huggingface

    Parameters
    ----------
    hf_id : str, the id of the model to load on huggingface, for example: "facebook/wav2vec2-base-960h" or "facebook/hubert-large-ls960-ft"

    Returns
    -------
    tokenizer : transformers.Wav2Vec2Processor, the tokenizer object
    model: transformers.Wav2Vec2ForCTC, the model object. For specialised models, this is a specialised object such as HubertForCTC
    """

    tokenizer = Wav2Vec2Processor.from_pretrained(
        hf_id
    )  # use wav2vec2processor for tokenization always
    if "wavlm" in hf_id.lower():
        # for example --model "patrickvonplaten/wavlm-libri-clean-100h-large"
        print(f"Loading wavlm model - {hf_id}")
        model = WavLMForCTC.from_pretrained(hf_id)
    elif "hubert" in hf_id.lower():
        print(f"Loading hubert model - {hf_id}")
        model = HubertForCTC.from_pretrained(
            hf_id
        )  # for example --model "facebook/hubert-large-ls960-ft"
    else:
        # for example --model "facebook/wav2vec2-large-960h-lv60-self"
        print(f"Loading wav2vec2 model - {hf_id}")
        model = Wav2Vec2ForCTC.from_pretrained(hf_id)
    return tokenizer, model


def wav2vec2_islarge(model_obj):
    """
    wav2vec2_check_size - compares the size of the passed model object, and whether
    it is in fact a wav2vec2 model. this is because the large model is a special case and
    uses an attention mechanism that is not compatible with the rest of the models

    https://huggingface.co/facebook/wav2vec2-base-960h

    Parameters
    ----------
    model_obj : transformers.Wav2Vec2ForCTC, the model object to check

    Returns
    -------
    is_large, whether the model is the large wav2vec2 model or not
    """
    approx_sizes = {
        "base": 94396320,
        "large": 315471520,  # recorded by  loading the model in known environment
    }
    if not isinstance(model_obj, Wav2Vec2ForCTC):
        warnings.warn(
            message="Model is not a wav2vec2 model - this function is for wav2vec2 models only",
            category=None,
            stacklevel=1,
        )
        return (
            False  # not a wav2vec2 model - return false so it is handled per standard
        )

    np_proposed = model_obj.num_parameters()

    dist_from_base = abs(np_proposed - approx_sizes.get("base"))
    dist_from_large = abs(np_proposed - approx_sizes.get("large"))
    return True if dist_from_large < dist_from_base else False


def save_transc_results(
    out_dir,
    vid_name: str,
    ttext: str,
    mdata: pd.DataFrame,
    verbose=False,
):
    """
    save_transc_results - save the transcribed text to a file and a metadata file

    Parameters
    ----------
    out_dir : str, path to the output directory
    vid_name : str, name of the video file
    ttext : str, the transcribed text
    mdata : pd.DataFrame, the metadata for the video file
    verbose : bool, whether to print the transcribed text locations to the console

    Returns
    -------
    None
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

    if verbose:
        print(
            f"Saved transcript and metadata to: {out_p_tscript} \n and {out_p_metadata}"
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
    use_attn = wav2vec2_islarge(
        ts_model
    )  # if they pass in a large model, use attention masking
    # get the audio chunks
    chunk_directory = prep_transc_pydub(
        clip_name, src_dir, ac_storedir, chunk_dur, verbose=verbose
    )
    torch_validate_cuda()
    gc.collect()  # free up memory
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # set device
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")
    for i, audio_chunk in enumerate(chunk_directory):

        # note that large-960h-lv60 has an attention mask of length of the input sequence, the base model does not
        if (i % GPU_update_incr == 0) and (GPU_update_incr != 0):
            # provide update on GPU usage
            check_runhardware()
            gc.collect()
        audio_input, clip_sr = librosa.load(
            join(ac_storedir, audio_chunk), sr=16000
        )  # 16000 is the sampling rate of the wav2vec model
        # convert audio to tensor
        inputs = ts_tokenizer(audio_input, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to(device)
        attention_mask = (
            inputs.attention_mask.to(device) if use_attn else None
        )  # if using attention masking, set it. for large wav2vec2 model.
        ts_model = ts_model.to(device)
        # run the model
        with torch.no_grad():
            if use_attn:
                logits = ts_model(input_values, attention_mask=attention_mask).logits
            else:
                logits = ts_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)  # get the predicted ids by argmax
        this_transc = ts_tokenizer.batch_decode(predicted_ids)
        this_transc = (
            "".join(this_transc) if isinstance(this_transc, list) else this_transc
        )
        # double-check if "" should be joined on  or " "
        full_transc.append(f"{this_transc}\n")
        pbar.update(1)
        # empty memory so you don't overload the GPU
        del input_values
        del logits
        del predicted_ids
        if device == "cuda:0":
            torch.cuda.empty_cache()  # empty memory on GPU

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
        verbose=verbose,
    )  # save the results here

    shutil.rmtree(ac_storedir, ignore_errors=True)  # remove audio chunks folder
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(f"finished transcription of {clip_name} base folder on {get_timestamp()}")

    return transc_res


def postprocess_transc(
    checker, tscript_dir, mdata_dir, merge_files=False, linebyline=True, verbose=False
):
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
            tscript_dir,
            this_transc,
            verbose=False,
            ns_checker=checker,
            linebyline=linebyline,
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
            verbose=verbose,
        )

        kw_all_vids = pd.concat([kw_all_vids, qk_df], axis=1)

    # save overall transcription file
    kwdb_fname = f"YAKE - all keywords for run at {get_timestamp()}.csv"
    kw_all_vids.to_csv(
        join(tscript_dir, kwdb_fname),
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
        "-i",
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
        help="if specified, will move files that finished transcription to the completed folder",
        # use case here is if there are so many files that run into CUDA memory issues resulting in a crash
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="print out more information",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        default=None,
        help="huggingface wav2vec2 model name, ex 'facebook/wav2vec2-base-960h'",
        # "facebook/wav2vec2-large-960h-lv60-self" is one of the best models but taxing on the GPU/CPU
        # for wavLM, "patrickvonplaten/wavlm-libri-clean-100h-large" or others
    )

    parser.add_argument(
        "-cl",
        "--chunk-length",
        required=False,
        default=15,  # pass lower value if running out of memory / GPU memory
        type=int,
        help="Duration of .wav chunks (in seconds) that the transformer model will be fed",
    )

    parser.add_argument(
        "--join-text",
        required=False,
        default=False,
        action="store_true",
        help="Save the transcribed text as a single line of text instead of one line per sentence",
    )

    return parser

def transcribe_dir(input_src, is_verbose=False, move_comp=False, chunk_length=15, model_arg=None, join_text=False):
    st = time.perf_counter()
    
    directory = os.path.abspath(input_src)
    linebyline = not join_text

    print(f"\nLoading models @ {get_timestamp(True)} - may take some time...")
    print("if RT seems excessive, try --verbose flag or checking logfile")
    # load the model facebook/hubert-large-ls960-ft
    wav_model = (
        "facebook/hubert-large-ls960-ft" if model_arg is None else model_arg
    )  # try "facebook/wav2vec2-base-960h" if crashes
    tokenizer, model = load_transcription_objects(wav_model)

    # load the spellchecker models. suppress outputs as there are way too many
    orig_out = sys.__stdout__
    sys.stdout = NullIO()
    try: 
        checker = init_neuspell()
    except Exception as e:
        print(f"issue using Neuspell, {e}, using SymSpell")
        checker = init_symspell()
    sys.stdout = orig_out  # return to default of print-to-console

    # load vid2cleantxt inputs
    approved_files = []
    for ext in get_av_fmts():  # now include audio formats and video formats
        approved_files.extend(find_ext_local(directory, req_ext=ext, full_path=False))

    print(f"\nFound {len(approved_files)} audio or video files in {directory}")

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
            move2completed(directory, filename=filename)  # move src to completed folder

    # postprocess the transcriptions
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")
    postprocess_transc(
        checker,
        tscript_dir=out_p_tscript,
        mdata_dir=out_p_metadata,
        merge_files=False,
        verbose=is_verbose,
        linebyline=linebyline,
    )

    print(
        f"\n\nFinished at: {get_timestamp()}. Total RT was {round((time.perf_counter() - st)/60, 3)} mins"
    )
    print(
        f"relevant files for run are in: \n{out_p_tscript} \n and: \n{out_p_metadata}"
    )


# TODO: change to pathlib from os.path

if __name__ == "__main__":

    # parse the command line arguments
    args = get_parser().parse_args()
    input_src = str(args.input_dir)
    # TODO: add output directory from user arg
    is_verbose = args.verbose
    move_comp = args.move_input_vids
    chunk_length = int(args.chunk_length)
    model_arg = args.model
    join_text = args.join_text

    transcribe_dir(input_src, is_verbose, move_comp, chunk_length, model_arg, join_text)
