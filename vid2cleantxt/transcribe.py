#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe.py - transcribe a video file using a pretrained ASR model such as wav2vec2 or whisper

Usage:
    vid2cleantxt.py --video <video_file> --model <model_id> [--out <out_dir>] [--verbose] [--debug] [--log <log_file>]

Tips for runtime:

- use the default model to start. If issues try "facebook/wav2vec2-base-960h"
- if model fails to work or errors out, try reducing the chunk length with --chunk-length <int>
"""

__author__ = "Peter Szemraj"

import argparse
import gc
import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))


import logging

logging.basicConfig(level=logging.INFO, filename="LOGFILE_vid2cleantxt_transcriber.log", format="%(asctime)s %(message)s")

import argparse
import math
import shutil
import time
import warnings

import librosa
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import (
    HubertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WavLMForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

#  filter out warnings that pretend transfer learning does not exist
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="initializing BertModel")
transformers.utils.logging.set_verbosity(40)

from vid2cleantxt.v2ct_utils import load_spacy_models

load_spacy_models()

from vid2cleantxt.audio2text_functions import (
    corr,
    create_metadata_df,
    get_av_fmts,
    init_neuspell,
    init_symspell,
    prep_transc_pydub,
    quick_keys,
    setup_out_dirs,
    spellcorrect_pipeline,
    trim_fname,
)
from vid2cleantxt.v2ct_utils import (
    NullIO,
    check_runhardware,
    create_folder,
    digest_txt_directory,
    find_ext_local,
    get_timestamp,
    load_spacy_models,
    move2completed,
    torch_validate_cuda,
)


def load_whisper_modules(
    hf_id: str, language: str = "en", task: str = "transcribe", chunk_length: int = 30
):
    """
    laod_whisper_modules - load the whisper modules from huggingface

    :param str hf_id: the id of the model to load on huggingface, for example: "openai/whisper-base.en" or "openai/whisper-medium"
    :param str language: the language of the model, for example "en" or "de"
    :param str task: the task of the model, for example "transcribe" or "translate"
    :param int chunk_length: the length of the chunks to transcribe, in seconds
    :return processor, model: the processor and model objects
    """

    processor = WhisperProcessor.from_pretrained(hf_id)
    model = WhisperForConditionalGeneration.from_pretrained(hf_id)

    processor.feature_extractor.chunk_length = chunk_length
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task=task
    )

    return processor, model


def load_wav2vec2_modules(hf_id: str):
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
    logging.info(f"Loaded model {hf_id} from huggingface")
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
    if isinstance(model_obj, HubertForCTC):
        logging.info("HubertForCTC is not a wav2vec2 model so not checking size")
        return False
    elif not isinstance(model_obj, Wav2Vec2ForCTC):
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
    _t_out = join(out_p_tscript, f"{header}_full.txt")
    # save the text
    with open(
        _t_out,
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

    logging.info(
        f"Saved transcript and metadata to: {out_p_tscript} and {out_p_metadata}"
    )


def transcribe_video_whisper(
    model,
    processor,
    clip_directory,
    clip_name: str,
    chunk_dur: int = 30,
    chunk_max_new_tokens=512,
    temp_dir: str = "audio_chunks",
    manually_clear_cuda_cache=False,
    print_memory_usage=False,
    verbose=False,
) -> dict:
    """
    transcribe_video_whisper - transcribe a video file using the whisper model

    :param model: the model object
    :param processor: the processor object
    :param clip_directory: the directory of the video file
    :param str clip_name: the name of the video file
    :param int chunk_dur: the duration of each chunk in seconds, default 30
    :param int chunk_max_new_tokens: max new tokens generated per chunk, default 512 (arbitrary upper bound)
    :param str temp_dir: the directory to store the audio chunks in. default "audio_chunks"
    :param bool manually_clear_cuda_cache: whether to manually clear the cuda cache after each chunk. default False
    :param bool print_memory_usage: whether to print the memory usage at set interval while transcribing. default False
    :param bool verbose: whether to print the transcribed text locations to the console. default False
    :return dict: a dictionary containing the transcribed text, the metadata

    """
    logging.info(f"Starting to transcribe {clip_name}")
    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    ac_storedir = join(clip_directory, temp_dir)
    create_folder(ac_storedir)
    chunk_directory = prep_transc_pydub(
        clip_name, clip_directory, ac_storedir, chunk_dur, verbose=verbose
    )
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"transcribing on {device}")
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )
    model = model.to(device)
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")
    for i, audio_chunk in enumerate(chunk_directory):

        if (i % GPU_update_incr == 0) and (GPU_update_incr != 0) and print_memory_usage:
            check_runhardware()  # utilization check
            gc.collect()

        try:
            audio_input, clip_sr = librosa.load(
                join(ac_storedir, audio_chunk), sr=16000
            )  # load the audio chunk @ 16kHz

            input_features = processor(
                audio_input, truncation=True, padding="max_length", return_tensors="pt"
            ).input_features  # audio to tensor
            input_features = input_features.to(device)  # send to device
            predicted_ids = model.generate(
                input_features, max_new_tokens=chunk_max_new_tokens
            )
            this_transc = processor.batch_decode(
                predicted_ids,
                max_length=chunk_max_new_tokens,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )[
                0
            ]  # decode the tensor to text
        except Exception as e:
            logging.warning(
                f"Error transcribing chunk {i} in {clip_name}"
            )
            logging.warning(e)
            warnings.warn(f"Error transcribing chunk {i} - see log for details")
            this_transc = ""
        this_transc = (
            "".join(this_transc) if isinstance(this_transc, list) else this_transc
        )
        full_transc.append(f"{this_transc}\n")

        if device == "cuda" and manually_clear_cuda_cache:
            torch.cuda.empty_cache()

        pbar.update()

    pbar.close()
    logging.info("completed transcription")

    md_df = create_metadata_df()  # blank df with column names
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
        len(chunk_directory),
        chunk_dur,
        (len(chunk_directory) * chunk_dur) / 60,  # duration in mins
        get_timestamp(),
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(
        copy=False,
    )
    save_transc_results(
        out_dir=clip_directory,
        vid_name=clip_name,
        ttext=full_text,
        mdata=md_df,
        verbose=verbose,
    )

    shutil.rmtree(ac_storedir, ignore_errors=True)
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(f"finished transcription of {clip_name} - {get_timestamp()}")
    logging.info(f"finished transcription of {clip_name} - {get_timestamp()}")
    return transc_res


def transcribe_video_wav2vec(
    model,
    processor,
    clip_directory,
    clip_name: str,
    chunk_dur: int = 15,
    temp_dir: str = "audio_chunks",
    manually_clear_cuda_cache=False,
    print_memory_usage=False,
    verbose=False,
) -> dict:
    """
    transcribe_video_wav2vec - transcribe a video file using the wav2vec model

    :param model: the model object
    :param processor: the processor object
    :param clip_directory: the directory of the video file
    :param str clip_name: the name of the video file
    :param int chunk_dur: the duration of each chunk in seconds, default 15
    :param str temp_dir: the directory to store the audio chunks in. default "audio_chunks"
    :param bool manually_clear_cuda_cache: whether to manually clear the cuda cache after each chunk. default False
    :param bool print_memory_usage: whether to print the memory usage at set interval while transcribing. default False
    :param bool verbose: whether to print the transcribed text locations to the console. default False
    :return dict: a dictionary containing the transcribed text, the metadata
    """
    logging.info(f"Starting to transcribe {clip_name}")
    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    ac_storedir = join(clip_directory, temp_dir)
    create_folder(ac_storedir)
    use_attn = wav2vec2_islarge(
        model
    )  # if they pass in a large model, use attention masking

    chunk_directory = prep_transc_pydub(
        clip_name, clip_directory, ac_storedir, chunk_dur, verbose=verbose
    )  # split the video into chunks
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # set device
    logging.info(f"transcribing on {device}")
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )
    model = model.to(device)
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")
    for i, audio_chunk in enumerate(chunk_directory):

        # note that large-960h-lv60 has an attention mask of length of the input sequence, the base model does not
        if (i % GPU_update_incr == 0) and (GPU_update_incr != 0) and print_memory_usage:
            check_runhardware()  # check utilization
            gc.collect()
        try:
            audio_input, clip_sr = librosa.load(
                join(ac_storedir, audio_chunk), sr=16000
            )  # load the audio chunk @ 16kHz (wav2vec2 expects 16kHz)

            inputs = processor(
                audio_input, return_tensors="pt", padding="longest"
            )  # audio to tensor
            input_values = inputs.input_values.to(device)
            attention_mask = (
                inputs.attention_mask.to(device) if use_attn else None
            )  # set attention mask if using large model

            with torch.no_grad():
                if use_attn:
                    logits = model(input_values, attention_mask=attention_mask).logits
                else:
                    logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)  # get the predicted ids
            this_transc = processor.batch_decode(predicted_ids)
            this_transc = (
                "".join(this_transc) if isinstance(this_transc, list) else this_transc
            )
        except Exception as e:
            logging.warning(
                f"Error transcribing chunk {i} in {clip_name}"
            )
            logging.warning(e)
            warnings.warn(f"Error transcribing chunk {i} - see log for details   ")
            this_transc = ""

        full_transc.append(f"{this_transc}\n")

        del input_values
        del logits
        del predicted_ids
        if device == "cuda" and manually_clear_cuda_cache:
            torch.cuda.empty_cache()

        pbar.update()

    pbar.close()
    logging.info("completed transcription")

    md_df = create_metadata_df()  # makes a blank df with column names
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
        len(chunk_directory),
        chunk_dur,
        (len(chunk_directory) * chunk_dur) / 60,
        get_timestamp(),
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(
        copy=False,
    )
    save_transc_results(
        out_dir=clip_directory,
        vid_name=clip_name,
        ttext=full_text,
        mdata=md_df,
        verbose=verbose,
    )

    shutil.rmtree(ac_storedir, ignore_errors=True)
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(f"finished transcription of {clip_name} - {get_timestamp()}")
    logging.info(f"finished transcription of {clip_name} - {get_timestamp()}")
    return transc_res


def postprocess_transc(
    tscript_dir,
    mdata_dir,
    merge_files=False,
    linebyline=True,
    verbose=False,
    spell_correct_method: str = "symspell",
    checker=None,
) -> None:
    """
    postprocess_transc - postprocess the transcribed text by consolidating the text and metadata, and spell checking + sentence splitting

    Parameters
    ----------
    tscript_dir : str, path to the directory containing the transcribed text files
    mdata_dir : str, path to the directory containing the metadata files
    merge_files : bool, optional, by default False, if True, create a new file that contains all text and metadata merged together
    linebyline : bool, optional, by default True, if True, split the text into sentences
    spell_correct_method : str, optional, by default "symspell", the method to use for spell checking. Options are "symspell", "neuspell"
    checker : spellchecker.SpellChecker, optional, by default None, the spell checker object to use for spell checking. If None, it will be created.
    verbose : bool, optional

    Returns
    -------
    str, filepath to the "complete" output directory
    """
    logging.info(
        f"Starting postprocessing of transcribed text with params {locals()}"
    )
    if checker is None:
        checker = (
            init_neuspell()
            if spell_correct_method.lower() == "neuspell"
            else init_symspell()
        )
    if verbose:
        print("Starting to postprocess transcription @ {}".format(get_timestamp()))

    if merge_files:
        digest_txt_directory(tscript_dir, iden=f"orig_transc_{get_timestamp()}")
        digest_txt_directory(
            mdata_dir,
            iden=f"meta_{get_timestamp()}",
            make_folder=True,
        )

    txt_files = find_ext_local(
        tscript_dir, req_ext=".txt", verbose=verbose, full_path=False
    )  # load the text files

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
            method=spell_correct_method,
            spell_checker=checker,
            linebyline=linebyline,
        )

        processed_dir = PL_out.get(
            "spell_corrected_dir"
        )  # directory where the corrected text is saved
        kw_name = PL_out.get("sc_filename")

        qk_df = quick_keys(
            filepath=processed_dir,
            filename=kw_name,
            num_kw=25,
            max_ngrams=3,
            save_db=False,
            verbose=verbose,
        )  # keyword extraction

        kw_all_vids = pd.concat([kw_all_vids, qk_df], axis=1)

    # save overall transcription file
    kwdb_fname = f"YAKE - all keywords for run at {get_timestamp()}.csv"
    kw_all_vids.to_csv(
        join(tscript_dir, kwdb_fname),
        index=True,
    )

    return PL_out["SBD_dir"]


def transcribe_dir(
    input_dir: str,
    chunk_length: int = 30,
    model_id: str = "openai/whisper-base.en",
    basic_spelling=False,
    move_comp=False,
    join_text=False,
    print_memory_usage=False,
    verbose=False,
):
    """
    transcribe_dir - transcribe all videos in a directory

    :param str input_src: the path to the directory containing the videos to transcribe
    :param int chunk_length: the length of the chunks to split the audio into, in seconds. Default is 30 seconds
    :param str model_id: the model id to use for the transcription. Default is openai/whisper-base.en
    :param bool basic_spelling: if True, use basic spelling correction instead of neural spell correction
    :param bool move_comp: if True, move the completed files to a new folder
    :param bool join_text: if True, join all lines of text into one long string
    :param bool print_memory_usage: if True, print the memory usage of the system during the transcription
    :param bool verbose: if True, print out more information

    -------
    :return str, str: the path to the directory of transcribed text files, and the path to the directory of metadata files
    """
    st = time.perf_counter()

    directory = os.path.abspath(input_dir)
    linebyline = not join_text
    base_spelling = basic_spelling
    logging.info(f"Starting transcription pipeline @ {get_timestamp(True)}" + "\n")
    print(f"\nLoading models @ {get_timestamp(True)} - may take some time...")
    print("if RT seems excessive, try --verbose flag or checking logfile")

    _is_whisper = "whisper" in model_id.lower()

    if _is_whisper:
        logging.info("whisper model detected, using special settings")
        if chunk_length != 30:
            warnings.warn(
                f"you have set chunk_length to {chunk_length}, but whisper models default to 30s chunks. strange things may happen"
            )

    processor, model = (
        load_whisper_modules(model_id)
        if _is_whisper
        else load_wav2vec2_modules(model_id)
    )
    # load the spellchecker models. suppressing outputs
    orig_out = sys.__stdout__
    sys.stdout = NullIO()
    if base_spelling:
        checker = init_symspell()
    else:
        try:
            checker = init_neuspell()

        except Exception as e:
            print(
                "Failed loading NeuSpell spellchecker, reverting to basic spellchecker"
            )
            logging.warning(
                f"Failed loading NeuSpell spellchecker, reverting to basic spellchecker"
            )
            logging.warning(f"{e}")
            base_spelling = True
            checker = init_symspell()

    sys.stdout = orig_out  # return to default of print-to-console

    approved_files = []
    for ext in get_av_fmts():  # load vid2cleantxt inputs
        approved_files.extend(find_ext_local(directory, req_ext=ext, full_path=False))

    print(f"\nFound {len(approved_files)} audio or video files in {directory}")

    storage_locs = setup_out_dirs(directory)  # create and get output folders
    for filename in tqdm(
        approved_files,
        total=len(approved_files),
        desc="transcribing...",
    ):
        t_results = (
            transcribe_video_whisper(
                model=model,
                processor=processor,
                clip_directory=directory,
                clip_name=filename,
                chunk_dur=chunk_length,
                print_memory_usage=print_memory_usage,
                verbose=verbose,
            )
            if _is_whisper
            else transcribe_video_wav2vec(
                model=model,
                processor=processor,
                clip_directory=directory,
                clip_name=filename,
                chunk_dur=chunk_length,
                print_memory_usage=print_memory_usage,
                verbose=verbose,
            )
        )

        if move_comp:
            move2completed(directory, filename=filename)  # move src to completed folder

    # postprocess the transcriptions
    out_p_tscript = storage_locs["t_out"]
    out_p_metadata = storage_locs["m_out"]
    processed_dir = postprocess_transc(
        tscript_dir=out_p_tscript,
        mdata_dir=out_p_metadata,
        merge_files=False,
        verbose=verbose,
        linebyline=linebyline,
        spell_correct_method="symspell" if base_spelling else "neuspell",
        checker=checker,
    )

    logging.info(f"Finished transcription pipeline @ {get_timestamp(True)}" + "\n")
    logging.info(f"Total time: {round((time.perf_counter() - st)/60, 3)} mins")

    return processed_dir, out_p_metadata


def get_parser():
    """
    get_parser - a helper function for the argparse module
    Returns: argparse.ArgumentParser object
    """

    parser = argparse.ArgumentParser(
        description="Transcribe a directory of videos using transformers"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="path to directory containing video files to be transcribed",
    )

    # parser.add_argument(
    #     "-o",
    #     "--output-dir",
    #     default=None,
    #     required=False,
    #     help="folder - where to save the output files. If not specified, will save to the input-dir",
    # )
    parser.add_argument(
        "-move",
        "--move-input-vids",
        required=False,
        default=False,
        action="store_true",
        help="if specified, will move files that finished transcription to the completed folder",
        # use case here is if there are so many files that run into CUDA memory issues resulting in a crash
    )

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        default="openai/whisper-base.en",
        help="huggingface ASR model name. try 'facebook/wav2vec2-base-960h' if issues running default.",
    )

    parser.add_argument(
        "-cl",
        "--chunk-length",
        required=False,
        default=30,
        type=int,
        help="Duration of .wav chunks (in seconds) that the transformer model will be fed. decrease if you run into memory issues",
    )

    parser.add_argument(
        "--join-text",
        required=False,
        default=False,
        action="store_true",
        help="Save the transcribed text as a single line of text instead of one line per sentence",
    )

    parser.add_argument(
        "--basic-spelling",
        required=False,
        default=False,
        action="store_true",
        help="Use the basic spelling correction pipeline with symSpell",
    )
    parser.add_argument(
        "--print-memory-usage",
        required=False,
        default=False,
        action="store_true",
        help="Print memory usage updates during transcription",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="print out more information",
    )

    return parser


# TODO: change to pathlib from os.path

if __name__ == "__main__":

    # parse the command line arguments
    args = get_parser().parse_args()
    input_src = str(args.input_dir)
    # TODO: add output directory from user arg
    move_comp = args.move_input_vids
    chunk_length = int(args.chunk_length)
    model_id = str(args.model)
    join_text = args.join_text
    basic_spelling = args.basic_spelling
    print_memory_usage = args.print_memory_usage
    is_verbose = args.verbose

    output_text, output_metadata = transcribe_dir(
        input_dir=input_src,
        chunk_length=chunk_length,
        model_id=model_id,
        move_comp=move_comp,
        join_text=join_text,
        basic_spelling=basic_spelling,
        print_memory_usage=print_memory_usage,
        verbose=is_verbose,
    )

    print(
        f"Complete. Relevant files for run are in: \n{output_text} \n{output_metadata}"
    )
