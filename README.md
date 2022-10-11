# vid2cleantxt

![vid2cleantext simple](https://user-images.githubusercontent.com/74869040/131500291-ed0a9d7f-8be7-4f4b-9acf-c360cfd46f1f.png)

[Jump to Quickstart](#quickstart)

**vid2cleantxt**: a [transformers-based](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) pipeline for turning heavily speech-based video files into clean, readable text from the audio. Robust speech transcription is now possible like never before with [OpenAI's whisper model](https://openai.com/blog/whisper/).

TL;DR check out [this Colab notebook](https://colab.research.google.com/gist/pszemraj/9678129fe0b552e114e3576606446dee/vid2cleantxt-minimal-example.ipynb) for a transcription and keyword extraction of a speech by John F. Kennedy by simply running all cells.

* * *

**Table of Contents**

<!-- TOC -->

-   [Motivation](#motivation)
-   [Overview](#overview)
    -   [Example Output](#example-output)
    -   [Pipeline Intro](#pipeline-intro)
-   [Quickstart](#quickstart)
    -   [Installation](#installation)
        -   [As a Python package](#as-a-python-package)
        -   [Install from source](#install-from-source)
        -   [install details & gotchas](#install-details--gotchas)
    -   [example usage](#example-usage)
-   [Notebooks on Colab](#notebooks-on-colab)
-   [Details & Application](#details--application)
    -   [How long does this take to run?](#how-long-does-this-take-to-run)
    -   [Now I have a bunch of long text files. How are these useful?](#now-i-have-a-bunch-of-long-text-files-how-are-these-useful)
        -   [Visualization and Analysis](#visualization-and-analysis)
        -   [Text Extraction / Manipulation](#text-extraction--manipulation)
    -   [Text Summarization](#text-summarization)
        -   [TextHero example use case](#texthero-example-use-case)
-   [ScatterText example use case](#scattertext-example-use-case)
-   [Design Choices & Troubleshooting](#design-choices--troubleshooting)
    -   [What python package dependencies does this repo have?](#what-python-package-dependencies-does-this-repo-have)
    -   [My computer crashes once it starts running the wav2vec2 model](#my-computer-crashes-once-it-starts-running-the-wav2vec2-model)
    -   [The transcription is not perfect, and therefore I am mad](#the-transcription-is-not-perfect-and-therefore-i-am-mad)
    -   [How can I improve the performance of the model from a word-error-rate perspective?](#how-can-i-improve-the-performance-of-the-model-from-a-word-error-rate-perspective)
    -   [Why use transformer models instead of SpeechRecognition or other transcription methods?](#why-use-transformer-models-instead-of-speechrecognition-or-other-transcription-methods)
    -   [Errors](#errors)
-   [Examples](#examples)
-   [Future Work, Collaboration, & Citations](#future-work-collaboration--citations)
    -   [Project Updates](#project-updates)
    -   [Future Work](#future-work)
    -   [I've found x repo / script / concept that I think you should incorporate or collaborate with the author](#ive-found-x-repo--script--concept-that-i-think-you-should-incorporate-or-collaborate-with-the-author)
    -   [Citations](#citations)
        -   [Video Citations](#video-citations)

<!-- /TOC -->

* * *

## Motivation

Video, specifically audio, is inefficient in conveying dense or technical information. The viewer has to sit through the whole thing, while only part of the video may be relevant to them. If you don't understand a statement or concept, you must search through the video or re-watch it. This project attempts to help solve that problem by converting long video files into text that can be easily searched and summarized.

## Overview

### Example Output

Example output text of a video transcription of [JFK's speech on going to the moon](https://www.c-span.org/classroom/document/?7986):

<https://user-images.githubusercontent.com/74869040/151491511-7486c34b-d1ed-4619-9902-914996e85125.mp4>

**vid2cleantxt output:**

> Now look into space to the moon and to the planets beyond and we have vowed that we shall not see it governed by a hostile flag of conquest but by a banner of freedom and peace we have vowed that we shall not see space filled with weapons of mass destruction but with instruments of knowledge and understanding yet the vow. In short our leadership in science and industry our hopes for peace and security our obligations to ourselves as well as others all require a. To solve these mysteries to solve them for the good of all men and to become the worlds leading space faring nation we set sail on this new sea because there is new knowledge to be gained and new rights to be won and they must be won and used for the progress of all people for space science like nuclear science and all technology. Has no conscience of its own whether it will become a force for good or ill depends on man and only if the united states occupies a position of preeminence can we help decide whether this new ocean will be a sea of peace

model = `openai/whisper-medium.en`
-

See the [demo notebook](https://colab.research.google.com/gist/pszemraj/9678129fe0b552e114e3576606446dee/vid2cleantxt-minimal-example.ipynb) for the full-text output.

### Pipeline Intro

![vid2cleantxt detailed](https://user-images.githubusercontent.com/74869040/131499569-c894c096-b6b8-4d17-b99c-a4cfce395ea8.png)

1.  The `transcribe.py` script uses `audio2text_functions.py` to convert video files to `.wav` format audio chunks of duration X\* seconds
2.  transcribe all X audio chunks through a pretrained transformer model
3.  Write all list results into a text file, store various runtime metrics into a separate text list, and delete the `.wav` audio chunk directory after using them.
4.  (Optional) create two new text files: one with all transcriptions appended and one with all metadata appended.
5.  FOR each transcription text file:
    -   Passes the 'base' transcription text through a spell checker (_Neuspell_) and auto-correct spelling. Saves as a new text file.
    -   Uses _pySBD_ to infer sentence boundaries on the spell-corrected text and add periods to delineate sentences. Saves as a new file.
    -   Runs essential keyword extraction (via _YAKE_) on spell-corrected file. All keywords per file are stored in one data frame for comparison and exported to the `.xlsx` format

_\*\* (where X is some duration that does not overload your computer/runtime)_

Given `INPUT_DIRECTORY`:

-   _final_ transcriptions in`.txt` will be in `INPUT_DIRECTORY/v2clntxt_transcriptions/results_SC_pipeline/`
-   metadata about transcription process will be in `INPUT_DIRECTORY/v2clntxt_transc_metadata`

* * *

## Quickstart

Install, then you can use `vid2cleantxt` in two ways:

1.  CLI via `transcribe.py` script from the command line (`python vid2cleantxt/transcribe.py --input-dir "path/to/video/files" --output-dir "path/to/output/dir"\`)
2.  As a python package, import `vid2cleantxt` and use the `transcribe` module to transcribe videos (`vid2cleantxt.transcribe.transcribe_dir()`)

Don't want to use it locally or don't have a GPU? you may be interested in the [demo notebook](https://colab.research.google.com/gist/pszemraj/9678129fe0b552e114e3576606446dee/vid2cleantxt-minimal-example.ipynb) on Google Colab.

### Installation

#### As a Python package

-   (recommended) Create a new virtual environment with `python3 -m venv venv`
    -   Activate the virtual environment with `source venv/bin/activate`
-   Install the repo with pip:

```bash
pip install git+https://github.com/pszemraj/vid2cleantxt.git
```

The library is now installed and ready to use in your Python scripts.

```python
import vid2cleantxt

text_output_dir, metadata_output_dir = vid2cleantxt.transcribe.transcribe_dir(
    input_dir="path/to/video/files",
    model_id="openai/whisper-base.en",
    chunk_length=30,
)

# do things with text files in text_output_dir
```

See below for more details on the `transcribe_dir` function.

#### Install from source

1.  `git clone https://github.com/pszemraj/vid2cleantxt.git`
    -   use the `--depth=1` switch to clone only the latest master (_faster_)
2.  `cd vid2cleantxt/`
3.  `pip install -e .`

As a shell block:

```bash
git clone https://github.com/pszemraj/vid2cleantxt.git --depth=1
cd vid2cleantxt/
pip install -e .
```

#### install details & gotchas

-   This should be automatically completed upon installation/import, but a spacy model may need to be downloaded for post-processing transcribed audio. This can be completed with `spacy download en_core_web_sm`
-   `FFMPEG` is required as a base system dependency to do anything with video/audio. This should be already installed on your system; otherwise see [the FFmpeg site](https://ffmpeg.org/).
-   We've added an implementation for whisper to the repo. Until further tests are completed, it's recommended to stick with the default 30s chunk length for these models. (_plus, they are fairly compute-efficient for the resulting quality_)

### example usage

**CLI example:** transcribe a directory of example videos in `./examples/` with the `whisper-small` model (not trained purely english) and print the transcriptions with the `cat` command:

```bash
python examples/TEST_folder_edition/dl_src_videos.py
python vid2cleantxt/transcribe.py -i ./examples/TEST_folder_edition/ -m openai/whisper-small
find ./examples/TEST_folder_edition/v2clntxt_transcriptions/results_SC_pipeline -name "*.txt" -exec cat {} +
```

Run `python vid2cleantxt/transcribe.py --help` for more details on the CLI.

**Python API example:** transcribe an input directory of user-specified videos using `whisper-tiny.en`, a smaller but faster model than the default.

```python
import vid2cleantxt

_my_input_dir = "path/to/video/files"
text_output_dir, metadata_output_dir = vid2cleantxt.transcribe.transcribe_dir(
    input_dir=_my_input_dir,
    model_id="openai/whisper-tiny.en",
    chunk_length=30,
)
```

Transcribed files can then be interacted with for whatever purpose (see [Visualization and Analysis](#visualization-and-analysis) and below for ideas).

```python
from pathlib import Path

v2ct_output_dir = Path(text_output_dir)
transcriptions = [f for f in v2ct_output_dir.iterdir() if f.suffix == ".txt"]

# read in the first transcription
with open(transcriptions[0], "r") as f:
    first_transcription = f.read()
print(
    f"The first 1000 characters of the first transcription are:\n{first_transcription[:1000]}"
)
```

See the docstrings of `transcribe_dir()` for more details on the arguments. One way you can do this is with `inspect`:

```python
import inspect
import vid2cleantxt

print(inspect.getdoc(vid2cleantxt.transcribe.transcribe_dir))
```

## Notebooks on Colab

Notebook versions are available on Google Colab as they offer accessible GPUs which makes vid2cleantxt _much_ faster.

As `vid2cleantxt` is now available as a package with python API, there is no longer a need for long, complicated notebooks. See [this notebook](https://colab.research.google.com/gist/pszemraj/9678129fe0b552e114e3576606446dee/vid2cleantxt-minimal-example.ipynb) for a relatively simple example - copy it to your drive and adjust as needed.

⚠️ The notebooks in `./colab_notebooks` are now deprecated and **not recommended to be used**. ⚠️ TODO: remove in a future PR.

**Resources for those new to Colab**

If you like the benefits Colab/cloud notebooks offer but haven't used them before, it's recommended to read the [Colab Quickstart](https://colab.research.google.com/notebooks/intro.ipynb), and some of the below resources as things like file I/O are different than your PC.

-   [Google's FAQ](https://research.google.com/colaboratory/faq.html)
-   [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
-   [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

* * *

## Details & Application

### How long does this take to run?

On Google Colab with a 16 GB GPU (available to free Colab accounts): **approximately 8 minutes to transcribe ~90 minutes of audio**. CUDA is supported - if you have an NVIDIA graphics card, you may see runtimes closer to that estimate on your local machine.

On my machine (CPU only due to Windows + AMD GPU), it takes approximately 30-70% of the total duration of input video files. You can also look at the "console printout" text files in `example_JFK_speech/TEST_singlefile`.

-   with model = `facebook/wav2vec2-base-960h` approx 30% of original video RT
-   with model = `facebook/hubert-xlarge-ls960-ft` (\_perhaps the best pre-whisper model anecdotally) approx 70-80% of original video RT
-   timing the whisper models is a TODO, but current estimate would be in between the above two models for `openai/whisper-base.en` on CPU.

**Specs:**

```text
Processor Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
Speed 4.8 GHz
Number of Cores 8
Memory RAM 32 GB
Video Card #1 Intel(R) UHD Graphics 620
Dedicated Memory 128 MB
Total Memory 16 GB
Video Card #2 AMD Radeon Pro WX3200 Graphics
Dedicated Memory 4.0 GB
Total Memory 20 GB
Operating System  Windows 10 64-bit
```

> _NOTE:_ that the default model is `openai/whisper-base.en`. See the [model card](https://huggingface.co/openai/whisper-base.en) for details.

### Now I have a bunch of long text files. How are these useful?

short answer: `noam_chomsky.jpeg`

More comprehensive answer:

With natural language processing and machine learning algorithms, text data can be visualized, summarized, or reduced in many ways. For example, you can use TextHero or ScatterText to compare audio transcriptions with written documents or use topic models or statistical models to extract key topics from each file. Comparing text data can help you understand how similar they are or identify vital differences.

#### Visualization and Analysis

1.  [TextHero](https://github.com/jbesomi/texthero) - cleans text, allows for visualization / clustering (k-means) / dimensionality reduction (PCA, TSNE)
    -   Use case here: I want to see how _this speaker_'s speeches differ from each other. Which are "the most related"?
2.  [Scattertext](https://github.com/JasonKessler/scattertext) - allows for comparisons of one corpus of text to another via various methods and visualizes them.
    -   Use case here: I want to see how the speeches by _this speaker_ compare to speeches by _speaker B_ in terms of topics, word frequency… so on

Some examples from my usage are illustrated below from both packages.

#### Text Extraction / Manipulation

1.  [Textract](https://textract.readthedocs.io/)
2.  [Textacy](https://github.com/chartbeat-labs/textacy)
3.  [YAKE](https://github.com/LIAAD/yake)
    -   A brief YAKE analysis is completed in this pipeline after transcribing the audio.

### Text Summarization

Several options are available on the [HuggingFace website](https://huggingface.co/models?pipeline_tag=summarization). To create a better, more general model for summarization, I have fine-tuned [this model](https://huggingface.co/pszemraj/bigbird-pegasus-large-K-booksum) on a [book summary dataset](https://arxiv.org/abs/2105.08209) which I find provides the best results for "lecture-esque" video conversion. I wrote a little about this and compared it to other models _WARNING: satire/sarcasm inside_ [here](https://www.dropbox.com/s/fsz9u4yk3hf9fak/A%20new%20benchmark%20for%20the%20generalizability%20of%20summarization%20models.pdf?dl=0).

I use several similar methods in combination with the transcription script. However, it isn't in a place to be officially posted yet. It will be posted to a public repo on this account when ready. You can now check out [this Colab notebook](https://colab.research.google.com/drive/1BSIsYHH0w5pdVxqo_nK5vHgMeBiJKKGm?usp=sharing) using the same example text that is output when the JFK speeches are transcribed.

#### TextHero example use case

Clustering vectorized text files into k-means groups:

![iml Plotting with TSNE + USE, Colored on Directory Name](https://user-images.githubusercontent.com/74869040/110546335-a0baaf80-812e-11eb-8d7d-48da00989dce.png)

![iml Plotting with TSNE + USE, Colored on K-Means Cluster](https://user-images.githubusercontent.com/74869040/110546452-c6e04f80-812e-11eb-9a4b-03213ec4a63b.png)

## ScatterText example use case

Comparing the frequency of terms in one body of text vs. another

![ST P 1 term frequency I ML 2021 Docs I ML Prior Exams_072122_](https://user-images.githubusercontent.com/74869040/110546149-69e49980-812e-11eb-9c94-81fcb395b907.png)

* * *

## Design Choices & Troubleshooting

### What python package dependencies does this repo have?

Upon cloning the repo, run the command `pip install -e .` (or`pip install -r requirements.txt` works too) in a terminal opened in the project directory. Requirements (upd. Oct 10, 2022) are:

```text
clean-text
GPUtil
humanize
joblib
librosa
moviepy~=1.0.3
natsort>=7.1.1
neuspell>=1.0.0
numpy
packaging
pandas>=1.3.0
psutil>=5.9.2
pydub>=0.24.1
pysbd>=0.3.4
requests
setuptools>=58.1.0
spacy>=3.0.0,<4.0.0
symspellpy~=6.7.0
torch>=1.8.2
tqdm
transformers>=4.23.0
wordninja==2.0.0
wrapt
yake>=0.4.8
```

If you encounter warnings/errors that mention FFmpeg, please download the latest version of FFMPEG from their website [here](https://www.ffmpeg.org/download.html) and ensure it is added to PATH.

### My computer crashes once it starts running the wav2vec2 model

First, try a smaller model: pass `-m openai/whisper-tiny.en` in CLI or `model_id="openai/whisper-tiny.en"` in python.

If that doesn't help, reducing the `chunk_length` duration can reduce computational intensity but is less accurate use `--chunk-len <INT>` when calling `vid2cleantxt/transcribe.py` or `chunk_length=INT` in python.

### The transcription is not perfect, and therefore I am mad

Perfect transcripts are not always possible, especially when the audio is not clean. For example, audio recorded with a microphone that is not always perfectly tuned to the speaker can cause the model to have issues. Additionally, the default models are not trained on specific speakers, and therefore the model will not be able to recognize the speaker / their accent.

Despite the small number of errors, the model can still recognize the speaker and their accent and capture a vast majority of the text. This should still save you a lot of time and effort.

### How can I improve the performance of the model from a word-error-rate perspective?

> As of Oct 2022: there's really shouldn't be much to complain about given what we had before whisper. That said, there may be some butgs or issues with the new model. Please report them in the issues section :)

The neural ASR model that transcribes the audio is typically the most crucial element to choose/tune. You can use **any whisper, wav2vec2, or wavLM model** from the [huggingface hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads); pass the model ID string with `--model` in CLI and `model_id="my-cool-model"` in python.

. _Note: It's recommended to experiment with the different variants of whisper first, as thhey are the most performant for the vast majority of "long speech" transcription use cases._

You can also train your own model, but that requires you to have a transcription of that person's speech. As you may find, manual transcription is a bit of a pain; therefore, transcripts are rarely provided - hence this repo. If interested see [this notebook](https://github.com/huggingface/notebooks/blob/master/examples/speech_recognition.ipynb)

### Why use transformer models instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters\*, which in my experience, can vary widely among English as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent transcription' can be made without spending a lot of time testing and optimizing parameters.

Also, because it's an API, you can't train it even if you wanted to, you have to be online for most of the script runtime functionally, and then, of course you have privacy concerns with sending data out of your machine.

_`*` these statements reflect the assessment completed around project inception in early 2021._

### Errors

-   \_pickle.UnpicklingError: invalid load key, '&lt;' --> Neuspell model was not downloaded correctly. Try re-downloading it.
-   manually open /Users/yourusername/.local/share/virtualenvs/vid2cleantxt-vMRD7uCV/lib/python3.8/site-packages/neuspell/../data
-   download the model from <https://github.com/neuspell/neuspell#Download-Checkpoints>
-   import neuspell
-   neuspell.seq_modeling.downloads.download_pretrained_model("scrnnelmo-probwordnoise")

## Examples

-   two examples are available in the `examples/` directory. One example is a single video (another speech), and the other is multiple videos (MIT OpenCourseWare). Citations are in the respective folders.
-   Note that the videos first need to be downloaded video the respective scripts in each folder first, i.e., run: `python examples/TEST_singlefile/dl_src_video.py`

## Future Work, Collaboration, & Citations

### Project Updates

A _rough_ timeline of what has been going on in the repo:

-   Oct 2022 Part 2 - Initial integration of [whisper](https://openai.com/blog/whisper/) model!
-   Oct 2022 - Redesign as Python package instead of an assortment of python scripts/notebooks that share a repository and do similar things.
-   Feb 2022 - Add backup functions for spell correction in case of NeuSpell failure (which, is a known issue at the time of writing).
-   Jan 2022 - add huBERT support, abstract the boilerplate out of Colab Notebooks. Starting work on the PDF generation w/ results.
-   Dec 2021 - greatly improved script runtime, and added more features (command line, docstring, etc.)
-   Sept-Oct 2021: Fixing bugs, and formatting code.
-   July 12, 2021 - sync work from Colab notebooks: add CUDA support for PyTorch in the `.py` versions, added Neuspell as a spell checker. General organization and formatting improvements.
-   July 8, 2021 - python scripts cleaned and updated.
-   April - June: Work done mostly on Colab, improving saving, grammar correction, etc.
-   March 2021: public repository added

### Future Work

> Note: these are largely not in order of priority.

0.  ~~add OpenAI's [whisper](https://github.com/openai/whisper) through integration with the transformers lib.~~
1.  Unfortunately, trying to use the [Neuspell](https://github.com/neuspell/neuspell) package is still not possible as the default package etc, has still not been fixed. I will add a permanent workaround to load/use with vid2cleantxt.
2.  ~~syncing improvements currently in the existing **Google Colab** notebooks (links) above, such as [NeuSpell](https://github.com/neuspell/neuspell)~~

    -   ~~this will include support for CUDA automatically when running the code (currently just on Colab)~~

3.  ~~clean up the code, add more features, and make it more robust.~~
4.  add a script to convert `.txt` files to a clean PDF report, [example here](https://www.dropbox.com/s/fpqq2qw7txbkujq/ACE%20NLP%20Workshop%20-%20Session%20II%20-%20Dec%202%202021%20-%20full%20transcription%20-%20txt2pdf%2012.05.2021%20%20Standard.pdf?dl=1)
5.  add summarization script/module
6.  further expand the functionality of the `vid2cleantxt` module
7.  Add support for transcribing the other languages in the whisper model (e.g., French, German, Spanish, etc.). This will require synchronized API changes to ensure that English spell correction is only applied to English transcripts, etc.

### I've found x repo / script / concept that I think you should incorporate or collaborate with the author

Could you send me a message / start a discussion? Always looking to improve. Or create an issue that works too.

### Citations

**whisper (OpenAI)**

    @report{,
       abstract = {We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zero-shot transfer setting without the need for any fine-tuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.},
       author = {Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine Mcleavey and Ilya Sutskever},
       title = {Robust Speech Recognition via Large-Scale Weak Supervision},
       url = {https://github.com/openai/},
    }

**wav2vec2 (fairseq)**

> Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations, 2019.

-   repo [link](https://github.com/pytorch/fairseq)

**HuBERT (fairseq)**

    @article{Hsu2021,
       author = {Wei Ning Hsu and Benjamin Bolte and Yao Hung Hubert Tsai and Kushal Lakhotia and Ruslan Salakhutdinov and Abdelrahman Mohamed},
       doi = {10.1109/TASLP.2021.3122291},
       issn = {23299304},
       journal = {IEEE/ACM Transactions on Audio Speech and Language Processing},
       keywords = {BERT,Self-supervised learning},
       month = {6},
       pages = {3451-3460},
       publisher = {Institute of Electrical and Electronics Engineers Inc.},
       title = {HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units},
       volume = {29},
       url = {<https://arxiv.org/abs/2106.07447v1>},
       year = {2021},
    }

**MoviePy**

-   [link](https://github.com/Zulko/moviepy) to repo as no citation info given

**symspellpy / symspell**

-   repo [link](https://github.com/mammothb/symspellpy/tree/e7a91a88f45dc4051b28b83e990fe072cabf0595)
-   copyright:
    > Copyright (c) 2020 Wolf Garbe Version: 6.7 Author: Wolf Garbe <mailto:wolf.garbe@seekstorm.com>
    > Maintainer: Wolf Garbe <mailto:wolf.garbe@seekstorm.com>
    > URL: <https://github.com/wolfgarbe/symspell>
    > Description: <https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f>
    >
    > MIT License
    >
    > Copyright (c) 2020 Wolf Garbe
    >
    > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    > documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    > rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
    > persons to whom the Software is furnished to do so, subject to the following conditions:
    >
    > The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    > Software.
    >
    > <https://opensource.org/licenses/MIT>

**YAKE (yet another keyword extractor)**

-   repo [link](https://github.com/LIAAD/yake)
-   relevant citations:
    > In-depth journal paper at Information Sciences Journal
    >
    > Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword
    > Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp
    > 257-289. pdf
    >
    > ECIR'18 Best Short Paper
    >
    > Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). A Text Feature Based Automatic
    > Keyword Extraction Method for Single Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances
    > in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772,
    > pp. 684 - 691. pdf
    >
    > Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). YAKE!
    > Collection-independent Automatic Keyword Extractor. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds).
    > Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol
    > 10772, pp. 806 - 810. pdf

#### Video Citations

-   <div class="csl-entry"><i>President Kennedy’s 1962 Speech on the US Space Program | C-SPAN Classroom</i>. (n.d.). Retrieved January 28, 2022, from https://www.c-span.org/classroom/document/?7986</div>

-   _Note: example videos are cited in respective `Examples/` directories_
