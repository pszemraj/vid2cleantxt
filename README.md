```

```

# vid2cleantxt

![vid2cleantext simple](https://user-images.githubusercontent.com/74869040/131500291-ed0a9d7f-8be7-4f4b-9acf-c360cfd46f1f.png)

**vid2cleantxt**: a [transformers-based](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) pipeline for turning heavily speech-based video files into clean, readable text from the audio.

TL;DR check out [this Colab script](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing) to see a transcription and keyword extraction of a speech by John F. Kennedy by simply running all cells.

* * *

**Table of Contents**

<!-- TOC -->

- [Motivation](#motivation)
- [Overview](#overview)
  - [Example Output](#example-output)
  - [Pipeline Intro](#pipeline-intro)
- [Installation](#installation)
  - [Quickstart (aka: how to get the script running)](#quickstart-aka-how-to-get-the-script-running)
  - [Notebooks on Colab](#notebooks-on-colab)
  - [How long does this take to run?](#how-long-does-this-take-to-run)
- [Application](#application)
  - [Now I have a bunch of long text files. How are these useful?](#now-i-have-a-bunch-of-long-text-files-how-are-these-useful)
    - [Visualization and Analysis](#visualization-and-analysis)
    - [Text Extraction / Manipulation](#text-extraction--manipulation)
    - [Text Summarization](#text-summarization)
  - [TextHero example use case](#texthero-example-use-case)
  - [ScatterText example use case](#scattertext-example-use-case)
- [Design Choices & Troubleshooting](#design-choices--troubleshooting)
  - [What python package dependencies does this repo have?](#what-python-package-dependencies-does-this-repo-have)
  - [My computer crashes once it starts running the wav2vec2 model:](#my-computer-crashes-once-it-starts-running-the-wav2vec2-model)
  - [The transcription is not perfect, and therefore I am mad:](#the-transcription-is-not-perfect-and-therefore-i-am-mad)
  - [How can I improve the performance of the model from a word-error-rate perspective?](#how-can-i-improve-the-performance-of-the-model-from-a-word-error-rate-perspective)
  - [Why use wav2vec2 instead of SpeechRecognition or other transcription methods?](#why-use-wav2vec2-instead-of-speechrecognition-or-other-transcription-methods)
- [Example](#example)
  - [Result](#result)
  - [Console output](#console-output)
- [Future Work, Collaboration, & Citations](#future-work-collaboration--citations)
  - [Project Updates](#project-updates)
  - [Future Work](#future-work)
  - [I've found x repo / script / concept that I think you should incorporate or collaborate with the author.](#ive-found-x-repo--script--concept-that-i-think-you-should-incorporate-or-collaborate-with-the-author)
  - [Citations](#citations)

<!-- /TOC -->

* * *

# Motivation

Video, specifically audio, is an inefficient way to convey dense or technical information. The viewer has to sit through the whole thing, while only part of the video may be relevant to them. If you don't understand a statement or concept, you have to search through the video, or re-watch it. This project attempts to help solve that problem by converting long video files into text that can be easily searched and summarized.

# Overview

## Example Output

Example output text of a video transcription of [JFK's peace speech](https://youtu.be/0fkKnfk4k40):

> Surely the opening vistas of space promise high costs and hardships as well as high reward so it is not surprising that some would have us stay where we are a little longer to rest to wait but this city of question this state of taxes this country of the united states was not built by those who waited and rested but if I were to say my fellow citizens. That we shall send to the moon two hundred and forty thousand miles away from the control station in Houston a giant rocket more than three hundred feet tall the length of this football field made of new metal alloys some of which have not yet been invented capable of standing heat and stresses several times more than have ever been experienced fitted together with a precision better than the. First watch carrying all the equipment needed for propulsion guidance control communications food and survival on an untried mission to an unknown celestial body and then return it safely to earth re entering the atmosphere at speeds of over twenty five thousand miles per hour causing heat about half that on the temperature of the sun almost as hot as it is here to day and do all this. And do all this and do it right and do it first before this dictate is out then we must be so I'm the one who's doing all the work so to get me to stay cool for a minute however I think we're going to do it and I think that we must pay what needs to be paid I don't think we ought to waste any. Money but I think we ought to do the job and this will be done in the decade of the sixty it may be done while some of you are still here at school at this college and university it will be done during the terms of office of some of the people who sit here on this platform but it will be done many years ago the great British explorer garage memory who was to die on mount everist was asked why did he want to climb it the said because it is there well space is there. And we're going to climb it and the moon and the planets are there and new hopes for knowledge and peace are there and therefore as we set sail we ask god's blessing on the most hazardous and dangerous and greatest adventure on which man has ever embarked thank you

See the examples folder for more detail / full transcript.

## Pipeline Intro

![vid2cleantxt detailed](https://user-images.githubusercontent.com/74869040/131499569-c894c096-b6b8-4d17-b99c-a4cfce395ea8.png)

1.  The `transcribe.py` script uses audio2text_functions.py to convert video files to .wav format audio chunks of duration X\* seconds,
2.  transcribe all X audio chunks through a pretrained wav2vec2 model, s
3.  Write all results of the list into a text file, stores various runtime metrics into a separate text list, and deletes the .wav audio chunk directory after completed using them.
4.  creates two new text files: one with all transcriptions appended, and one with all metadata appended. The script then
5.  FOR each transcription text file:
    -   Passes the 'base' transcription text through a spell checker (_Neuspell_) and autocorrects spelling. Saves as new text file.
    -   Uses pySBD to infer sentence boundaries on the spell-corrected text and add periods in to delineate sentences. Saves as new file.
    -   Runs basic keyword extraction (via YAKE) on spell-corrected file. All keywords per file are stored in one dataframe for comparison, and exported to .xlsx format

_\*\* (where X is some duration that does not overload your computer or crash your IDE)_

By default,

-   results are stored in `~/v2clntxt_transcriptions`
-   metadata in `~/v2clntxt_transc_metadata`

(where **~** is path entered by user)

# Installation

## Quickstart (aka: how to get the script running)

Essentially, clone the repo, and run `vid2cleantxt/transcribe.py --input-dir "path to inputs`. the main arg to pass is `--input-dir` for, well, the inputs.

> **Note:** _the first time the code runs on your machine, it will download the pretrained transformers models_ which include wav2vec2 and a scibert model for spell correction. After the first run, it will be cached locally, and you will not need to sit through that again.

1.  fastest (in bash command line):

    1.  `git clone https://github.com/pszemraj/vid2cleantxt.git`
    2.  `cd vid2cleantxt/`
    3.  `pip install -r requirements.txt`
    4.  `python vid2cleantxt/transcribe.py --input-dir "example_JFK_speech/TEST_singlefile"`
        > in this example, all video and audio files in the repo example "example_JFK_speech/TEST_singlefile" would be transcribed.

2.  Clone with [github desktop](https://desktop.github.com/)

    1.  install requirements.txt either from your IDE prompt or via the command above
    2.  open terminal in the local folder via your IDE or manual
    3.  `python vid2cleantxt/transcribe.py --input-dir "example_JFK_speech/TEST_singlefile"` in said terminal

    > in this example, all video and audio files in "example_JFK_speech/TEST_singlefile" would be transcribed.

3.  If neither of those are convenient, see the next section on how to use Colab (which for most users, would be faster anyway)

## Notebooks on Colab

Notebook versions are available on Google Colab, because they offer free GPUs which makes vid2cleantxt _much_ faster. If you want a notebook to run locally for whatever reason, in Colab you can download as .ipynb, but you may need to make some small changes (some directories, packages, etc. are specific to Colab's structure) - the same goes for the colab notebooks in this repo.

Links to Colab Scripts:

1.  Single-File Version (Implements GPU)
    -   Link [here](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing)
    -   This script downloads the video from a public link to one of the JFK videos stored on my Google Drive. As such, no
        authentication / etc. is required and **this link is recommended for seeing how this pipeline works**.
    -   The only steps required are checking / adjusting the runtime to a GPU, and _Run All_
2.  Multi-File Version (Implements GPU)
    -   Link [here](https://colab.research.google.com/drive/1qOUkiPMaUZgBTMfCFF-fCRTPCMg1997J?usp=sharing), _note file was updated and posted to the repo July 13, 2021._
    -   This script connects to the user's google drive to convert a whole folder of videos using Google's Colab Python
        package.
    -   It **does require the video files to be hosted on the user's drive**, as well as authorization of Colab (it will prompt you and walk you through this)

New to Colab? Some links I found useful:

-   [Google's FAQ](https://research.google.com/colaboratory/faq.html)
-   [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
-   [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
-   [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

## How long does this take to run?

On Google Colab with a 16 gb GPU (should be available to free Colab accounts): **approximately 8 minutes to transcribe ~
90 minutes of audio**. As of July 13, 2021, updated to ensure that CUDA will be used on local machines - if you have an NVIDIA graphics card, you may see runtimes closer to that estimate.

On my machine (CPU only due to Windows + AMD GPU) it takes approximately 30-70% of the total duration of input video files. You can also take a look at the "console printout" text files in `example_JFK_speech/TEST_singlefile`.

-   with model = "facebook/wav2vec2-base-960h" (default) approx 30% of original RT
-   with model = "facebook/wav2vec2-large-960h-lv60-self" approx 70% of original RT

**Specs:**

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

> _NOTE:_ that the default model is facebook/wav2vec2-base-960h. This is a pre-trained model that is trained on the librispeech corpus. If you want to use a different model, you can pass the `--model` argument (for example `--model "facebook/wav2vec2-large-960h-lv60-self"`). The model is downloaded from the internet if it does not exist locally. The large model is more accurate, but is also slower to run. I do not have stats on differences in WER, but [facebook](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) may have some posted.

* * *

# Application

## Now I have a bunch of long text files. How are these useful?

short answer: noam_chomsky.jpeg

more comprehensive answer:

Text data can be visualized, summarized, or reduced in many ways with natural language processing and machine learning algorithms. For example, you can use TextHero or ScatterText to compare audio transcriptions with written documents, or use topic models or statistical models to extract key topics from each file. Comparing text data in this way can help you understand how similar they are, or identify key differences.

### Visualization and Analysis

1.  [TextHero](https://github.com/jbesomi/texthero) - cleans text, allows for visualization / clustering (k-means) / dimensionality reduction (PCA, TSNE)
    -   Use case here: I want to see how _this speaker_'s speeches differ from each other. Which are "the most related"?
2.  [Scattertext](https://github.com/JasonKessler/scattertext) - allows for comparisons of one corpus of text to another via various methods and visualizes them.
    -   Use case here: I want to see how the speeches by _this speaker_ compare to speeches by _speaker B_ in terms of topics, word frequency… so on

Some examples from my own usage are illustrated below from both packages.

### Text Extraction / Manipulation

1.  [Textract](https://textract.readthedocs.io/)
2.  [Textacy](https://github.com/chartbeat-labs/textacy)
3.  [YAKE](https://github.com/LIAAD/yake)
    -   A brief YAKE analysis is completed in this pipeline after transcribing the audio.

### Text Summarization

Several options are available on the [HuggingFace website](https://huggingface.co/models?pipeline_tag=summarization). I have personally found [Google's pegasus](https://huggingface.co/google/pegasus-xsum) to be most effective for "lecture-esque" video conversion.

I personally use several similar methods in combination with the transcription script, however it isn't in a place to be officially posted yet. It will be posted to a public repo on this account when ready. For now, you can check out [this Colab notebook](https://colab.research.google.com/drive/1BSIsYHH0w5pdVxqo_nK5vHgMeBiJKKGm?usp=sharing) using the same example text that is output when the JFK speeches are transcribed.

## TextHero example use case

Clustering vectorized text files into k-means groups:

![iml Plotting with TSNE + USE, Colored on Directory Name](https://user-images.githubusercontent.com/74869040/110546335-a0baaf80-812e-11eb-8d7d-48da00989dce.png)

![iml Plotting with TSNE + USE, Colored on K-Means Cluster](https://user-images.githubusercontent.com/74869040/110546452-c6e04f80-812e-11eb-9a4b-03213ec4a63b.png)

## ScatterText example use case

Comparing frequency of terms in one body of text vs. another

![ST P 1 term frequency I ML 2021 Docs I ML Prior Exams_072122_](https://user-images.githubusercontent.com/74869040/110546149-69e49980-812e-11eb-9c94-81fcb395b907.png)

* * *

# Design Choices & Troubleshooting

## What python package dependencies does this repo have?

Upon cloning the repo, run the command `pip install -r requirements.txt` in a terminal opened in the project directory. Requirements (upd. Dec 14, 2021) are:

    librosa~=0.8.1
    wordninja~=2.0.0
    psutil~=5.8.0
    natsort~=7.1.1
    pandas~=1.3.0
    moviepy~=1.0.3
    transformers~=4.8.2
    numpy~=1.21.0
    pydub~=0.24.1
    symspellpy~=6.7.0
    joblib~=1.0.1
    torch~=1.9.0
    tqdm~=4.43.0
    plotly~=4.14.3
    yake~=0.4.8
    pysbd~=0.3.4
    clean-text
    GPUtil~=1.4.0
    humanize~=3.13.1
    neuspell~=1.0.0
    openpyxl >=3
    unidecode~=1.3.2
    spacy>=3.0.0,<4.0.0
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm

-   _Note: the github link in the reqs above downloads the spaCy model `en_core_web_sm` as part of the setup/installation process so you don't have to manually type `python -m spacy download en_core_web_sm` into the terminal to be able to run the code. More on this is described on spaCy's website [here](https://spacy.io/usage/models#production)_

## My computer crashes once it starts running the wav2vec2 model:

Try passing a lower `--chunk-len <INT>` when calling `vid2cleantxt/transcribe.py`. Until you get to really small intervals (say &lt; 8 seconds) each audio chunk can be treated as approximately independent as they are different sentences.

## The transcription is not perfect, and therefore I am mad:

Perfect transcripts are not always possible, especially when the audio is not clean. For example, the audio is recorded with a microphone that is not always perfectly tuned to the speaker can cause the model to have issues. Additionally, the default models are not trained on specific speakers and therefore the model will not be able to recognize the speaker / their accent.

Despite the small amount of errors, the model is still able to recognize the speaker and their accent and capture a vast majority of the text. This should still save you a lot of time and effort.

## How can I improve the performance of the model from a word-error-rate perspective?

if you change the default model by passing `--model "facebook/wav2vec2-large-960h-lv60-self"`from the default `facebook/wav2vec2-base-960h` the model will be considerably more accurate - I defer to facebook for the official stats.

You can also train the model, but that requires that you already have a transcription of that person's speech already. As you may find, manual transcription is a bit of a pain and therefore transcripts are rarely provided - hence this repo.

## Why use wav2vec2 instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters\*, which in my experience can vary widely among english as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent transcription' can be made without having to spend a bunch of time testing and optimizing parameters.

Also, because it's an API you can't train it even if you wanted to, you have to be online for functionally most of the script runtime, and then of course you have privacy concerns with sending data out of your machine.

_`*` these statements reflect the assessment completed around project inception about early 2021._

# Example

Transcription of Public Domain Speeches from President John F. Kennedy

## Result

> Surely the opening vistas of space promise high costs and hardships as well as high reward so it is not surprising that some would have us stay where we are a little long. To rest to wait but this city of question this state of taxes this country of the united states was not built by those who waited and rested but if I were to say my fellow citizens. That we shall send to the moon two hundred and forty thousand miles away from the control station in Houston a giant rocket more than three hundred feet tall the length of this football field. Made of new metal alloys some of which have not yet been invented capable of standing heat and stresses several times more than have ever been experienced fitted together with a precision better than the. Itself watch carrying all the equipment needed for propulsion guidance control communications food and survival on an untried mission to an unknown celestial body and then. Turn it safely to earth re entering the atmosphere at speeds of over twenty five thousand miles per hour causing heat about half that on the temperature of the sun almost as hot as it is here to day and do all this. And do all this and do it right and do it first before this dictate is out then we must be for. I'm the one who's doing all the work so well to get it to stay cool for a minute however I think we're going to do it and I think that we must pay what needs to be paid I don't think we ought to waste any. Money but I think we ought to do the job and this will be done in the decade of the sixty it may be done while some of you are still here at school at this college and university it will be done during the terms of office of some of the people who sit here on this platform. It will be gone many years ago the great British explorer george military who was to die on mount everist was asked why did he want to climb it he said because it is there well space is there. And we're going to climb it and the moon and the planets are there and new hopes for knowledge and peace are there and therefore as we set sail we ask god's blessing on the most hazardous and dangerous and. Greatest adventure on which man has ever embarked again.

## Console output

    (v2ct) C:\Users\peter\Dropbox\programming_projects\vid2cleantxt>python vid2cleantxt\transcribe.py --input-dir "C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech" --model "facebook/wav2vec2-large-960h-lv60-self"
    data folder is set to `C:\Users\peter\.conda\envs\v2ct\lib\site-packages\neuspell\../data` script
    Loading models @ Dec-19-2021_-20-51-56 - may take a while...
    If RT seems excessive, try --verbose flag or checking logfile

    Found 1 audio or video files in C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech
    Creating .wav audio clips: 100%|███████████████████████████████████████████████████████| 12/12 [00:00<00:00, 84.82it/s]
    Creating .wav audio clips:   8%|████▋                                                   | 1/12 [00:00<00:01,  9.11it/s]
    created audio chunks for wav2vec2 - Dec-19-2021_-20
    No GPU being used by this machine :(

                                                                                                                           No GPU being used :/   0%|                                                                       | 0/12 [00:00<?, ?it/s]


    Gen RAM Free: 10.8 GB | Proc size: 3.1 GB | 8 CPUs  loaded at 22.7 % |

                                                                                                                           No GPU being used :/  50%|███████████████████████████████▌                               | 6/12 [00:57<00:51,  8.65s/it]


    Gen RAM Free: 10.3 GB | Proc size: 3.2 GB | 8 CPUs  loaded at 67.8 % |

    Transcribing video: 100%|██████████████████████████████████████████████████████████████| 12/12 [01:43<00:00,  8.67s/it]
    Saved transcript and metadata to C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech\v2clntxt_transcriptions and C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech\v2clntxt_transc_metadata
    transcribing vids: 100%|████████████████████████████████████████████████████████████████| 1/1 [01:45<00:00, 105.41s/it]
    SC_pipeline - transcribed audio:   0%|                                                           | 0/1 [00:00<?, ?it/s]
    Top 10 Key Phrases from YAKE, with max n-gram length 3
    ['forty thousand miles',
     'promise high costs',
     'hundred feet tall',
     'guidance control communications',
     'unknown celestial body',
     'station in Houston',
     'high reward',
     'causing heat',
     'surely the opening',
     'waited and rested']
    SC_pipeline - transcribed audio: 100%|███████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.09s/it]


    Finished at: Dec-19-2021_-20. Total RT was 2.2819159250000003 mins
    relevant files for run are in:
    C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech\v2clntxt_transcriptions
     and:
    C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\scratch\moon-speech\v2clntxt_transc_metadata

* * *

# Future Work, Collaboration, & Citations

## Project Updates

A _rough_ timeline of what has been going on in the repo:

-   Dec 2021 - greatly improved runtime of the script, and added more features
-   Sept-Oct 2021: Fixing bugs, formatting code.
-   July 12, 2021 - sync work from Colab notebooks: add CUDA support for pytorch in the `.py` versions, added Neuspell as a spell checker. General organization and formatting improvements.
-   July 8, 2021 - python scripts cleaned and updated.
-   April - June: Work done mostly on Colab improving saving, grammar correction, etc.
-   March 2021: public repository added

## Future Work

1.  ~~syncing improvements currently in the existing **Google Colab** notebooks (links) above such as [NeuSpell](https://github.com/neuspell/neuspell)~~

    -   ~~this will include support for CUDA automatically when running the code (currently just on Colab)~~

2.  convert groups of functions to a class object
3.  publish class as a python package to streamline process / reduce overhead, making it easier to use + adopt.
4.  Include additional features that are currently not public:

    -   Summarization of video transcriptions
    -   Paragraph Disambiguation in both transcription & summarization
    -   report generation (see results in .PDF for notes, etc.)

5.  py2exe (once code optimized)

## I've found x repo / script / concept that I think you should incorporate or collaborate with the author.

Send me a message / start a discussion! Always looking to improve.

## Citations

**wav2vec2 (fairseq)**

> Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations, 2019.

-   repo [link](https://github.com/pytorch/fairseq)

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


```

```
