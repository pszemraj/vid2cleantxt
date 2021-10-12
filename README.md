# vid2cleantxt

![vid2cleantext simple](https://user-images.githubusercontent.com/74869040/131500291-ed0a9d7f-8be7-4f4b-9acf-c360cfd46f1f.png)

**vid2cleantxt**: a [transformers-based](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) pipeline for turning heavily speech-based video files into clean, readable text from the audio.

TL;DR check out [this Colab script](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing) to see a transcription and keyword extraction of a speech by John F. Kennedy by simply running all cells.

* * *

**Table of Contents**

<!-- TOC -->

-   [vid2cleantxt](#vid2cleantxt)
-   [Motivation](#motivation)
-   [Overview](#overview)
    -   [Example Output](#example-output)
    -   [Pipeline Intro](#pipeline-intro)
-   [Installation](#installation)
    -   [How to get this to work on your machine](#how-to-get-this-to-work-on-your-machine)
    -   [Is there a jupyter notebook file?](#is-there-a-jupyter-notebook-file)
    -   [How long does this take to run?](#how-long-does-this-take-to-run)
-   [Application](#application)
    -   [Now I have a bunch of long text files. How are these useful?](#now-i-have-a-bunch-of-long-text-files-how-are-these-useful)
        -   [Visualization and Analysis](#visualization-and-analysis)
        -   [Text Extraction / Manipulation](#text-extraction--manipulation)
        -   [Text Summarization](#text-summarization)
    -   [TextHero example use case](#texthero-example-use-case)
    -   [ScatterText example use case](#scattertext-example-use-case)
-   [Design Choices & Troubleshooting](#design-choices--troubleshooting)
    -   [What python package dependencies does this repo have?](#what-python-package-dependencies-does-this-repo-have)
    -   [I tried to transcribe an audio file, and it gave me an error:](#i-tried-to-transcribe-an-audio-file-and-it-gave-me-an-error)
    -   [My computer crashes once it starts running the wav2vec2 model:](#my-computer-crashes-once-it-starts-running-the-wav2vec2-model)
    -   [How can I improve the performance of the model from a word-error-rate perspective?](#how-can-i-improve-the-performance-of-the-model-from-a-word-error-rate-perspective)
    -   [Why use wav2vec2 instead of SpeechRecognition or other transcription methods?](#why-use-wav2vec2-instead-of-speechrecognition-or-other-transcription-methods)
-   [Example](#example)
    -   [Description](#description)
    -   [Output (sentence boundary disambiguation) of JFK's Moon Speech @ Rice University:](#output-sentence-boundary-disambiguation-of-jfks-moon-speech--rice-university)
    -   [Output script run log for the "single_file" version:](#output-script-run-log-for-the-single_file-version)
-   [Future Work, Collaboration, & Citations](#future-work-collaboration--citations)
    -   [Project Updates](#project-updates)
    -   [Future Work](#future-work)
    -   [What about a version where I don't need python at all?](#what-about-a-version-where-i-dont-need-python-at-all)
    -   [I've found x repo / script / concept that I think you should incorporate or collaborate with the author.](#ive-found-x-repo--script--concept-that-i-think-you-should-incorporate-or-collaborate-with-the-author)
    -   [Citations](#citations)

<!-- /TOC -->

* * *

# Motivation

When compared to other media (such as text and pictures), video (specifically the audio) is an inefficient way to convey dense or technical information, as in the base case the viewer has to sit through the whole thing, while only part of the video may be relevant to them. Even worse, if you don't understand a statement or concept, you have to search through the video, or re-watch it, taking up significant amounts of time. This project attempts to help solve that problem by converting long video files into text that you can read, CTRL+F, keyword extract, and summarize.

# Overview

## Example Output

Example output text of a video transcription of [JFK's peace speech](https://youtu.be/0fkKnfk4k40):

> Surely the opening vistas of space promise high costs and hardships as well as high reward so it is not surprising that some would have us stay where we are a little longer to rest to wait but this city of question this state of taxes this country of the united states was not built by those who waited and rested but if I were to say my fellow citizens. That we shall send to the moon two hundred and forty thousand miles away from the control station in Houston a giant rocket more than three hundred feet tall the length of this football field made of new metal alloys some of which have not yet been invented capable of standing heat and stresses several times more than have ever been experienced fitted together with a precision better than the. First watch carrying all the equipment needed for propulsion guidance control communications food and survival on an untried mission to an unknown celestial body and then return it safely to earth re entering the atmosphere at speeds of over twenty five thousand miles per hour causing heat about half that on the temperature of the sun almost as hot as it is here to day and do all this. And do all this and do it right and do it first before this dictate is out then we must be so I'm the one who's doing all the work so to get me to stay cool for a minute however I think we're going to do it and I think that we must pay what needs to be paid I don't think we ought to waste any. Money but I think we ought to do the job and this will be done in the decade of the sixty it may be done while some of you are still here at school at this college and university it will be done during the terms of office of some of the people who sit here on this platform but it will be done many years ago the great British explorer garage memory who was to die on mount everist was asked why did he want to climb it the said because it is there well space is there. And we're going to climb it and the moon and the planets are there and new hopes for knowledge and peace are there and therefore as we set sail we ask god's blessing on the most hazardous and dangerous and greatest adventure on which man has ever embarked thank you

See the examples folder for more detail / full transcript.

## Pipeline Intro

![vid2cleantxt detailed](https://user-images.githubusercontent.com/74869040/131499569-c894c096-b6b8-4d17-b99c-a4cfce395ea8.png)

Here's a high-level overview of what happens in the `vid2cleantxt_folder.py` script to create the output shown above:

1.  Imports relevant packages, and imports relevant functions from audio2text_functions.py
2.  Receive **directory** string input from user in "script run window\*. Then iterates through that directory, and finds
    all video files
3.  FOR each video file found:
    -   convert video to .wav format audio chunks of duration X\*\* seconds with MoviePy
    -   transcribe all X audio chunks through
        a [pretrained wav2vec2 model](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)
        (in this repo - using huggingface transformers) and store the resulting text in a list
    -   write all results of the list into a text file, store various runtime metrics into a separate text list
    -   Delete .wav audio chunk directory after completed using them
4.  Next, create two new text files: one with all transcriptions appended, one with all metadata appended.
5.  Then FOR each transcription text file:
    -   Pass the 'base' transcription text through a spell checker (symspellpy) and autocorrect spelling. save as new text
        file.
    -   Use pySBD to infer sentence boundaries on the spell-corrected text and add periods in to delineate sentences. Save  as new file.
    -   Run basic keyword extraction (via YAKE) on spell-corrected file. All keywords per file are stored in one dataframe
        for comparison, and exported to .xlsx format.
6.  cleanup tasks, report runtime, and exit.

_\* the 'single file' version needs to have the name defined in the python code itself_

_\*\* (where X is some duration that does not overload your computer or crash your IDE)_

By default,

-   results are stored in ~/w2v2_video_transcriptions
-   metadata in ~/w2v2_transcription_metadata

(where **~** is path entered by user)

# Installation

## How to get this to work on your machine

Essentially, clone the repo, and run `vid2cleantxt/vid2cleantxt_folder.py`. The script does not require any args as it will immediately prompt you for the directory to search for video files in.

> **Note:** the first time the code runs on your machine, it will download the pretrained transformers model (~1
> gb). After the first run, it will be cached locally, and you will not need to sit through that again.

1.  fastest (in bash command line):


    1. `git clone https://github.com/pszemraj/vid2cleantxt.git` 
    2. `cd vid2cleantxt/`
    3. `pip install -r requirements.txt`
    4. `python vid2cleantxt/vid2cleantxt_folder.py`
    

2.  Clone with github desktop - run `vid2cleantxt/vid2cleantxt_folder.py` from your IDE
    -   `vid2cleantxt_standalone.py` contains all the python functions, classes, etc. to run a transcription. If you are experiencing issues, try to see if running this script (from whatever location) works.
3.  If neither of those are convenient, see the next section on how to use Colab

## Is there a jupyter notebook file?

No, but there are versions of these scripts on Google Colab. From Colab you can download as .ipynb, but you may need to
make some small changes (some directories, packages, etc. are specific to Colab's structure). Links to Colab Scripts:

1.  Single-File Version (Implements GPU)
    -   Link [here](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing)
    -   This script downloads the video from a public link to one of the JFK videos stored on my Google Drive. As such, no
        authentication / etc. is required and **this link is recommended for seeing how this pipeline works**.
    -   The only steps required are checking / adjusting the runtime to a GPU, and _Run All_
2.  Multi-File Version (Implements GPU)
    -   Link [here](https://colab.research.google.com/drive/1qOUkiPMaUZgBTMfCFF-fCRTPCMg1997J?usp=sharing), *note file was updated and posted to the repo July 13, 2021.*
    -   This script connects to the user's google drive to convert a whole folder of videos using Google's Colab Python
        package.
    -   It **does require the video files to be hosted on the user's drive**, as well as authorization of Colab (it will
        prompt you and walk you through this)

New to Colab? Some links I found useful:

-   [Google's FAQ](https://research.google.com/colaboratory/faq.html)
-   [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
-   [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
-   [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

## How long does this take to run?

On Google Colab with a 16 gb GPU (should be available to free Colab accounts): **approximately 8 minutes to transcribe ~
90 minutes of audio**. As of July 13, 2021, updated to ensure that CUDA will be used on local machines - if you have an NVIDIA graphics card, you may see runtimes closer to that estimate.

On my machine (CPU only due to Windows + AMD GPU) it takes approximately 80-120% of the total duration of input video files. You can also take a look at the "console printout" text files in `example_JFK_speech/TEST_folder_edition` and `TEST_singlefile` for more details.

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

> _Note:_ if you change `wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"` to `wav2vec2_model = "
> facebook/wav2vec2-base-960h"`the runtime will be considerably faster. I do not have stats on differences in WER, but  [facebook](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) may have some posted.

* * *

# Application

## Now I have a bunch of long text files. How are these useful?

short answer: noam_chomsky.jpeg

more comprehensive answer:

A large corpus of text can be visualized / summarized / reduced in many ways with natural language processing / typical
machine learning algorithms (i.e., classifying text, so on). Some packages to check out regarding this are TextHero and ScatterText. An example use case is combining the text from audio transcriptions with written documents (say textbooks or lecture slides converted to text) for comparison of how similar they are. You can also use topic models (available in ScatterText and many other packages) or statistical models (YAKE) to extract key topics from each file (or file group) and compare those (how they change over time, what are the key topics in practice exam PDF files, etc).

### Visualization and Analysis

1.  [TextHero](https://github.com/jbesomi/texthero) - cleans text, allows for visualization / clustering (k-means) /     dimensionality reduction (PCA, TSNE)
    -   Use case here: I want to see how _this speaker_'s speeches differ from each other. Which are "the most related"?
2.  [Scattertext](https://github.com/JasonKessler/scattertext) - allows for comparisons of one corpus of text to another     via various methods and visualizes them.
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

Upon cloning the repo, run the command `pip install -r requirements.txt` in a terminal opened in the project directory. Requirements (upd. Oct 12, 2021) are:

```
GPUtil>=1.4.0
clean-text>=0.4.0
humanize>=3.10.0
librosa>=0.8.1
moviepy==1.0.3
natsort>=7.1.1
neuspell>=1.0.0
openpyxl >=3
pandas>=1.2.5
plotly>=5.1.0
psutil>=5.8.0
pysbd>=0.3.4
pyspellchecker>=0.6.2
spacy>=3.0.0,<4.0.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm
spellchecker>=0.4
symspellpy>=6.7.0
texthero>=1.1.0
torch~=1.9.0
tqdm>=4.61.2
transformers>=4.8.2
wordninja>=2.0.0
yake>=0.4.8
```

- _Note: the github link in the reqs above downloads the spaCy model `en_core_web_sm` as part of the setup/installation process so you don't have to manually type `python -m spacy download en_core_web_sm` into the terminal to be able to run the code. More on this is described on spaCy's website [here](https://spacy.io/usage/models#production)_


## I tried to transcribe an audio file, and it gave me an error:

Planning to update the code to detect audio files and handle those. For now, only works on video files. If you want to try yourself, `convert_vidfile` and `convert_vid_for_transcription` just need to be updated.

## My computer crashes once it starts running the wav2vec2 model:

Try decreasing 'chunk_length' in vid2cleantxt_folder.py or vid2cleantxt_single.py (whichever you use). Until you get to really small intervals (say &lt; 10 seconds) each audio chunk can be treated as approximately independent as they are different sentences.

## How can I improve the performance of the model from a word-error-rate perspective?

You can train the model, but that requires that you already have a transcription of that person's speech already. As you may find, manual transcription is a bit of a pain and therefore transcripts are rarely provided - hence this repo.

## Why use wav2vec2 instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters\*, which in my experience can vary widely among english as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent transcription' can be made without having to spend a bunch of time testing and optimizing parameters.

Also, because it's an API you can't train it even if you wanted to, you have to be online for functionally most of the script runtime, and then of course you have privacy concerns with sending data out of your machine.

_`*` these statements reflect the assessment completed around project inception about early 2021._

# Example

Transcription of Public Domain Speeches from President John F. Kennedy

## Description

The "example_JFK_speech" folder contains the results and interim files of running both the single file and folder version. Recap:

-   for the single file version, you need to update the `input_file` variable with the filepath to the desired video file.
    -   if the path to the video file does not exist, the console should prompt the user to enter a new path
-   for the folder version, just run the .py script, and the console will prompt the user for input right away. Paste the directory path (to the video file folder), and it will handle it from there.
-   output files from already-run scripts on the examples are located in:

    1.  `vid2cleantxt\example_JFK_speech\TEST_folder_edition`
    2.  `vid2cleantxt\example_JFK_speech\TEST_singlefile`

        for the folder and single-file versions respectively.

## Output (sentence boundary disambiguation) of JFK's Moon Speech @ Rice University:

The input video was `example_JFK_speech/TEST_folder_edition/completed/GPU_President Kennedy speech on the space effort a_part_1.mp4`. The video was originally downloaded from [C-Span](https://www.c-span.org/video/?96805-52/john-f-kennedy-audio-recording):

> President pitzer minister vice president governor congressman thomas senator widely and congressman miller minister web mystery bill scientists distinguished guest of ladies and gentlemen and appreciate your president having made me an honorary visiting professor and I will assure you that my first letter will be a very. If I am delighted to be here and I am particularly delighted to be here on this occasion we meet at a college noted for knowledge in a city noted for progress in a state noted for strength and we stand in need of all three of we meet in an hour of change and challenge in a decade of hope and fear in an age of both knowledge and ignorance the greater. Our knowledge increases the greater our ignorance unfolds despite the striking fact that most of the scientists that the world has ever known are alive and working to say despite the fact that this nation's own scientific human power is doubling every twelve years in a rate of growth more than three times that of our population as a whole despite that. The vast stretches of the unknown and the unanswered and the unfinished still far outstripped are collective comprehension no man can fully grasp how far and how fast we have come but immense if you will the fifty thousand years of man's recorded history in a time span of about half a century. Stated in these terms we know very little about the first forty years except at the end of them advanced men had learned to use the skins of animals to cover them then about ten years ago under this standard man emerged from his caves to construct other kinds of shelter only five years ago man learned to write and use a cart with wheels christianity began. Less than two years ago the printing press came this year and then less than two months ago during this whole fifty years span of human history the steam engine provided a new source of power network explored the meaning of gravity last month electric lights and telephones and automobiles and airplanes became available only last week. Did we develop penecilum and television and nuclear power and now if america's new spacecraft succeeds in reaching Venus we will have literally reached the start before midnight to night this is a breath taking place and such a pace cannot help but create new ills as it dispels old new ignorance new problems new dangers. Surely the opening vistas of space promise

(continued in next video file + transcription)

## Output script run log for the "single_file" version:

For a transcription of the **President John F. Kennedy's Peace Speech.mp4** video file. See `example_JFK_speech/TEST_singlefile/v2clntxt_transcriptions/NSC + SBD` to read the full text for this.

    C:\Users\peter\AppData\Local\Programs\Python\Python39\python.exe C:/Users/peter/GIT_repos/vid2cleantxt/vid2cleantxt/vid2cleantxt_single.py
    data folder is set to `C:\Users\peter\AppData\Local\Programs\Python\Python39\lib\site-packages
    \neuspell\../data` script

     Enter full path to the video to be transcribed ---->C:\Users\peter\GIT_repos\vid2cleantxt\exa
    mple_JFK_speech\TEST_singlefile\President John F. Kennedy's Peace Speech.mp4

    Preparing to load model: facebook/wav2vec2-large-960h-lv60-self -  2021-07-13 01:09:57.762653
    Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2
    vec2-large-960h-lv60-self and are newly initialized: ['wav2vec2.masked_spec_embed']
    You should probably TRAIN this model on a down-stream task to be able to use it for prediction
    s and inference.
    Some weights of the model checkpoint at bert-base-cased were not used when initializing BertMo
    del: ['cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.predictio
    ns.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predic
    tions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relati
    onship.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on
     another task or with another architecture (e.g. initializing a BertForSequenceClassification
    model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that y
    ou expect to be exactly identical (initializing a BertForSequenceClassification model from a B
    ertForSequenceClassification model).
    Converting Video to Audio: 100%|█████████████████████████████| 55/55 [03:29<00:00,  3.80s/it]
    Finished creating audio chunks at  _01.14.09
    WARNING - unable to start CUDA. If you wanted to use a GPU, exit and check hardware.

    Gen RAM Free: 8.8 GB  | Proc size: 3.1 GB  | 8 CPUs  loaded at 34.8 % |
    No GPU being used :(
    -----------------

    Transcribing President John F. Ke... :   0%|                          | 0/55 [00:00<?, ?it/s]
    Gen RAM Free: 8.8 GB  | Proc size: 3.1 GB  | 8 CPUs  |
    No GPU being used :(
    -----------------

    2021-07-13 01:14:13.854223: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Co
    uld not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
    2021-07-13 01:14:13.867744: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above
    cudart dlerror if you do not have a GPU set up on your machine.
    Transcribing President John F. Ke... :  51%|████████▋        | 28/55 [17:51<16:56, 37.65s/it]
    Gen RAM Free: 13.6 GB  | Proc size: 1.5 GB  | 8 CPUs  loaded at 67.2 % |
    No GPU being used :(
    -----------------

    Transcribing President John F. Ke... : 100%|█████████████████| 55/55 [34:19<00:00, 37.45s/it]
    successfully moved the file President John F. Kennedy's Peace Speech.mp4 to */completed.

    ----------------------------
    transcription process completed -  2021-07-13 01:48:29.504284

    top 5 phrases are:

    ['nations world peace',
     'war total war',
     'interests nuclear powers',
     'world security system',
     'peace corps abroad']
    Transcription files used to extract KW can be found in:
      C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcript
    ions\neuspell_sc


    ----------------------------------- Script Complete -------------------------------
    2021-07-13 01:49:20.620811
    Transcription files + more in folder:
     C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcripti
    ons
    Metadata for each transcription located @
     C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transc_meta
    data
    total runtime was 40 minutes

    Process finished with exit code 0

* * *

# Future Work, Collaboration, & Citations

## Project Updates

A _rough_ timeline of what has been going on in the repo:

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

## What about a version where I don't need python at all?

Plan to do this eventually [py2exe](https://www.py2exe.org/). Currently backlogged - will update repo when complete.

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
