# vid2cleantxt

**vid2cleantxt**: a [transformers-based](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) pipeline for turning
heavily speech-based video files into clean, readable text from the audio.

TL;DR check
out [this Colab script](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing)
to see a transcription and keyword extraction of a speech by John F. Kennedy by simply running all cells.

** **
**Table of Contents**
<!-- TOC -->

- [vid2cleantxt](#vid2cleantxt)
- [Motivation](#motivation)
- [Overview](#overview)
  - [Example Output](#example-output)
  - [Pipeline Intro](#pipeline-intro)
- [Installation](#installation)
  - [How to get this to work on your machine](#how-to-get-this-to-work-on-your-machine)
  - [Is there a jupyter notebook file?](#is-there-a-jupyter-notebook-file)
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
  - [I tried to transcribe an audio file, and it gave me an error:](#i-tried-to-transcribe-an-audio-file-and-it-gave-me-an-error)
  - [My computer crashes once it starts running the wav2vec2 model:](#my-computer-crashes-once-it-starts-running-the-wav2vec2-model)
  - [How can I improve the performance of the model from a word-error-rate perspective?](#how-can-i-improve-the-performance-of-the-model-from-a-word-error-rate-perspective)
  - [Why use wav2vec2 instead of SpeechRecognition or other transcription methods?](#why-use-wav2vec2-instead-of-speechrecognition-or-other-transcription-methods)
- [Example](#example)
  - [Description](#description)
  - [Output (sentence boundary disambiguation) of the single file version:](#output-sentence-boundary-disambiguation-of-the-single-file-version)
  - [Output script run log for the "single_file" version:](#output-script-run-log-for-the-single_file-version)
- [Future Work, Collaboration, & Citations](#future-work-collaboration--citations)
  - [Project Updates](#project-updates)
  - [Future Work](#future-work)
  - [What about a version where I don't need python at all?](#what-about-a-version-where-i-dont-need-python-at-all)
  - [I've found x repo / script / concept that I think you should incorporate or collaborate with the author.](#ive-found-x-repo--script--concept-that-i-think-you-should-incorporate-or-collaborate-with-the-author)
  - [Citations](#citations)

<!-- /TOC -->
** **

# Motivation

When compared to other media (such as text and pictures), video (specifically the audio) is an inefficient way to convey
dense or technical information, as in the base case the viewer has to sit through the whole thing, while only part of
the video may be relevant to them. Even worse, if you don't understand a statement or concept, you have to search
through the video, or re-watch it, taking up significant amounts of time. This project attempts to help solve that
problem by converting long video files into text that you can read, CTRL+F, keyword extract, and summarize.

# Overview

## Example Output

Example output text of a video transcription of [JFK's peace speech](https://youtu.be/0fkKnfk4k40):

	president anderson members of the faculty board of trustees distinguished guests my old colleague senator bob bird to has earned his degree through many years of
	attending night for school while i am earning mine in the next thirty minutes distinguished guests ladies and gentlemen it is with great pride that i participate
	in this ceremony of the american university.
	[ ... ]
	there are few earthly things more beautiful than a university wrote john mansfield in his tribute to english universities and his words are equally true
	to day he did not refer to towers or tho campuses he admired the splendid beauty of a university because it was he said a place were those who hate ignorance may
	strive to know were those who perceive truth. may strive to make others i i have therefore chosen this time and place to discuss a topic on which ignorance too
	often abounds but the truth too rarely perceived and that is the most important topic on earth peace what kind of peace do i mean and what kind of peace do were
	not a packs americana enforced on the world by american weapons of war not the peace of the grave or the security of the slave. i am talking about genuine peace
	the kind of peace that makes life on earth worth living be the kind that enables men and nations to grow and to hope and build a better life for their children
	not merely peace for americans but peace for all men and women not merely peace in our time but peace in all time i speak of peace because of the new face of war
	total war makes no sense in an age where. great powers can maintain large and relatively invulnerable nuclear forces and refuse to surrender without resort to
	those forces it makes no sense in an age were a single nuclear weapon contains almost ten times the explosive force delivered by all the allied air forces in the
	second world war it makes no sense in an age when the deadly poisons produced by a nuclear exchange. would be carried by wind and water and soil and seed to the
	far corners of the globe and he generations yet unborn to day the expenditure of billions of dollars every year on weapons acquired for the purpose of making sure
	we never need them is essential to the keeping of peace but surely the acquisition of such idle stock piles which can only destroy and never create is not the only
	much. as the most efficient means of assuring peace i speak of peace therefore as the necessary rational end of rational men i realize the pursuit of peace is not
	as dramatic as the pursuit of war and frequently the words of the pursuers fall on deaf ears but we have no more urgent task some say that it is useless to speak
	of peace or world law or world disarmament...

See the examples folder for more detail / full transcript.

## Pipeline Intro

Here's a high-level overview of what happens in the ```vid2cleantxt_folder.py``` script to create the output shown above:

1. Imports relevant packages, and imports relevant functions from audio2text_functions.py
2. Receive **directory** string input from user in "script run window*. Then iterates through that directory, and finds all video files
3. FOR each video file found:
    - convert video to .wav format audio chunks of duration X** seconds with MoviePy
    - transcribe all X audio chunks through a [pretrained wav2vec2 model](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) 
      (in this repo - using huggingface transformers) and store the resulting text in a list
    - write all results of the list into a text file, store various runtime metrics into a separate text list
    - Delete .wav audio chunk directory after completed using them
4. Next, create two new text files: one with all transcriptions appended, one with all metadata appended.
5. Then FOR each transcription text file:
    - Pass the 'base' transcription text through a spell checker (symspellpy) and autocorrect spelling. 
      save as new text file.
    - Use pySBD to infer sentence boundaries on the spell-corrected text and add periods in to delineate sentences. 
      Save as new file.
    - Run basic keyword extraction (via YAKE) on spell-corrected file. All keywords per file are stored in one dataframe 
      for comparison, and exported to .xlsx format.
6. cleanup tasks, report runtime, and exit.

_* the 'single file' version needs to have the name defined in the python code itself_

_** (where X is some duration that does not overload your computer or crash your IDE)_

By default,

- results are stored in ~/directory/w2v2_video_transcriptions
- metadata in ~/directory/w2v2_transcription_metadata

(where **directory** is path entered by user)

# Installation

## How to get this to work on your machine

**Important** the first time the code runs on your machine, it will download the pretrained transformers model (~1 gb).
After the first run, it will be cached locall, and you will not need to sit through that again.

1. fastest:
```
git clone https://github.com/pszemraj/vid2cleantxt.git
pip install -r requirements.txt
python -m vid2cleantxt_folder.py
```

2. Using github desktop - run vid2cleantxt_folder.py from your IDE
3. If neither of those are convenient, see the next section on how to use Colab 

## Is there a jupyter notebook file?

No, but there are versions of these scripts on Google Colab. From Colab you can download as .ipynb, but you may need to
make some small changes (some directories, packages, etc. are specific to Colab's structure). Links to Colab Scripts:

1. Single-File Version (Implements GPU)
    * Link [here](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing)
    * This script downloads the video from a public link to one of the JFK videos stored on my Google Drive. As such, no
      authentication / etc. is required and **this link is recommended for seeing how this pipeline works**.
    * The only steps required are checking / adjusting the runtime to a GPU, and *Run All*
2. Multi-File Version (Implements GPU)
    * Link [here](https://colab.research.google.com/drive/1UMCSh9XdvUABjDJpFUrHPj4uy3Cc26DC?usp=sharing)
    * This script connects to the user's google drive to convert a whole folder of videos using Google's Colab Python
      package.
    * It **does require the video files to be hosted on the user's drive**, as well as authorization of Colab (it will
      prompt you and walk you through this)

New to Colab? Some links I found useful:

- [Google's FAQ](https://research.google.com/colaboratory/faq.html)
- [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
- [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
- [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

## How long does this take to run?

On Google Colab with a 16 gb GPU (should be available to free Colab accounts): approximately 8 minutes to transcribe ~90
minutes of audio.

On my machine (CPU only due to AMD GPU) it takes approximately 80-120% of the total duration of input video files.

**Specs:**

```
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

> *Note:* if you change ```wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self"``` to ```wav2vec2_model = "
facebook/wav2vec2-base-960h"```the runtime will be considerably faster. I do not have stats on differences in WER, but
[facebook](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) may have some posted.

---

# Application

## Now I have a bunch of long text files. How are these useful?

short answer: noam_chomsky.jpeg

more comprehensive answer:

A large corpus of text can be visualized / summarized / reduced in many ways with natural language processing / typical
machine learning algorithms (i.e., classifying text, so on). Some packages to check out regarding this are TextHero and
ScatterText. An example use case is combining the text from audio transcriptions with written documents (say textbooks
or lecture slides converted to text) for comparison of how similar they are. You can also use topic models (available in
ScatterText and many other packages) or statistical models (YAKE) to extract key topics from each file (or file group)
and compare those (how they change over time, what are the key topics in practice exam PDF files, etc).

### Visualization and Analysis

1. [TextHero](https://github.com/jbesomi/texthero) - cleans text, allows for visualization / clustering (k-means) /
   dimensionality reduction (PCA, TSNE)
    * Use case here: I want to see how this speaker's speeches differ from each other. Which are "
      the most related"?
3. [Scattertext](https://github.com/JasonKessler/scattertext) - allows for comparisons of one corpus of text to another
   via various methods and visualizes them.
    * Use case here: I want to see how the speeches by this speaker compare to speeches by speaker B in terms of topics,
      word frequency… so on

Some examples from my own usage are illustrated below from both packages.

### Text Extraction / Manipulation

1. [Textract](https://textract.readthedocs.io/)
2. [Textacy](https://github.com/chartbeat-labs/textacy)
3. [YAKE](https://github.com/LIAAD/yake)
    * A brief YAKE analysis is completed in this pipeline after transcribing the audio.

### Text Summarization

Several options are available on the [HuggingFace website](https://huggingface.co/models?pipeline_tag=summarization). I
have personally found Google's T5 to be most effective for "lecture-esque" video conversion.

I personally use several similar methods in combination with the transcription script, however it isn't in a place to
posted yet. It will be posted to a public repo on this account when ready.

## TextHero example use case

Clustering vectorized text files into k-means groups:

![iml Plotting with TSNE + USE, Colored on Directory Name](https://user-images.githubusercontent.com/74869040/110546335-a0baaf80-812e-11eb-8d7d-48da00989dce.png)

![iml Plotting with TSNE + USE, Colored on K-Means Cluster](https://user-images.githubusercontent.com/74869040/110546452-c6e04f80-812e-11eb-9a4b-03213ec4a63b.png)

## ScatterText example use case

Comparing frequency of terms in one body of text vs. another

![ST P 1 term frequency I ML 2021 Docs I ML Prior Exams_072122_](https://user-images.githubusercontent.com/74869040/110546149-69e49980-812e-11eb-9c94-81fcb395b907.png)

** **

---

# Design Choices & Troubleshooting

## What python package dependencies does this repo have?

Upon cloning the repo, run the command `` pip install -r requirements.txt`` in a terminal opened in the project
directory. As of July 8th, 2021, requirements are as follows:

```
librosa==0.8.1
moviepy==1.0.3
natsort==7.1.1
pandas==1.3.0
pysbd==0.3.4
symspellpy==6.7.0
texthero==1.1.0
torch~=1.9.0
transformers==4.8.2
wordninja==2.0.0
yake==0.4.8
pyspellchecker>=0.6.2
spellchecker>=0.4
openpyxl >=3
tqdm>=4.61.2
```

## I tried to transcribe an audio file, and it gave me an error:

Planning to update the code to detect audio files and handle those. For now, only works on video files. If you want to
try yourself, convert_vidfile and convert_vid_for_transcription just need to be updated.

## My computer crashes once it starts running the wav2vec2 model:

Try decreasing 'chunk_length' in vid2cleantxt_folder.py or vid2cleantxt_single.py (whichever you use). Until you get to
really small intervals (say < 10 seconds) each audio chunk can more or less be treated independently as they are
different sentences.

## How can I improve the performance of the model from a word-error-rate perspective?

You can train the model, but that requires that you already have a transcription of that person's speech already. As you
may find, manual transcription is a bit of a pain and therefore transcripts are rarely provided - hence this repo.

## Why use wav2vec2 instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters, which in my experience
can vary widely among english as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent
transcription' can be made without having to spend a bunch of time testing and optimizing parameters.

Also, because it's an API you can't train it even if you wanted to, you have to be online for functionally most of the
script runtime, and then of course you have privacy concerns with sending data out of your machine.

# Example

Transcription of Public Domain Speeches from President John F. Kennedy

## Description

The "example_JFK_speech" folder contains the results and interim files of running both the single file and folder
version. Recap:

- for the single file version, you need to update the ```input_file``` variable with the filepath to the
  desired video file.
    - if the path to the video file does not exist, the console should prompt the user to enter a new path
- for the folder version, just run the .py script, and the console will prompt the user for input right away. 
  Paste the directory path (to the video file folder), and it will handle it from there.
- output files from already-run scripts on the examples are located in 
  ```vid2cleantxt\example_JFK_speech\TEST_folder_edition```  and  
  ```vid2cleantxt\example_JFK_speech\TEST_singlefile``` for the folder and single-file versions respectively.
  
## Output (sentence boundary disambiguation) of JFK's Moon Speech @ Rice University:

Input video was JFK_rice_moon_speech.mp4. Originally downloaded
from [C-Span](https://www.c-span.org/video/?96805-52/john-f-kennedy-audio-recording):

```
transcription of rice moon speech mp4 at date 09 07 2021. surely the opening vistas of space 
promise high costs and hardships as well as high reward so it is not surprising that some would have us stay where  
we are a little longer to rest to wait but this city of huston this state of texas this country of the united states 
was not built by those who waited and rested but if i were to say my fellow citizens. that we shall send to the moon 
two hundred  and forty thousand miles away from the control station in houston a giant rocket more than three 
hundred feet tall the  length of this football field made of new metal alloys some of which have not yet been 
invented  capable of standing heat and stresses several times more than have ever been experienced fitted together 
with  a precision better than the. test watch carrying all the equipment needed for propulsion guidance control  
communications food and survival on an untried mission to an unknown celestial body and then return it safely to earth 
re entering the atmosphere at speeds of over twenty five thousand miles per hour causing heat about half that on the
temperature of the sun almost as hot as it is here to day and do all this. and do all this and do it right and do it 
first before this dictate is out then we must be to i'm the one who is doing all the work so to semi to stay cool for 
a minute however i think were going to do it and i think that we must pay what needs to be paid i don't think we ought 
to waste any. money but i think we ought to do the job and this will be done in the decade of the sixty it may be done
while some of you are still here at school at this college and university it will be done during the terms of office of
some of the people who sit here on this platform but it will be done many years ago the great british explorer george 
mallory who was to die on mount everest was asked why did he want to climb it he said because it is there well space is 
there. and were going to climb it and the moon and the planets are there and new hopes for knowledge and peace are 
there and therefore as we set sail we ask gods blessing on the most hazardous and dangerous and greatest adventure on 
which man has ever embarked thank you
```

## Output script run log for the "single_file" version:

A transcription of the **President John F. Kennedy's Peace Speech.mp4** video file. The output of this is shown at the 
top of the README here, or the .txt can be read (in the examples folder).

```
C:\Users\peter\AppData\Local\Programs\Python\Python39\python.exe C:/Users/peter/GIT_repos/vid2cleantxt/vid2cleantxt/vid2cleantxt_single.py
Will transcribe the file President John F. Kennedy's Peace Speech.mp4 stored in:
 C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile


Preparing to load model: facebook/wav2vec2-large-960h-lv60-self
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2vec2-large-960h-lv60-self and are newly initialized: ['wav2vec2.masked_spec_embed']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Converting Video to Audio: 100%|██████████| 55/55 [02:18<00:00,  2.51s/it]
Finished creating audio chunks at  _02.14.22
Transcribing President John F. Ke...: 100%|██████████| 55/55 [26:38<00:00, 29.07s/it]

 Starting to spell-correct and extract keywords

PySymSpell - Starting to check and correct the file:  President John F Kennedy ' Peace Speec_tscript__02.41.01.txt
RT for this file was 0.147500 minutes
Finished correcting w/ symspell President John F Kennedy ' Peace Speec_tscript__02.41.01.txt  at time:  02:41:10

Top Key Phrases from YAKE, with max n-gram length:  3

                  key_phrase  ...  freq_in_text
0        nations world peace  ...             1
1              war total war  ...             1
2   interests nuclear powers  ...             1
3      world security system  ...             1
4         soviet union adopt  ...             1
5         peace corps abroad  ...             1
6     ends americans weapons  ...             1
7  safeguard human interests  ...             1
8       kennedy peace speech  ...             1
9       cold war remembering  ...             1

[10 rows x 4 columns]


----------------------------------- Script Complete -------------------------------
Transcription file + more can be found here:  C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile\w2v2_video_transcriptions
Metadata for each transcription is located:  C:\Users\peter\GIT_repos\vid2cleantxt\example_JFK_speech\TEST_singlefile\w2v2_transcription_metadata
total runtime was 29.345329  minutes

Process finished with exit code 0
```
---

# Future Work, Collaboration, & Citations

## Project Updates

A *rough* timeline of what has been going on in the repo:

- July 2021: Python scripts and 
- April - June: Work done mostly on Colab improving saving, grammar correction, etc. 
- March 2021: public repository added
## Future Work

1. syncing improvements currently in the existing **Google Colab** notebooks (links) above such
   as [NeuSpell](https://github.com/neuspell/neuspell)
    - this will include support for CUDA automatically when running the code (currently just on Colab)
2. Include additional features that are currently not public:
    - T5 Summarization of transcribed videos
    - Paragraph Disambiguation in both transcription & summarization
    - T5 Summarization
3. py2exe (once code optimized)

## What about a version where I don't need python at all?

Plan to do this eventually [py2exe](https://www.py2exe.org/). Currently backlogged - will update repo when complete.

## I've found x repo / script / concept that I think you should incorporate or collaborate with the author.

Send me a message / start a discussion! Always looking to improve.

## Citations

wav2vec2 (fairseq)

- Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq:
  A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations,

2019.

- repo [link](https://github.com/pytorch/fairseq)

MoviePy

- [link](https://github.com/Zulko/moviepy) to repo as no citation info given

symspellpy / symspell

- repo [link](https://github.com/mammothb/symspellpy/tree/e7a91a88f45dc4051b28b83e990fe072cabf0595)
- copyright:

  Copyright (c) 2020 Wolf Garbe Version: 6.7 Author: Wolf Garbe <wolf.garbe@seekstorm.com>
  Maintainer: Wolf Garbe <wolf.garbe@seekstorm.com>
  URL: https://github.com/wolfgarbe/symspell
  Description: https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f

  MIT License

  Copyright (c) 2020 Wolf Garbe

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
  persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
  Software.

  https://opensource.org/licenses/MIT

YAKE (yet another keyword extractor)

- repo [link](https://github.com/LIAAD/yake)
- relevant citations:
  In-depth journal paper at Information Sciences Journal

  Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword
  Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp
  257-289. pdf

  ECIR'18 Best Short Paper

  Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). A Text Feature Based Automatic
  Keyword Extraction Method for Single Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances
  in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772,
  pp. 684 - 691. pdf

  Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). YAKE!
  Collection-independent Automatic Keyword Extractor. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds).
  Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol
  10772, pp. 806 - 810. pdf
