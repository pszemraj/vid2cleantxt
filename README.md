# vid2cleantxt

vid2cleantxt: a pipeline for turning heavily speech-based video files into clean, readable text from the audio.

TL;DR check out [this Colab script](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing) to see a trancription and keyword extraction of a JFK speech by simply running all cells.

*Note: this is a work-in-progress, and my first 'real' repo. All feedback is welcome!*

** **
**Table of Contents**
<!-- TOC -->

- [vid2cleantxt](#vid2cleantxt)
- [Motivation](#motivation)
- [Basic overview of pipeline](#basic-overview-of-pipeline)
- [Installation](#installation)
  - [How to get this to work on your machine](#how-to-get-this-to-work-on-your-machine)
  - [Is there a jupyter notebook file?](#is-there-a-jupyter-notebook-file)
  - [How long does this take to run?](#how-long-does-this-take-to-run)
- [Application](#application)
  - [Now I have a bunch of long text files. How are these useful?](#now-i-have-a-bunch-of-long-text-files-how-are-these-useful)
    - [Visualization and Analysis:](#visualization-and-analysis)
    - [Text Extraction / Manipulation:](#text-extraction--manipulation)
    - [Text Summarization:](#text-summarization)
  - [TextHero example use case:](#texthero-example-use-case)
  - [ScatterText example use case:](#scattertext-example-use-case)
- [Design Choices & Troubleshooting](#design-choices--troubleshooting)
  - [What python package dependencies does this repo have?](#what-python-package-dependencies-does-this-repo-have)
  - [I tried to transcribe an audio file and it gave me an error:](#i-tried-to-transcribe-an-audio-file-and-it-gave-me-an-error)
  - [My computer crashes once it starts running the wav2vec2 model:](#my-computer-crashes-once-it-starts-running-the-wav2vec2-model)
  - [How can I improve the performance of the model from a word-error-rate perspective?](#how-can-i-improve-the-performance-of-the-model-from-a-word-error-rate-perspective)
  - [Why use wav2vec2 instead of SpeechRecognition or other transcription methods?](#why-use-wav2vec2-instead-of-speechrecognition-or-other-transcription-methods)
- [Example](#example)
  - [Description](#description)
  - [Output (sentence boundary disambiguation) of the single file version:](#output-sentence-boundary-disambiguation-of-the-single-file-version)
  - [Output script run log for the "single_file" version:](#output-script-run-log-for-the-single_file-version)
- [Future Work, Collaboration, & Citations](#future-work-collaboration--citations)
  - [I've found x repo / script / concept that I think you should incorporate or collaborate with the author.](#ive-found-x-repo--script--concept-that-i-think-you-should-incorporate-or-collaborate-with-the-author)
  - [Future Work](#future-work)
  - [What about a version where I don't need python at all?](#what-about-a-version-where-i-dont-need-python-at-all)
  - [Citations (work in progress)](#citations-work-in-progress)

<!-- /TOC -->
** **

# Motivation

When compared to other media (such as text and pictures), video (specifically the audio) is an inefficient way to convey dense or technical information, as in the base case the viewer has to sit through the whole thing, while only part of the video may be relevant to them. Even worse, if you don't understand a statement or concept, you have to search through the video, or rewatch it, taking up significant amounts of time. This repo attempts to help solve that problem by converting long video files into text that you can read, CTRL+F, keyword extract, and summarize.

# Basic overview of pipeline

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

Here's a high-level overview of what happens in the vid2cleantxt_folder.py script to create the output shown above:

	- Imports relevant packages, and imports relevant functions from audio2text_functions.py
	- Receive **directory** string input from user in "script run window*. Then iterates through that directory, and finds all video files
	- FOR each video file found:
		- convert video to .wav format audio chunks of duration X** seconds with MoviePy
		- transcribe all X audio chunks through a pretrained wav2vec2model (transformers), store results in a list
		- write all results of the list into a text file, store various runtime metrics into a separate text list
		- Deletes .wav audio chunks after completed using them
	- Next, create two new text files: one with all transcriptions appended, one with all metadata appended.
	- Then FOR each transcription text file:
		- Pass the 'base' transcription text through a spell checker (symspellpy) and autocorrect spelling. save as new text file
		- Use pySBD to infer sentence boundaries on the spell-corrected text and add periods in to delineate sentences. Save as new file
		- Run basic keyword extraction (via YAKE) on spell-corrected file. All keywords per file are stored in one dataframe for comparison, and exported to .xlsx format
	- cleanup tasks (delete the X .wav files created for audio transcription, etc), report runtime, and exit

	* the 'single file' version needs to have the name defined in the python code itself
	** (where X is some duration that does not overload your computer or crash your IDE)


* results are stored in ~/directory/w2v2_video_transcriptions
* metadata in ~/directory/w2v2_transcription_metadata

(where **directory** is path entered by user)

# Installation

## How to get this to work on your machine

**Important** the first time the code runs on your machine, it will download the pretrained transformers model (approx 1 gb). After the first run, it will be cached locally on your machine, and you will not need to sit through that again.

* Currently, just normal Git methods. You can also download the .zip from the site, if you run the scripts in the same configuration it should work fine.

* Will make some changes that don't require vid2cleantxt_folder to be in the same folder as audio2text_functions for example.

## Is there a jupyter notebook file?

No, but there are versions of these scripts on Google Colab. From Colab you can download as .ipynb, but you may need to make some small changes (some directories, packages, etc. are specific to Colab's structure). Links to Colab Scripts:

1. Single-File Version (Implements GPU)
	* Link [here](https://colab.research.google.com/drive/1WfJ1yQn-jtyZsoQXdzXx91FPbLo5t7Mg?usp=sharing)
	* This script downloads the video from a public link to one of the JFK videos stored on my Google Drive. As such, no authentication  / etc. is required and **this link is recommended for seeing how this pipeline works**.
	* The only steps required are checking / adjusting the runtime to a GPU, and *Run All*
2. Multi-File Version (Implements GPU)
	* Link [here](https://colab.research.google.com/drive/1UMCSh9XdvUABjDJpFUrHPj4uy3Cc26DC?usp=sharing)
	* This script connects to the user's google drive to convert a whole folder of videos using Google's Colab Python package.
	* It **does require the video files to be hosted on the user's drive**, as well as authorization of Colab (it will prompt you and walk you through this)

New to Colab? Some links I found useful:
- [Google's FAQ](https://research.google.com/colaboratory/faq.html)
- [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
- [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
- [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)


## How long does this take to run?

On Google Colab with a 16 gb GPU (should be available to free Colab accounts): approximately 8 minutes to transcribe ~90 minutes of audio.

On my machine (CPU only due to AMD GPU) it takes approximately 80-120% of the total duration of input video files.

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

*Note:* if you change wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self" to wav2vec2_model = "facebook/wav2vec2-base-960h" the runtime will be considerably faster. I have not compared accuracy on my own use cases, but I am sure Facebook has.

# Application

## Now I have a bunch of long text files. How are these useful?

short answer: noam_chomsky.jpeg

more comprehensive answer:

A large corpus of text can be visualized / summarized / reduced in many ways with natural language processing / typical machine learning algorithms (i.e. classifying text, so on). Some packages to check out regarding this are TextHero and ScatterText. An example use case is combining the text from audio transcriptions with written documents (say textbooks or lecture slides converted to text) for comparison of how similar they are. You can also use topic models (available in ScatterText and many other packages) or statistical models (YAKE) to extract key topics from each file (or file group) and compare those (how they change over time, what are the key topics in practice exam PDF files, etc).

### Visualization and Analysis

1. [TextHero](https://github.com/jbesomi/texthero) - cleans text, allows for visualization / clustering (k-means) / dimensionality reduction (PCA, TSNE)
	* Use case here: I want to see how this speaker's speeches differ from each other. Which are "the most related"?
3. [Scattertext](https://github.com/JasonKessler/scattertext) - allows for comparisons of one corpus of text to another via various methods and visualizes them.
	* Use case here: I want to see how the speeches by this speaker compare to speeches by speaker B in terms of topics, word frequency.. so on

Some examples from my own usage are illustrated below from both packages.

### Text Extraction / Manipulation

1. [Textract](https://textract.readthedocs.io/)
2. [Textacy](https://github.com/chartbeat-labs/textacy)
3. [YAKE](https://github.com/LIAAD/yake)
	* A brief YAKE analysis is completed in this pipeline after transcribing the audio.

### Text Summarization

Several options are available on the [HuggingFace website](https://huggingface.co/models?pipeline_tag=summarization). I have personally found Google's T5 to be most effective for "lecture-esque" video conversion.

I personally use several similar methods in combination with the transcription script, however it isn't in a place to post yet. It will be posted to a public repo on this account when ready.

## TextHero example use case

Clustering vectorized text files into k-means groups

![iml Plotting with TSNE + USE, Colored on Directory Name](https://user-images.githubusercontent.com/74869040/110546335-a0baaf80-812e-11eb-8d7d-48da00989dce.png)

![iml Plotting with TSNE + USE, Colored on K-Means Cluster](https://user-images.githubusercontent.com/74869040/110546452-c6e04f80-812e-11eb-9a4b-03213ec4a63b.png)

## ScatterText example use case

Comparing frequency of terms in one body of text vs. another

![ST P 1 term frequency I ML 2021 Docs I ML Prior Exams_072122_](https://user-images.githubusercontent.com/74869040/110546149-69e49980-812e-11eb-9c94-81fcb395b907.png)

** **
# Design Choices & Troubleshooting

## What python package dependencies does this repo have?

Contexts of requirements.txt as of April 25:
```
librosa==0.8.0
moviepy==1.0.3
natsort==7.1.1
pandas==1.0.5
pysbd==0.3.4
symspellpy==6.7.0
texthero==1.0.9
torch==1.7.1
transformers==4.3.2
wordninja==2.0.0
yake==0.4.3

setuptools~=51.3.1
```


## I tried to transcribe an audio file and it gave me an error:

Planning to update the code to detect audio files and handle those. For now, only works on video files. If you want to try yourself, convert_vidfile and convert_vid_for_transcription just need to be updated.

## My computer crashes once it starts running the wav2vec2 model:

Try decreasing 'chunk_length' in vid2cleantxt_folder.py or vid2cleantxt_single.py (whichever you use). Until you get to really small intervals (say < 10 seconds) each audio chunk can more or less be treated independently as they are different sentences.

## How can I improve the performance of the model from a word-error-rate perspective?

You can train the model, but that requires that you already have a transcription of that person's speech already. As you may find, manual transcription is a bit of a pain and therefore transcripts are rarely provided - hence this repo.

## Why use wav2vec2 instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters, which in my experience can vary widely among english as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent transcription' can be made without having to spend a bunch of time testing and optimizing parameters.

Also, because it's an API you can't train it even if you wanted to, you have to be online for functionally most of the script runtime, and then of course you have privacy concerns with sending data out of your machine.

# Example

Transcription of Public Domain Speeches from President John F. Kennedy

## Description

The "example_JFK_speech" folder contains the results and interim files of running both the single file and folder version. Recap:
* for the single file version, you need to update two variables in the .py script representing the file directory and name. Then you run it
* for the folder version, you just run the .py script, and it will prompt you for input. Paste the directory path (to the video files) and it will handle it from there.

**FYI WITH RESPECT TO THE EXAMPLE:** I had to split "GPU_President Kennedy speech on the space effort a" when pushing to Git due to file size constraints. As such, there are 5 video files (Parts 1-5) in the example for this speech but when I ran the folder transcriber it was just one. Again, this shouldn't really be an issue as the audio is "independent" in practicality as explained earlier

## Output (sentence boundary disambiguation) of the single file version:

Input video was JFK_rice_moon_speech.mp4. Originally downloaded from [C-Span](https://www.c-span.org/video/?96805-52/john-f-kennedy-audio-recording):

	transcription of of rice moon speech mp4 at date 09 03 2021 time 22 07 41. surely the opening vistas of space promise high costs and hardships as well as
	high reward so it is not surprising that some would have us stay where we are a little longer to rest to wait but this city of huston this state of texas
	this country of the united states was not built by those who waited and rested but if i were to say my fellow citizens that we shall send to the moon two
	hundred and forty thousand. a away from the control station in huston a giant rocket more than three hundred feet tall the length of this football field
	made of new metal alloys some of which have not yet been invented capable of standing heat and stresses several times more than have ever been experienced
	fitted together with a precision better than the finest watch carrying all the equipment needed for propulsion guidance control communications food and
	survival on a. tried mission to an unknown celestial body and then return it safely to earth re entering the atmosphere at speeds of over twenty five
	thousand miles per hour causing heat about half that on the temperature of the sun almost as hot as it is here to day and do all this and do all this and
	do it right and do it first before the dictator out and be. i'm the one who is doing all the work to stay col for a minute however i think were going to
	do it and i think that we must pay what needs to be paid i don't think we ought to waste any money but i think we ought to do the job and this will be done
	in the decade of the sixties it may be done while some of you are still here at school at this college university it will be done during the terms of office
	of some of the people who sit here on this platform but it will be done many years ago the great british explorer george mallory. who was to die on mount everest
	was asked why did he want to climb it he said because it is there or space is there and we are going to climb it and the moon and the planets are there and new
	hopes for knowledge and peace are there and therefore as we set sail we ask gods blessing on the most hazardous and dangerous and greatest adventure on which
	man has ever embarked a

## Output script run log for the "single_file" version:

	C:\Users\peter\AppData\Local\Microsoft\WindowsApps\python.exe C:/Users/peter/PycharmProjects/vid2cleantxt/vid2cleantxt/vid2cleantxt_single.py
	2021-03-09 22:07:21.289433: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
	2021-03-09 22:07:21.289808: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

	Preparing to load model: facebook/wav2vec2-large-960h-lv60-self

	============================================================
	Converting video to audio for file:  JFK_rice_moon_speech.mp4
	============================================================

	convert vid2audio: 100%
	55/55 [02:24<00:00, 2.63s/it]

	Finished creating audio chunks at  _22.14.44

	============================================================
	converted video to audio. About to start transcription loop for file:  JFK_rice_moon_speech.mp4
	============================================================

	Cuda availability (PyTorch):  True

	Gen RAM Free: 25.3 GB  | Proc size: 4.5 GB
	GPU RAM Free: 14143MB | Used: 2137MB | Util  13% | Total 16280MB

	Converting Video: 100%
	55/55 [01:55<00:00, 2.09s/it]
	Cuda availability (PyTorch):  True

	Gen RAM Free: 25.3 GB  | Proc size: 4.5 GB
	GPU RAM Free: 14143MB | Used: 2137MB | Util  13% | Total 16280MB

	Cuda availability (PyTorch):  True

	Gen RAM Free: 25.3 GB  | Proc size: 4.5 GB
	GPU RAM Free: 14143MB | Used: 2137MB | Util  13% | Total 16280MB

	Cuda availability (PyTorch):  True

	Gen RAM Free: 25.3 GB  | Proc size: 4.6 GB
	GPU RAM Free: 14143MB | Used: 2137MB | Util  13% | Total 16280MB

	Cuda availability (PyTorch):  True

	Gen RAM Free: 25.3 GB  | Proc size: 4.6 GB
	GPU RAM Free: 14143MB | Used: 2137MB | Util  13% | Total 16280MB



	Finished audio transcription of JFK_rice_moon_speech.mp4 and now saving metrics.

	Deleted Audio Chunk Folder + Files

	Finished transcription successfully for JFK_rice_moon_speech.mp4 at date_22_05_2021_time_22-16-39
	completed transcription in 2.4121029416720075 minutes

	 Starting to spell-correct and extract keywords

	PySymSpell - Starting to check and correct the file:  JFK rice moon speec_tscript__22.10.58.txt
	loaded text with      6 lines
	RT for this file was 0.085566 minutes
	Finished correcting w/ symspell JFK rice moon speec_tscript__22.10.58.txt  at time:  22:11:04

	Top Key Phrases from YAKE, with max n-gram length:  3

							key_phrase  ...  freq_in_text
	0                 rice moon speech  ...             1
	1               promise high costs  ...             1
	2                hundred feet tall  ...             1
	3  guidance control communications  ...             1
	4                hour causing heat  ...             1
	5              football field made  ...             1
	6            finest watch carrying  ...             1
	7           unknown celestial body  ...             1
	8           great british explorer  ...             1
	9                      high reward  ...             1

	[10 rows x 4 columns]


	----------------------------------- Script Complete -------------------------------
	Transcription file + more can be found here:  C:\Users\peter\PycharmProjects\vid2cleantxt\example_JFK_speech\wav2vec2_sf_transcript
	Metadata for each transcription is located:  C:\Users\peter\PycharmProjects\vid2cleantxt\example_JFK_speech\wav2vec2_sf_metadata
	total runtime was 3.600498  minutes

	Process finished with exit code 0
# Future Work, Collaboration, & Citations

## I've found x repo / script / concept that I think you should incorporate or collaborate with the author.

Send me a message / start a discussion! Always looking to improve.

## Future Work
1. Update spell correction to use [NeuSpell](https://github.com/neuspell/neuspell) so BERT can get involved
2. Add additional features
	- PDF gen
	- Paragraph Disambiguation 
	- T5 Summarization
3. py2exe (once code optimized)

## What about a version where I don't need python at all?

Work in progress with [py2exe](https://www.py2exe.org/). Will update repo when complete.


## Citations

wav2vec2

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
- repo [link](https://github.com/pytorch/fairseq)

MoviePy

- [link](https://github.com/Zulko/moviepy) to repo as no citation info given

symspellpy / symspell

	https://github.com/mammothb/symspellpy/tree/e7a91a88f45dc4051b28b83e990fe072cabf0595

	Copyright (c) 2020 Wolf Garbe
	Version: 6.7
	Author: Wolf Garbe <wolf.garbe@seekstorm.com>
	Maintainer: Wolf Garbe <wolf.garbe@seekstorm.com>
	URL: https://github.com/wolfgarbe/symspell
	Description: https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f

	MIT License

	Copyright (c) 2020 Wolf Garbe

	Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
	documentation files (the "Software"), to deal in the Software without restriction, including without limitation
	the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
	and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	https://opensource.org/licenses/MIT

YAKE (yet another keyword extractor)
- repo [link](https://github.com/LIAAD/yake)

	In-depth journal paper at Information Sciences Journal

	Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289. pdf

	ECIR'18 Best Short Paper

	Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). A Text Feature Based Automatic Keyword Extraction Method for Single Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 684 - 691. pdf

	Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). YAKE! Collection-independent Automatic Keyword Extractor. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 806 - 810. pdf
