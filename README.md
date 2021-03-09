# vid2cleantxt

vid2cleantxt: a pipeline for turning heavily speech-based video files into clean, readable text from the audio.

	Note: this is a work-in-progress, and my first 'real' repo. 
	As such, code isn't fully optimized or following software eng norms yet, but will continue to be improved over time.
	All feedback is welcome!
	
# Motivation

When compared to text and pictures, video (specifically the audio) is an inefficient way to convey information, as in the base the viewer has to sit through the whole thing, while only part of the video may be relevant to them. This repo attempts to help solve that problem by converting long video files into text that you can read, CTRL+F, keyword extract, and summarize.
	
## Basic explanation of pipeline process
	
	- Receive directory input from user in script run window*. Iterate through that directory, find all video files
	- FOR each video file 
		- convert video to .wav format audio chunks of duration X** seconds with MoviePy
		- transcribe all X audio chunks through a pretrained wav2vec2model (transformers), store results in a list
		- write all results of the list into a text file, store various runtime metrics into a separate text list 
	- after above completed, create wo new files: one with all transcriptions appended, one with all metadata appended. 
	- FOR each transcription text file:
		- pass created textfile through a spell checker (PySpellChecker) and autocorrect spelling. save as new file
		- use pySBD to infer sentence boundaries and add periods in to delineate sentences. Save as new file 
		- run basic keyword extraction (via YAKE) on spell-corrected file. All keywords per file are stored in one dataframe for comparison , and exported to .xlsx format 
	- cleanup tasks (delete the X .wav files created for audio transcription, etc), report runtime, and exit
	
	* the 'single file' version needs to have the name defined in the python code itself
	** (where X is some duration that does not overload your computer or crash your IDE)
	
	
results are stored in ~/directory/w2v2_video_transcriptions and metadata in ~/directory/w2v2_transcription_metadata
	
## How to get this to work on your machine (aka installation)

* Currently, just normal Git methods. You can also download the .zip from the site, if you run the scripts in the same configuration it should work fine.
	
* Will make some changes that don't require vid2cleantxt_folder to be in the same folder as audio2text_functions for example.

## Is there a jupyter notebook file?
	
No, but am working on a version in Google Colab. Will update repo when done.

## Now I have a bunch of long text files. How are these useful?
	
short answer: noam_chomsky.jpeg
	
more comprehensive answer:
	
	A large corpus of text can be visualized / summarized / reduced in many ways with natural language processing / typical machine learning algorithms (i.e. classifying text, so on). Some packages to check out regarding this are TextHero and ScatterText. An example use case is combining the text from audio transcriptions with written documents (say textbooks or lecture slides converted to text) for comparison of how similar they are. You can also use topic models (available in ScatterText and many other packages) or statistical models (YAKE) to extract key topics from each file (or file group) and compare those (how they change over time, what are the key topics in practice exam PDF files, etc).
	
	Visualization and Analysis:
	
	https://github.com/jbesomi/texthero
	https://github.com/JasonKessler/scattertext
	
	Text Extraction / Manipulation:
	
	https://textract.readthedocs.io/
	https://github.com/chartbeat-labs/textacy
	https://github.com/LIAAD/yake
	
	Text Summarization:
	
	https://huggingface.co/models?pipeline_tag=summarization
	*I have personally found Google's T5 to be most effective for "lecture-esque" video conversion
	
I do have some of my own code that does this (given the motivation statement) but it needs some work before I publish an initial commit. It will be in a public repo on this account. 
	
## TextHero example use case: 

Clustering vectorized text files into k-means groups

![iml Plotting with TSNE + USE, Colored on Directory Name](https://user-images.githubusercontent.com/74869040/110546335-a0baaf80-812e-11eb-8d7d-48da00989dce.png)

![iml Plotting with TSNE + USE, Colored on K-Means Cluster](https://user-images.githubusercontent.com/74869040/110546452-c6e04f80-812e-11eb-9a4b-03213ec4a63b.png)

## ScatterText example use case: 

Comparing frequency of terms in one body of text vs. another

![ST P 1 term frequency I ML 2021 Docs I ML Prior Exams_072122_](https://user-images.githubusercontent.com/74869040/110546149-69e49980-812e-11eb-9c94-81fcb395b907.png)

## Why use wav2vec2 instead of SpeechRecognition or other transcription methods?

Google's SpeechRecognition (with the free API) requires optimization of three unknown parameters, which in my experience can vary widely among english as a second language speakers. With wav2vec2, the base model is pretrained, so a 'decent transcription' can be made without having to spend a bunch of time testing and optimizing parameters.
	
Also, because it's an API you can't train it even if you wanted to, you have to be online for functionally most of the script runtime, and then of course you have privacy concerns with sending data out of your machine.
	
## What about a version where I don't need python at all?

Work in progress the following package https://www.py2exe.org/. Will update repo when done
	
## How long does this take to run?

On my machine (CPU only due to AMD GPU) it takes approximately 80-120% of the total duration of input video files.

** Specs: **
	
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
	
Note: if you change wav2vec2_model = "facebook/wav2vec2-large-960h-lv60-self" to wav2vec2_model = "facebook/wav2vec2-base-960h" the runtime will be considerably faster. I have not compared accuracy on my own use cases, but I am sure Facebook has.

## How can I improve the performance of the model from a word-error-rate perspective?

You can train the model, but that requires that you already have a transcription of that person's speech already. As you may find, manual transcription is a bit of a pain and therefore transcripts are rarely provided - hence this repo.

## I've found x repo / script / concept that I think you should incorporate or collaborate with the author.

Send me a message / start a discussion! Always looking to improve.
		
# Troubleshooting

## I tried to transcribe an audio file and it gave me an error:
	
Planning to update the code to detect audio files and handle those. For now, only works on video files. If you want to try yourself, convert_vidfile and convert_vid_for_transcription just need to be updated.
	
## My computer crashes once it starts running the wav2vec2 model:
	
Try decreasing 'chunk_length' in vid2cleantxt_folder.py or vid2cleantxt_single.py (whichever you use). Until you get to really small intervals (say < 10 seconds) each audio chunk can more or less be treated independently as they are different sentences.
		
# Example

Transcription of Public Domain Speeches from President John F. Kennedy

## Description

The "example_JFK_speech" folder contains the results and interim files of running both the single file and folder version. NOTE: I had to split "GPU_President Kennedy speech on the space effort a" when pushing to Git, so there are 5 video files but when I ran the folder transcriber it was just one. Again, this shouldn't really be an issue as the audio is "independent"
	
## Output script run log for the "single_file" version:

	C:\Users\peter\AppData\Local\Microsoft\WindowsApps\python.exe C:/Users/peter/PycharmProjects/vid2cleantxt/vid2cleantxt/vid2cleantxt_single.py
	2021-03-09 22:07:21.289433: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
	2021-03-09 22:07:21.289808: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

	Preparing to load model: facebook/wav2vec2-large-960h-lv60-self

	============================================================
	Converting video to audio for file:  JFK_rice_moon_speech.mp4
	============================================================

	converting into 5 audio chunks
	separating audio into chunks starting at  _22.07.39
	Video conversion to Audio - chunks 20.000000 % done
	Video conversion to Audio - chunks 40.000000 % done
	Video conversion to Audio - chunks 60.000000 % done
	Video conversion to Audio - chunks 80.000000 % done
	Finished creating audio chunks at  _22.07.41
	Files are located in  C:\Users\peter\PycharmProjects\vid2cleantxt\example_JFK_speech\audio_chunks

	============================================================
	converted video to audio. About to start transcription loop for file:  JFK_rice_moon_speech.mp4
	============================================================


	Starting run   1 out of    5
	Current time for this run is  date_09_03_2021_time_22-08-12
	Based runtime average, ETA is   1.55  minutes

	Starting run   2 out of    5
	Current time for this run is  date_09_03_2021_time_22-09-00
	Based runtime average, ETA is   1.31  minutes

	Starting run   3 out of    5
	Current time for this run is  date_09_03_2021_time_22-09-45
	Based runtime average, ETA is   0.69  minutes

	Starting run   4 out of    5
	Current time for this run is  date_09_03_2021_time_22-10-27
	Based runtime average, ETA is   0.00  minutes

	Finished audio transcription of JFK_rice_moon_speech.mp4 and now saving metrics.

	Deleted Audio Chunk Folder + Files

	Finished transcription successfully for JFK_rice_moon_speech.mp4 at date_09_03_2021_time_22-10-57

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

## Citations (work in progress)
	
wav2vec2
	
MoviePy
	
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


	
	
