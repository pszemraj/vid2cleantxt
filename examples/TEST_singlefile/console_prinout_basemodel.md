# Console output of the script: base Wav2Vec2 model

The source video for the below is ~27 minutes long, and took about 7.5 minutes to finish running on my machine.

## The following is the output of the script:

peter@Shem-ryePowerStation MINGW64 ~/Dropbox/programming*projects/vid2cleantxt (user-friendly)
$ python vid2cleantxt/transcribe.py --input-dir "example_JFK_speech\TEST_singlefile"
data folder is set to `C:\Users\peter\.conda\envs\v2ct\lib\site-packages\neuspell\../data` script
Loading models @ Dec-15-2021*-00-55-45 - may take a while...
If RT seems excessive, try --verbose flag or checking logfile

Found 1 video files in C:\Users\peter\Dropbox\programming*projects\vid2cleantxt\example_JFK_speech\TEST_singlefile
Creating .wav audio clips: 100%|█| 109/109 [00:00<00:00, 2
Creating .wav audio clips: 66%|▋| 72/109 [00:00<00:00, 26
created audio chunks for wav2vec2 - Dec-15-2021*-00
No GPU being used by this machine :(

                                                          No GPU being used :/   0%|         | 0/109 [00:00<?, ?it/s]

Gen RAM Free: 11.8 GB | Proc size: 2.2 GB | 8 CPUs loaded at 21.3 % |

                                                          No GPU being used :/  50%|▌| 55/109 [03:05<02:54,  3.23s/it

Gen RAM Free: 11.9 GB | Proc size: 2.3 GB | 8 CPUs loaded at 51.8 % |

Transcribing video: 100%|█| 109/109 [06:08<00:00, 3.38s/i
Saved transcript and metadata to C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcriptions and C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transc_metadata
transcribing vids: 100%|███| 1/1 [06:14<00:00, 374.09s/it]
SC_pipeline - transcribed audio: 0%| | 0/1 [00:00<?, ?it
Top 10 Key Phrases from YAKE, with max n-gram length 3
['war total war',
'powers genuine peace',
'genuine world security',
'good human interest',
'soviet union adopt',
'attainable peace based',
'interest pupil powers',
'nations closest allies',
'cold war remembering',
'single nuclear weapon']
SC_pipeline - transcribed audio: 100%|█| 1/1 [00:39<00:00,

Finished at: Dec-15-2021\_-01 taking a total of 7.335023368333333 mins
relevant files for run are in:
C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcriptions
 and:
C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transc_metadata
(v2ct)
