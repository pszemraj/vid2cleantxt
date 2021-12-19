# Console output of the script: largeWav2Vec2 model

The source video for the below is ~27 minutes long, and took about 17 minutes to finish running on my machine. 


## printout on console

peter@Shem-ryePowerStation MINGW64 ~/Dropbox/programming*projects/vid2cleantxt (user-friendly)
$ python vid2cleantxt/transcribe.py --input-dir "example_JFK_speech\TEST_singlefile" --model "facebook/wav2vec2-large-960h-lv60-self"
data folder is set to `C:\Users\peter\.conda\envs\v2ct\lib\site-packages\neuspell\../data` script
Loading models @ Dec-15-2021*-01-07-07 - may take a while...
If RT seems excessive, try --verbose flag or checking logfile

Found 1 video files in C:\Users\peter\Dropbox\programming*projects\vid2cleantxt\example_JFK_speech\TEST_singlefileCreating .wav audio clips: 100%|███████████████████| 109/109 [00:00<00:00, 162.44it/s]
Creating .wav audio clips: 99%|██████████████████▊| 108/109 [00:00<00:00, 208.00it/s] created audio chunks for wav2vec2 - Dec-15-2021*-01
No GPU being used by this machine :(
No GPU being used :/ 0%| | 0/109 [00:00<?, ?it/s]

Gen RAM Free: 10.2 GB | Proc size: 3.1 GB | 8 CPUs loaded at 30.0 % |

2021-12-15 01:07:46,390 WARNING:C:\Users\peter\.conda\envs\v2ct\lib\site-packages\transformers\models\wav2vec2\modeling_wav2vec2.py:1055: UserWarning: **floordiv** is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
return (input_length - kernel_size) // stride + 1

                                                                                      No GPU being used :/  50%|██████████████▏             | 55/109 [07:54<07:32,  8.37s/it]

Gen RAM Free: 11.4 GB | Proc size: 3.2 GB | 8 CPUs loaded at 64.3 % |

Transcribing video: 100%|███████████████████████████| 109/109 [15:42<00:00, 8.65s/it]
Saved transcript and metadata to C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcriptions and C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transc_metadata
transcribing vids: 100%|███████████████████████████████| 1/1 [15:50<00:00, 950.22s/it]
SC_pipeline - transcribed audio: 0%| | 0/1 [00:00<?, ?it/s]
Top 10 Key Phrases from YAKE, with max n-gram length 3
['war total war',
'tons world peace',
'interests nuclear powers',
'world security system',
'peace corps abroad',
'safeguard human interests',
'soviet union adopt',
'daily lives live',
'arm controls designed',
'american imperialist circles']
SC_pipeline - transcribed audio: 100%|██████████████████| 1/1 [00:36<00:00, 36.60s/it]

Finished at: Dec-15-2021\_-01 taking a total of 16.910507316666667 mins
relevant files for run are in:
C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transcriptions
and:
C:\Users\peter\Dropbox\programming_projects\vid2cleantxt\example_JFK_speech\TEST_singlefile\v2clntxt_transc_metadata
(v2ct)
peter@Shem-ryePowerStation MINGW64 ~/Dropbox/programming_projects/vid2cleantxt (user-friendly)
$
