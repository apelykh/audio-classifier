# Audio Classifier

## Description

Simple convolutional model for audio classification, trained on MFCC features.

<p align="center">
  <img width="460" height="300" src="http://www.fillmurray.com/460/300">
</p>


The model is trained and evaluated on the subset of **Google Speech Commands Dataset** ([download](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)) that includes `down`, `go`, `left`, `no`, `off`, `on`, `right`, `stop`, `up`, `yes` commands.


## Evaluation results

Results achieved on a test set (randomly selected 10% of the whole data):
* Test accuracy: **0.744**
* Avg. F1 score: **0.747**


## Future work

To further improve the model performance, the following directions might me investigated:
- Deeper and more sophisticated model;
- More thorough data preprocessing (e.g. noise removal, audio aligning);
- Other time-frequency features instead/in addition to MFCC;

