# Project Title

A LSTM-based trajectory classifier.

## Usage

use example.ipynb 

## Label 

| Label | 中文 | English |
|---------|---------|---------|
| 0  | 對方沒接到，球落在對方場內 / 對方有接到球但球沒有過場 | The rally concludes with an unreturnable spike.   |
| 1  | 對方接到但出界                                     | The rally concludes with the ball going out of bounds after the opponent's touch.  |
| 2  | 對方攔阻成功，球落在已方場內                        | The rally concludes with the opponent's successful block, and the ball drops on our side of the court. |
| 3  | 對方攔阻失敗，球落在對方場內                        | The rally concludes with the opponent's block failing, and the ball drops on their side of the court.|
| 4  | 對方攔阻失敗，球落在界外                            | The rally concludes with the opponent's block failing, and the ball landing out of bounds.|
| 5  | 扣球失誤(扣球到界外、扣球掛網、扣球觸網)             |  The rally concludes with a spike fault.|
| 6  |發球失誤(掛網、出界)|The rally concludes with a service fault.|
| 7  |發球得分（對方接到，但失誤，球落在場內/外、對方未接到，球落在場內）|The rally concludes with a service ace.|
