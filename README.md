# ML_final_project

## Description


## The Package smb2-gym
We used the Package smb2-gym (https://pypi.org/project/smb2-gym/) to create a training enviroment for super mario bros 2. This package provided a simplified action space for mario with 12 actions to be able to have a simpler training. We counldn't find any documentation, which action is what. Based on trying it out we assumed the following meanings: <br>
0: Nothing <br>
1: Right <br>
2: Left <br>
3: Up (which enters the door) <br>
4: A Button (Jump) <br>
5: B Button (Pickup/Throw) <br>
6: Right + A <br>
7: Left + A <br>
8: Right + B <br>
9: Left + B <br>
10: Down (Duck) <br>
11: Down + A <br>

## How to run 
Thos project is a python project. All dependecies needed to run the project are listed in requirements.txt. 
To train the models run 'python train.py'. Then you can choose which agent should be trained. To exit savely press q. It will finish the current episode, save the data and then exit. The rendering while training is optional. To load a model and let it play run 'python play.py'. Then you can choose which agent should play or if you want to play yourself. If an angent is chosen it will play 1 episode and print metrics how it did on that episode.

## Credits
- This project is based on the idea of a project in another course ( Topics in Computer Sience). in this project we tried to train a DQN agent to play Clash Royal, but didn't reall think about the networks used and didn't compare different types. Some Files of that project were used as a base for this one (corresponding files are marked): https://github.com/adriandbf/Merge-Tactics-AI by Adrian Fudge and Vera Schmitt
- Articel we used to understand and work with DQN: https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae

## Authors
Alina Haider <br>
Colin Hewlett <br>
Vera Schmitt <br>