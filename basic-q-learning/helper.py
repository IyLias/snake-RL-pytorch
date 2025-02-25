import matplotlib.pyplot as plt
from IPython import display

import os
os.environ[ 'KMP_DUPLICATE_LIB_OK'] = 'True'

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)



def file_save(operation_date, scores, mean_scores):

    with open('snake-RL-results.txt','a') as file:
        file.write("Operation Date: " + str(operation_date) + "\n")
        file.write("Number of Games: " + str(len(scores)) + "\n")
        file.write("Highest Score: " + str(max(scores)) + "\n")
        file.write("Scores: " + str(scores) + "\n")
        file.write("Mean Score: " + str(mean_scores) + "\n")

        file.write("\n\n")


