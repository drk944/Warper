import re
import numpy as np

count = 0
data = np.empty(shape=[0, 3], dtype=object)
correctGuess = False

with open("/Users/Derek/Documents/GitHub/Game/codenames/first_guess_log.txt") as fp:
    while True:
        count += 1
        line = fp.readline()
        #regex = re.compile('^a-zA-Z0-9')
        regex = re.sub(r'[^A-Za-z ]', '', line)
        # print(regex)
        if not line:    # if end of file
            break
        # for word in regex.split():
            # print(word)
        comparisons = list(map(lambda x: x, regex.split()))
        # print(comparisons)
        # commented out code is for adding the possibility to
        # differentiate between correct and incorrect guesses
        # if (comparisons[1].upper() == comparisons[4]):
        for i in range(5, len(comparisons), 1):
            entry1 = comparisons[3]  # clue given
            entry2 = comparisons[1]  # answer given
            entry3 = comparisons[i].lower()  # other words in list
            new_row = np.array([entry1, entry2, entry3])

            data = np.row_stack((data, new_row))
        # else:
        #     for i in range(5, len(comparisons), 1):
        #         entry1 = comparisons[3]  # clue given
        #         entry2 = comparisons[1]  # incorrect but guessed answer
        #         entry3 = comparisons[i].lower()  # other words in list
        #         new_row = np.array([entry1, entry2, entry3])

        #         data = np.row_stack((data, new_row))
    # print(data)
    np.savetxt("/Users/Derek/Documents/GitHub/Game/codenames/comparisons.csv",
               data, fmt='%s', delimiter=",")
    # codenames\first_guess_log.txt
    # for item in range(array):
fp.close()
