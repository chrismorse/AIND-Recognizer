import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses


    test_sequences = list(test_set.get_all_Xlengths().values())

    for test_X, test_Xlength in test_sequences:

        words_logL = {}
        for word, model in models.items():
            try:
                words_logL[word] = model.score(test_X,test_Xlength)
            except:    
                pass

        probabilities.append(words_logL)

    for prob in probabilities:
        max_word = None
        max_score = float("-inf")
        #print(prob)
        #print("")
        
        for word in prob:
            #print(word, prob[word])

            if prob[word] > max_score:
                max_word = word
                max_score = prob[word]
            

        guesses.append(max_word)

    return probabilities, guesses