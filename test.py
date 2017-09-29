import numpy as np
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit


asl = AslDb() # initializes the database
print(asl.df.head()) # displays the first five rows of the asl database, indexed by video and frame
print("- " * 50)
print(" ")

print(asl.df.ix[98,1])  # look at the data available for an individual frame
print("- " * 50)
print(" ")

# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)
print("- " * 50)


# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
print([asl.df.ix[98,1][v] for v in features_ground])


# Building training set
training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))
print("get_all_sequences: {}".format(training.get_all_sequences))
print("get_all_Xlengths: {}".format(training.get_all_Xlengths))
print("get_word_sequences: {}".format(training.get_word_sequences))
print("get_word_Xlengths: {}".format(training.get_word_Xlengths))


# train multiple sequences with the hmmlearn library
# In the following example, notice that there are two lists; 
# the first is a concatenation of all the sequences(the X portion) and the 
# second is a list of the sequence lengths(the Lengths portion).
print(training.get_word_Xlengths('CHOCOLATE'))


print("- " * 50)

# More feature sets

# normalize each speaker's range of motion
df_means = asl.df.groupby('speaker').mean()
print("df_means")
print(df_means)

print("- " * 50)
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
print(asl.df.head())


print("- " * 50)

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker


df_std = asl.df.groupby('speaker').std()
print("df_std")

# test the code
test_std_tryit(df_std)


print("- " * 50)


#Features Implementation Submission

