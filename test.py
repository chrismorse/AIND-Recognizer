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





#------------------------------------------
#Features Implementation Submission
#------------------------------------------

# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

# normalize each speaker's range of motion
df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

df_std = asl.df.groupby('speaker').std()
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])

asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
asl.df['polar-rr'] = np.sqrt((asl.df['right-x']-asl.df['nose-x'])**2 + (asl.df['right-y']-asl.df['nose-y'])**2)
asl.df['polar-rtheta'] = np.arctan2(asl.df['right-x']-asl.df['nose-x'],asl.df['right-y']-asl.df['nose-y'])
asl.df['polar-lr'] = np.sqrt((asl.df['left-x']-asl.df['nose-x'])**2 + (asl.df['left-y']-asl.df['nose-y'])**2)
asl.df['polar-ltheta'] = np.arctan2(asl.df['left-x']-asl.df['nose-x'],asl.df['left-y']-asl.df['nose-y'])


features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(value=0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(value=0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(value=0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(value=0)

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like

asl.df['delta-grnd-rx'] = asl.df['grnd-rx'].diff().fillna(value=0)
asl.df['delta-grnd-ry'] = asl.df['grnd-ry'].diff().fillna(value=0)
asl.df['delta-grnd-lx'] = asl.df['grnd-lx'].diff().fillna(value=0)
asl.df['delta-grnd-ly'] = asl.df['grnd-ly'].diff().fillna(value=0)

features_custom = ['delta-grnd-rx', 'delta-grnd-ry', 'delta-grnd-lx', 'delta-grnd-ly']

import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)



