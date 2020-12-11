import pandas as pd
import numpy
from numpy.random import RandomState
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import torch
from torch import pow
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DIMENSIONALITY = 50
BATCH_SIZE = 10
TEST_BATCH_SIZE = 50  # make as big as test data set
EPPOCHS = 20
LR = 1e-3


class Warper(nn.Module):
    def __init__(self):
        self.input_size = DIMENSIONALITY
        super(Warper, self).__init__()
        self.Linear_1 = Linear(self.input_size, 150)
        self.Linear_2 = Linear(150, 150)
        self.Linear_3 = Linear(150, DIMENSIONALITY)

    def forward(self, x):
        # assert torch.all(torch.eq(x, x))
        x = F.relu(self.Linear_1(x))      # leaky_relu, swoosh function,
        x = F.relu(self.Linear_2(x))
        x = self.Linear_3(x)
        return x


def optmize_model():
    for a, b, c in dataloader:
        a_warp = warper(a)
        b_warp = warper(b)
        c_warp = warper(c)
        dist1 = pow(a_warp - b_warp, 2.)
        dist1 = dist1.sum(1)
        dist1 = pow(dist1, 0.5)

        dist2 = pow(a_warp - c_warp, 2.)
        dist2 = dist2.sum(1)
        dist2 = pow(dist2, 0.5)

        error = torch.max(input=dist1 - dist2, other=torch.FloatTensor([0]))
        error_mse = pow(error, 2)
        error_mse = torch.mean(error_mse)

        optimizer.zero_grad()   # clear out previous gradients
        error_mse.backward()    # This line back prop's everything
        optimizer.step()        # Adjust all the parameters


def test_accuracy():
    for a, b, c in test_dataloader:
        a_warp = warper(a)
        b_warp = warper(b)
        c_warp = warper(c)
        dist1 = torch.pow(a - b, 2.)
        dist1 = dist1.sum(1)
        dist1 = torch.pow(dist1, 0.5)

        dist2 = torch.pow(a - c, 2.)
        dist2 = dist2.sum(1)
        dist2 = torch.pow(dist2, 0.5)

        total_greater = torch.sum(dist2 > dist1).float()
        total_instance = dist1.shape[0]

        return total_greater / total_instance


df = pd.read_csv('50d_testing.csv')

my_np_array = numpy.empty
my_np_array = genfromtxt('50d_testing.csv', delimiter=' ', dtype='float32')

# my_np_array = df.to_numpy()
# first_column = df.iloc[:, 0]
# print(first_column)
# first_column.to_csv('temp.csv')
# df = df     #.split(",", n=1, expand=True)
# my_data = genfromtxt('50d_training.csv', delimiter=',')

h, num_cols = my_np_array.shape
train = round(h * .7)

a_train = my_np_array[0:train, 0:50]
b_train = my_np_array[0:train, 50:100]
c_train = my_np_array[0:train, 100:150]

a_test = my_np_array[train:h, 0:50]
b_test = my_np_array[train:h, 50:100]
c_test = my_np_array[train:h, 100:150]

# numpy.savetxt("foo.csv", a_test, delimiter=",") # for debugging purposes

# pylint: disable=E1101
a_torch_train = torch.from_numpy(a_train)
b_torch_train = torch.from_numpy(b_train)
c_torch_train = torch.from_numpy(c_train)

a_torch_test = torch.from_numpy(a_test)
b_torch_test = torch.from_numpy(b_test)
c_torch_test = torch.from_numpy(c_test)

dataset = TensorDataset(a_torch_train, b_torch_train, c_torch_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(a_torch_test, b_torch_test, c_torch_test)
test_dataloader = DataLoader(
    test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

warper = Warper()
optimizer = optim.Adam([{'params': warper.parameters(),
                         'lr': LR, 'weight_decay': 1e-4}])

print('First iter', test_accuracy())

for run in tqdm(range(EPPOCHS)):
    optmize_model()

    print('EPPOCH:', run, ' %', test_accuracy())
    # pylint: enable=E1101

    # a_tf_train = tf.convert_to_tensor(a_train, numpy.float32)
    # b_tf_train = tf.convert_to_tensor(b_train, numpy.float32)
    # c_tf_train = tf.convert_to_tensor(c_train, numpy.float32)

    # a_tf_test = tf.convert_to_tensor(a_test, numpy.float32)
    # b_tf_test = tf.convert_to_tensor(b_test, numpy.float32)
    # c_tf_test = tf.convert_to_tensor(c_test, numpy.float32)

    # rng = RandomState()  # What was this again?

    # train = df.sample(frac=0.7, random_state=rng)
    # test = df.loc[~df.index.isin(train.index)]

    # my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    # for x, y, z in dloader:
    #     print(x)
    #     print(y)
    #     print(z)

    # train.to_csv("train.csv"), val.to_csv("val.csv")

    # Add any other params such as transforms here
    # train_dataset = Roof_dataset(csv_file="train.csv")
    # val_dataset = Roof_dataset(csv_file="val.csv")  # Again add any other params

    # print('train dataset')
    # print(train_dataset[0])

    # print('test dataset')
    # print(val_dataset[0])
