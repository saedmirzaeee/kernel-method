import csv 
import pandas as pd
import pickle as pkl
import sys
from random import seed
from random import randrange
from functools import reduce

data_folder = '../data/'

letterCodes = { "A" : 0, "C" : 1, "G" : 2, "T": 3}
letters = ["A", "C", "G", "T"]

class Converter:
    """Utility class to encode kmers as integers and vice versa

    Each letter in a kmer of length k is encoded as two bits as follows
    "A" : 0, "C" : 1, "G" : 2, "T": 3
    A string of lenght k is represented as an int with at least k bits. 
    The most significant bits (position 2*k and 2*k-1 from the right) encode the first letter,
    the subsequent two bits the second letter and so on.
    """
    def __init__(self, k):
        self.k = k
        self.mask = 4**k - 1

    def _processNextLetter(self, acc, c):
        #shift int representation 2 bits to the right
        #truncate highest 2 bits
        #encode current letter in the lowest two bits
        return (acc << 2) & self.mask | letterCodes[c]

    def kmer2int(self, kmer):
        return reduce(self._processNextLetter, kmer, 0)

    def int2kmer(self, intRepr):
        return "".join(letters[(intRepr >> (2 * i)) & 3] for i in range(self.k - 1, -1, -1))

    def all_kmers_as_ints(self, read):
        """Encodes all kmers of length k in a read as integers
        
        Example for k = 2: "ACGT" would be result [01, 12, 23] (the numbers are written in base 4).
        Returns a lazy evaluated iterator.
        """

        first_int = self.kmer2int(read[i] for i in range(self.k))  #representation of the first k letters
        return scanl(self._processNextLetter, first_int, (read[i] for i in range(self.k, len(read))))  


#Similar to Haskell's scanl
def scanl(f, z, xs):
    yield z
    for x in xs:
        z = f(z, x)
        yield z


def read_raw_file(file_name):
    """
    docstring
    """
    lines = []
    # opening the CSV file 
    with open(file_name, mode ='r')as file: 
        # reading the CSV file 
        csvFile = csv.reader(file) 

        # displaying the contents of the CSV file 
        for line in csvFile:
            lines.append(line[1])

        return lines[1:]


def read_mat_file(file_name):
    """
    docstring
    """
    df = pd.read_csv(file_name, header=None, delimiter=r"\s+")
    return list(df.values)


def read_x_data(train=True, raw=True):
    """
        Read the 3 files in raw/matrix format for train/test, depending on the flags(raw,train)
    """
    x_data = list([])
    raw_or_prep = '' if raw==True else '_mat100'
    tr_or_test = 'tr' if train==True else 'te'

    for k in range(0,3):
        x_file_name = '{}X{}{}{}.csv'.format(data_folder, tr_or_test, k, raw_or_prep)
        if raw == False:
            x_data_k = read_mat_file(x_file_name)
        else:
            x_data_k = read_raw_file(x_file_name)

        x_data.append(x_data_k)
    return x_data


def read_y_data():
    """
        Read the labels from all 3 files as a list of integers.
        Keep '1' as 1
        Transform '0' in -1
    """
    y_train = list([])

    for k in range(0,3):
        y_file = '{}Ytr{}.csv'.format(data_folder, k)
        y_train_k = read_raw_file(y_file)

        # Convert '0's into '1's
        y_train_k = [-1 if x == '0' else 1 for x in y_train_k]

        # Append the content of the current file to 'y_train'
        y_train.append(y_train_k)
    
    return y_train


def save_object(obj, name):
    file_path = '../data/kernel_mat/{}.pkl'.format(name)
    with open(file_path, 'wb') as output:
        pkl.dump(obj, output)


def load_object(name):
    file_path = '../data/kernel_mat/{}.pkl'.format(name)
    try:
        with open(file_path, 'rb') as pkl_file:
            read_obj = pkl.load(pkl_file)
            return read_obj    
    except IOError:
        print('File {} does not exist.'.format(file_path))
        sys.exit()


def array_to_labels(elements, negative_label=0):
    """
        For each element in the list:
            if element > 0 => replace it with 1
            if element < 0 => replace it with 0 OR -1, depending on the 'negative_label'
        
        negative_label is 0 in the original training data
                          -1 when used in the algortihms
    """

    return [negative_label if x < 0 else 1 for x in elements]


def k_fold_split(data, folds=4):
    data_split = list()
    data_copy = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split


def accuracy_score(predicted, expected):
    """
        Accuracy = #of labels well predicted / #of all labels
    """
    count = 0

    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            count += 1
    return count / len(predicted)


def build_kmers_dict(sequences, k, distribution):
    """
        Function which build a dictionary with all the substrings of length k for
        each entry in the list of sequences
    """
    conv = Converter(k)
    kmer_dict = {}

    for seq in sequences:
        # Add each new kmer in the dictionary
        for kmer in conv.all_kmers_as_ints(seq):
            kmer_dict[kmer] = 0

    # Save the dictionary in pickle format
    # save_object(kmer_dict, 'kmer_dict_distribution={}_k={}'.format(distribution, k))

    return kmer_dict


def augment_data(train_x, train_y):
    all_new_train_x = []
    all_new_train_y = []
    nucleotide_dic = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}

    for i in range(len(train_x)): # range(3) 
        new_train_x = []
        new_train_y = []
        for seq_i in range(len(train_x[i])):
            new_seq = list(map(lambda x: nucleotide_dic[x], train_x[i][seq_i]))
            new_seq = ''.join(new_seq)

            new_train_x.append(train_x[i][seq_i])
            new_train_x.append(new_seq)
            new_train_y.append(train_y[i][seq_i])
            new_train_y.append(train_y[i][seq_i])
        all_new_train_x.append(new_train_x)
        all_new_train_y.append(new_train_y)

    return [all_new_train_x, all_new_train_y]

def write_labels_csv(labels):
    fields = ['Id', 'Bound']
    filename = 'test_results.csv'

    rows = []

    for i in range(len(labels)):
        rows.append([i, labels[i]])

    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow(fields)  

        # writing the data rows  
        csvwriter.writerows(rows)
        
def write_predicted_labels_csv(labels, filename='Yte.csv'):
    fields = ['Id', 'Bound']

    rows = []

    for i in range(len(labels)):
        rows.append([i, labels[i]])

    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow(fields)  

        # writing the data rows  
        csvwriter.writerows(rows)


def write_labels_csv_KRR_spectrum(labels, file_path, distrib):
    fields = ['Id', 'Bound']

    rows = []

    for i in range(len(labels)):
        rows.append([i + distrib * len(labels), labels[i]])

    with open(file_path, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow(fields)  

        # writing the data rows  
        csvwriter.writerows(rows)

