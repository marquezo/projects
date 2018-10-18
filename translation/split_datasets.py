import numpy as np
import os, argparse

def split_save_datasets(from_lang, to_lang, split_percentage=0.1):
    """
    Given a file with bilingual data, create two files: train and dev
    Do not do any normalization

    :param from_lang:
    :param to_lang:
    :param split_percentage:
    :return:
    """
    print("Reading lines...")
    data_file = 'data/%s-%s.txt' % (from_lang, to_lang)

    if os.path.isfile(data_file):
        # Read the file and split into lines
        lines = open(data_file, encoding='utf-8').read().strip().split('\n')
        dev_size = int(len(lines) * split_percentage)
        dev_set_indices = np.random.choice(len(lines), dev_size, replace=False)

        print("Size of train/dev set to create is {}/{}".format(len(lines) - dev_size, dev_size))
        dev_set_indices.sort()  # sort so that we pad efficiently

        dev_pairs = []
        train_pairs = []

        for idx in range(len(lines)):
            if idx in dev_set_indices:
                dev_pairs.append(lines[idx])
            else:
                train_pairs.append(lines[idx])

        del lines

        print("Size of created train/dev set is {}/{}".format(len(train_pairs), len(dev_pairs)))

        train_file = "data/%s-%s.train" % (from_lang, to_lang)
        dev_file = "data/%s-%s.dev" % (from_lang, to_lang)

        with open(train_file, 'w') as f:
            for pair in train_pairs:
                f.write("{}\n".format(pair))

        with open(dev_file, 'w') as f:
            for pair in dev_pairs:
                f.write("{}\n".format(pair))

        print("Wrote datasets to {} and {}".format(train_file, dev_file))

        return train_pairs, dev_pairs

    else:
        raise Exception("Dataset {} not found".format(data_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an NMT dataset")
    parser.add_argument('from_lang', type=str)
    parser.add_argument('to_lang', type=str)
    parser.add_argument('split_percentage', default=0.1, type=float)
    args = parser.parse_args()

    split_save_datasets(args.from_lang, args.to_lang, args.split_percentage)
