import pickle


def generate_comp_tagged():
    # Load data (deserialize)
    with open('filename.pickle', 'rb') as handle:
        unserialized_data = pickle.load(handle)


if __name__ == '__main__':
    generate_comp_tagged()
