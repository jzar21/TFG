import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, help='Folder with the data')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch', type=int, default=25, help='Num Epochs')
    parser.add_argument('--train', type=bool, default=True, help='If train or evaluate')

    return parser.parse_args()

def main():
    pass


if __name__ == '__main__':

    args = parse_args()

    main()