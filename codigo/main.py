import argparse

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, help='Folder with the data')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')

    args = parser.parse_args()


    main()