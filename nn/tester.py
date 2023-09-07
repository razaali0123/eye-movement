import argparse

def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '2' to filter out warnings as well

    parser = argparse.ArgumentParser()

    # )
    parser.add_argument('-save_dir', '--save_dir', type=str, default='True')
    parser.add_argument('-seq_len', '--seq_len', type=int, default=50)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)

    

    args = parser.parse_args()

    print("adding 1 to dropout ", args.dropout + 1)

if __name__ == '__main__':
    raise SystemExit(main())
