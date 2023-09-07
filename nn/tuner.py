import subprocess
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-save_dir', '--save_dir', type=str, default='True')
    parser.add_argument('-seq_len_list', '--seq_len_list', type=int, nargs='+')
    parser.add_argument('-dropout_list', '--dropout_list', type=float, nargs='+',)

    args = parser.parse_args()

    # cmd = "/content/eye-movement/nn/model2.py --seq_len {seq_val} --save_dir \"/content/eye-movement/nn/results/\" --dropout {dropout_val}"
    cmd = "C:/Users/r.ali/repo/eye-movement/nn/tuner.py --seq_len {seq_val} --save_dir '/content/eye-movement/nn/results/' --dropout {dropout_val}"

    for seq_val in args.seq_len_list:
        for dropout_val in args.dropout_list:
            cmd2 = cmd.format(seq_val = seq_val, dropout_val = dropout_val)
            print(cmd2)
            result = subprocess.run(["python","C:/Users/r.ali/repo/eye-movement/nn/tester.py"], capture_output= True)
            print(result)

            # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            # out, err = p.communicate() 
