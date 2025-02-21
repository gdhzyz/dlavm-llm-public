import sys
import socket
import numpy as np
import time
from transformers import AutoTokenizer


def encode_command(cmd, attrs):
    return "command%sattrs%04d" % (cmd, len(attrs))


def decode_command(encode_message):
    message = str(encode_message, 'utf-8')
    if "command" == message[:7]:
        return message[7:11], message[14:15]
    else:
        print(message)
        return "Error", 0


def run_model(client, input_ids, token, kvcache):
    attrs = np.array([token, kvcache], dtype="uint16").tobytes()
    attrs += input_ids.numpy().astype("uint16").tobytes()
    client.sendall(bytes(encode_command("rmod", attrs), 'utf-8') + attrs)
    strClientData = client.recv(1024)
    state = np.frombuffer(strClientData, dtype="uint16")
    return int(state[0])


def run_model_kvcache(client, input_ids, token, kvcache, memory):
    attrs = np.array([token, kvcache, memory], dtype="uint16").tobytes()
    attrs += input_ids.numpy().astype("uint16").tobytes()
    client.sendall(bytes(encode_command("rkvc", attrs), 'utf-8') + attrs)
    strClientData = client.recv(1024)
    ids = np.frombuffer(strClientData, dtype="uint16")
    ids_len = ids[0]
    ids = ids[1:1+ids_len].astype("int32").tolist()
    strClientData = client.recv(1024)
    state = np.frombuffer(strClientData, dtype="uint16")
    return int(state[0]), ids_len, ids


def run_model_kvcache_show(client, input_ids, token, kvcache, memory):
    attrs = np.array([token, kvcache, memory], dtype="uint16").tobytes()
    attrs += input_ids.numpy().astype("uint16").tobytes()
    client.sendall(bytes(encode_command("rkvs", attrs), 'utf-8') + attrs)


def get_next_ids(client):
    strClientData = client.recv(1024)
    ids = np.frombuffer(strClientData, dtype="uint16")
    ids_len = ids[0]
    state = ids[0]
    ids = ids[1:1+ids_len].astype("int32").tolist()
    return state, ids_len, ids


def main_kvcache_show(tokenizer, client, memory=0):
    round = 1
    while True:
        query = input("User: ")
        if query == "exit" or query == "quit":
            break
        prompt = "[Round {}]\n\n问：{}\n\n答：".format(round, query)
        inputs = tokenizer([prompt], return_tensors="pt")
        print("FPGA: ", end="")
        run_model_kvcache_show(client, inputs["input_ids"], inputs["input_ids"].shape[1], 0, memory)
        while True:
            state, ids_len, ids = get_next_ids(client)
            if state == 0:
                print("")
                break
            generated_text = tokenizer.decode(ids)
            print(generated_text, end="")
        round += memory
    client.sendall(bytes("quit", 'utf-8'))
    client.close()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mytokens", trust_remote_code=True, local_files_only=True)
    print("Success load tokenizer")
    client = socket.socket()
    ip, port = "127.0.0.1", 8123
    if len(sys.argv) == 2:
        ip = sys.argv[1]
    elif len(sys.argv) == 3:
        ip, port = sys.argv[1], int(sys.argv[2])
    print(f"{ip}:{port}")
    client.connect((ip, port))
    print('Client V1：连接到 FPGA ChatGLM2')
    main_kvcache_show(tokenizer, client, memory=0)
