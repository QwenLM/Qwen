import logging
import subprocess
import socket
import openai


def run_in_subprocess(cmd):
    try:
        with subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as return_info:
            while True:
                next_line = return_info.stdout.readline()
                return_line = next_line.decode("utf-8", "ignore").strip()
                if return_line == "" and return_info.poll() != None:
                    break
                if return_line != "":
                    logging.info(return_line)

            err_lines = ""
            while True:
                next_line = return_info.stderr.readline()
                return_line = next_line.decode("utf-8", "ignore").strip()
                if return_line == "" and return_info.poll() != None:
                    break
                if return_line != "":
                    logging.info(return_line)
                    err_lines += return_line + "\n"

            return_code = return_info.wait()
            if return_code:
                raise RuntimeError(err_lines)
    except Exception as e:
        raise e


def simple_openai_api(model):
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"

    # create a request not activating streaming response
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "你好"}],
        stream=False,
        stop=[],  # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
    )
    print(response.choices[0].message.content)


def TelnetPort(server_ip, port):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(1)
    connect_flag = False
    try:
        sk.connect((server_ip, port))
        connect_flag = True
    except Exception:
        connect_flag = False
    sk.close()
    return connect_flag
