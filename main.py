import time
import os
from scipy.io import wavfile
import asyncio
import subprocess
import requests
import json
import argparse
import torch
from torch import no_grad, LongTensor
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import commons

chatGPTUri = "https://api.openai.com/v1/completions"
chatGPTKey = "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"
chatGPTModel = "text-davinci-003"
chatGPTMaxTokens = 512
chatGPTTemperature = 0.5
chatGPTInitPrompt = "请扮演一个AI虚拟主播。不要强调你是AI虚拟主播，不要答非所问，不要有重复的语句，回答精简一点。这是观众的提问："

vitsNoiseScale = 0.6
vitsNoiseScaleW = 0.668
vitsLengthScale = 1.2

_proxies = {'http': "http://127.0.0.1:7890", 'https': "http://127.0.0.1:7890"}
_init_vits_model = False

hps_ms = None
device = None
net_g_ms = None

async def send_chatgpt_request(send_msg):
    data = json.dumps({ "model": f"{chatGPTModel}", "prompt": f"{chatGPTInitPrompt}{send_msg}", "max_tokens": chatGPTMaxTokens, "temperature": chatGPTTemperature})
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {chatGPTKey}"}
    response = requests.post(chatGPTUri, data=data, headers=headers, proxies=_proxies)
    output = response.json()
    result = output["choices"][0]["text"].strip("\n")
    print("[AI回复] : ", result)
    return result

def play_audio(audio_file_name):
    command = f'mpv.exe -vo null {audio_file_name}'
    subprocess.run(command, shell=True)

def init_vits_model():
    global hps_ms, device, net_g_ms

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--colab", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    hps_ms = utils.get_hparams_from_file(r'./model/config.json')
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(r'./model/G_953000.pth', net_g_ms, None)
    _init_vits_model = True

def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    global over

    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 200 and limitation:
        return f"输入文字过长！{len(text)}>100", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = LongTensor([speaker_id]).to(device)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"

async def start():
    while True:
        print("请输入 >")
        input_str = await asyncio.get_event_loop().run_in_executor(None, input, '')
        if "关闭AI" in input_str:
            return
        result = await send_chatgpt_request(input_str)
        status, audios, time = vits(result, 0, 124, vitsNoiseScale, vitsNoiseScaleW, vitsLengthScale)
        print("VITS : ", status, time)
        wavfile.write("output.wav", audios[0], audios[1])
        play_audio("output.wav")

async def main():
    if not _init_vits_model:
        init_vits_model()
    await asyncio.gather(start(),)

asyncio.run(main())
    
    

