import re
import os
import threading
import traceback
import base64
import threading
import traceback
import soundfile as sf
import numpy as np
import torch
import commons

from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from fastapi import status
from torch import no_grad, LongTensor
from funasr import AutoModel

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


class ProcessJobBase(object):
    instance = None
    
    def __init__(self, kwargs: dict):
        if not ProcessJobBase.instance:
            ProcessJobBase.instance = self
        else:
            raise Exception("ProcessJobBase instance already exists")    
    
    def run_job(self, **kwargs):
        pass 


class ProcessJobVITS(ProcessJobBase): 
    
    def __init__(self, kwargs: dict):
        super().__init__(kwargs=kwargs)
        self.args = kwargs["args"]
        self._load_t2s_model()
        ProcessJobVITS.instance = self
    
    @staticmethod
    def load_t2s_model(task_id):
        ProcessJobVITS.instance._load_t2s_model()
    
    def _load_t2s_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        hps = utils.get_hparams_from_file(self.args.config)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            8192 // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = utils.load_model(self.args.model, net_g)

        self.vits = net_g
        self.speakers = hps.speaker
        print("speaker", self.speakers)
        self.hps = hps
        self.device = device    

        self.asr_model = AutoModel(model=hps.asr.asr_path, model_revision="v2.0.4",
                                   vad_model=hps.asr.vad_path,
                                   vad_model_revision="v2.0.4",
                                   punc_model=hps.asr.punc_path,
                                   punc_model_revision="v2.0.4")

    def __cut_sentence(self, para):
        para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
        para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
        para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
        para = re.sub("(\…{1})([^”’])", r"\1\n\2", para)  # 中文省略号
        para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        return para.split("\n")

    def __get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def _t2s(self, text, speaker_id, speed=1.0):
        texts = self.__cut_sentence(text)
        audio_list = []
        for text in texts:
            stn_tst = self.__get_text(text)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(self.device)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
                sid = LongTensor([speaker_id]).to(self.device)
                audio = self.vits.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                    length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
                audio_list.append(audio)
                silence = np.zeros((int)(22050 * 0.5 / speed))
                audio_list.append(silence)
            del stn_tst, x_tst, x_tst_lengths, sid
        return self.hps.data.sampling_rate, np.concatenate(audio_list)
    
    def _s2t(self, audio_path):
        try:
            result = self.asr_model.generate(input=audio_path)
            return result[0]["text"]
        except Exception as e:
            print(f"func: _s2t, error: {str(e)}")
            traceback.print_exc()
            return None


class ProcessJobVITSApi(ProcessJobVITS):
    
    def __init__(self, kwargs: dict):
        super().__init__(kwargs=kwargs)
        ProcessJobVITSApi.instance = self
        self.thread_pool = ThreadPoolExecutor(max_workers=self.args.thread_num_per_worker, thread_name_prefix="api_work_thread_in_sub_process")
    
    @staticmethod
    def pipe_send_wrapper(func, pipe_conn, *_args):
        result = func(*_args)
        pipe_conn.send(result)
    
    @staticmethod
    def run_task_by_thread_pool(func, pipe_conn, /, *_args):
        return ProcessJobVITSApi.instance.thread_pool.submit(ProcessJobVITSApi.pipe_send_wrapper, func, pipe_conn, *_args)
    
    @staticmethod
    def api_s2t(audio_path: str) -> dict:
        return ProcessJobVITSApi.instance.__api_s2t(audio_path)

    @staticmethod
    def api_t2s(text: str, speaker_id: str, speed: float) -> dict:
        return ProcessJobVITSApi.instance.__api_t2s(text, speaker_id, speed)
    
    @staticmethod
    def api_t2s_bin(text: str, speaker_id: str, speed: float) -> dict:
        return ProcessJobVITSApi.instance.__api_t2s_bin(text, speaker_id, speed)
    
    @staticmethod
    def api_get_speakers() -> dict:
        return ProcessJobVITSApi.instance.__api_get_speakers()

    def __api_s2t(self, audio_path: str) -> dict:
        print(f"api_s2t, pid={os.getpid()}, thread_id={threading.get_ident()}")
        try:
            text = self._s2t(audio_path)
            Path(audio_path).unlink()
            return {
                "code": status.HTTP_200_OK,
                "text": text
            }
        except Exception as e:
            print(f"func: api_s2t, error: {str(e)}")
            traceback.print_exc()
            return {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": str(e)
            }    
    
    def __api_t2s(self, text: str, speaker_id: str, speed=1.0) -> dict:
        print(f"api_t2s spk_name={speaker_id}, text={text}, pid={os.getpid()}, thread_id={threading.get_ident()}")
        if speaker_id not in self.speakers:
            return {
                "code": status.HTTP_400_BAD_REQUEST,
                "message": f"invalid speaker_id:{speaker_id}"
            }
        speaker_id = self.speakers[speaker_id]

        try:
            sr, wav_data = self._t2s(text=text, speaker_id=speaker_id, speed=speed)
            return {
                "code": status.HTTP_200_OK,
                "sample_rate": sr,
                "data": wav_data.tolist()
            }
        except Exception as e:
            print(f"func: api_t2s, args: {text}, {speaker_id}, error: {str(e)}")
            traceback.print_exc()
            return {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": str(e)
            }
    
    def __api_t2s_bin(self, text: str, speaker_id: str, speed=1.0) -> dict:
        print(f"api_t2s_bin spk_name={speaker_id}, text={text}, pid={os.getpid()}, thread_id={threading.get_ident()}")

        if speaker_id not in self.speakers:
            return {
                "code": status.HTTP_400_BAD_REQUEST,
                "message": f"invalid speaker_id:{speaker_id}"
            }
        
        speaker_id = self.speakers[speaker_id]
        try:
            sr, wav_data = self._t2s(text=text, speaker_id=speaker_id, speed=speed)
            audio_io = BytesIO()
            with sf.SoundFile(audio_io, mode='w', format='WAV', channels=1, samplerate=sr) as f:
                f.write(wav_data)
            audio_binary = audio_io.getvalue()
            audio_io.close()

            return {
                "code": status.HTTP_200_OK,
                "data": base64.b64encode(audio_binary)
            }

        except Exception as e:
            print(f"func: __api_t2s_bin, args: {text}, {speaker_id}, error: {str(e)}")
            traceback.print_exc()
            return {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": str(e)
            }
    
    def __api_get_speakers(self) -> dict:
        try:
            speaker_list = list(self.speakers.keys())

            sorted(speaker_list)
            return {
                "code": status.HTTP_200_OK,
                "speakers": speaker_list
            }
        except Exception as e:
            print(f"func: __api_get_speakers, error: {str(e)}")
            return {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": str(e)
            }
