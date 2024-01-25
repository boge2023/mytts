import requests
import concurrent.futures
import os
import datetime
import base64
import soundfile as sf

base_url = "http://127.0.0.1:9000"


def load_test(text: str, spk_name: str, task_id: int):
    data = {
        "text": text,
        "speaker_id": spk_name
    }
    response = requests.post(f"{base_url}/tts/t2s_bin", json=data)
    base64.b64decode(response.json()['data'])


if __name__ == "__main__":
    
    num_threads = 16  # 指定线程数量
    task_num = 500
    speaker_list = ["zh_female_speaker_1", "zh_female_speaker_2"]
    text_list = [
        "让人遗憾的是，有家我以前常去的成衣店居然已经倒闭很久了。",
        "我还在想小可爱什么时候才会注意到这件新衣服呢，怎么样，你觉得好看吗？",
        "昨天有个可爱的红头发女孩子跳了一场舞，我就在前排哦，真是漂亮呀。",
        "那当然，先做那些麻烦的工作的话，连好吃的饭菜也会变味了吧。",
        "离开须弥之后，除了和恩师保持礼节性的通信，我和教令院已经没有什么往来了。幸好，不是每位被寄予了重望的弟子，都像我一样怠惰…听说那位师弟赛诺就相当活跃呢。",
        "唔？好奇怪的品味。送我这个，还不如一句让我成为守护你的剑来得实在呢"
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务给线程池
        start_ts = datetime.datetime.now()
        futures = [executor.submit(load_test, text_list[i % len(text_list)], speaker_list[i % len(speaker_list)], i) for i in range(task_num)]
        # 等待所有任务完成
        concurrent.futures.wait(futures)
        print(f"run {task_num} tasks cost {datetime.datetime.now() - start_ts} seconds") 