from qwen_vl_utils import process_vision_info


def main():
    messages = (
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file:///root/chenkeqin.ckq/public/self-collect/VID_20240827183600471.mp4",
                    },
                    {
                        "type": "video",
                        "video": "file:///root/chenkeqin.ckq/public/self-collect/youtube_1.mp4",
                    },
                    {
                        "type": "video",
                        "video": "http://ofasys-multimodal-wlcb-3.oss-cn-wulanchabu.aliyuncs.com/data/bilibili/videos/crawler/mit_bilibili/14787505/BV11E41127r7.mp4?OSSAccessKeyId=LTAI5tGp88P39pDZvbnF2qKM&Expires=1727174668&Signature=LHxiGfAAsI6%2Bho%2Btad4MyLWsm04%3D",
                    },
                    {
                        "type": "video",
                        "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            },
        ],
    )
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import time

    st = time.time()
    image_inputs, video_inputs = process_vision_info(messages)
    print(f"Time: {time.time() - st:.3f}s")
    print([video_input.shape for video_input in video_inputs])


if __name__ == "__main__":
    main()
