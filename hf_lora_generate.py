import os
import sys
import pandas as pd
import argparse
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# CHAT_PROMPT = "[source sentence]\n\nTranslate to {TGT}"
# SYSTEM_MESSAGE = "You are a machine translation system that translates sentences from {SRC} to {TGT}. You just respond with the translation, without any additional comments."


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "",
    test_file: str = "",
    output_file: str = "",
    batch_size: int = 8,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", trust_remote_code=True)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if isinstance(model, LlamaForCausalLM):
            # unwind broken decapoda-research config
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )



    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0,
        top_p=1,
        top_k=50,
        num_beams=1,
        max_length=1024,
        do_sample=False,
        stream_output=False,
        **kwargs,
    ):
        prompts = [chat_prompt.replace("[source sentence]", src) for src in instruction]
        prompts = [prompter.generate_prompt(prompt, system_message=system_message) for prompt in prompts]

        inputs = tokenizer(prompts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
        generation_config = GenerationConfig(
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )

        generate_params = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<0x0A>"])[0]],
            "pad_token_id": tokenizer.pad_token_id,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            # "max_length": max_length,
            "max_new_tokens": 400,
        }

        with torch.no_grad():
            # generation_output = model.generate(
            #     input_ids=input_ids,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     max_new_tokens=max_new_tokens,
            # )
            generation_output = model.generate(
                **generate_params
            )
            
        outputs = generation_output.sequences
        res = []
        for k in range(len(outputs)):
            res.append(tokenizer.decode(outputs[k][len(inputs.input_ids[k]):], skip_special_tokens=True))
        
        return res

    test_data = pd.read_csv(test_file)
    test_data = test_data.assign(mt="")
    # pick up at where it is lost
    existed_results = None    
    if os.path.exists(output_file):
        existed_results = pd.read_csv(output_file)
        test_data = existed_results
    
    total_len = len(test_data)
    step_size = total_len // batch_size
    for i in tqdm(range(step_size + 1)):
        start_idx = i * batch_size
        if (i + 1) * batch_size > total_len:
            end_idx = total_len
        else:
            end_idx = (i + 1) * batch_size
            
        # pick up at where it is lost
        if existed_results is not None:
            skip = True
            for j in range(start_idx, end_idx):
                # mt不存在 
                if pd.isna(existed_results.loc[j, 'mt']):
                    start_idx = j
                    skip = False
                    print(f"************Resuming LLM for mt at idx: {start_idx}************")
                    existed_results = None
                    # 一旦检测到有mt不存在 说明后面都是不存在的
                    break
            if skip:
                continue
        outputs = evaluate(test_data[start_idx: end_idx]['src'])
        for k in range(len(outputs)):
            zeroshot_mt = outputs[k]
            # print(outputs[k])
            test_data.at[k + start_idx, 'mt'] = zeroshot_mt
        test_data.to_csv(output_file, index=None)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--template_name', default="llama2_chat", type=str,
                        help="template name used for inference")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument('--test_file', default=None, type=str, required=True,
                        help="path to the test file")
    parser.add_argument('--output_file', default=None, type=str, required=True,
                        help="path to the output file")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lang_pair', default=None, type=str, required=True)
    args = parser.parse_args()
    src, tgt = args.lang_pair.split('-')
    if args.base_model:
        # prepare the Chat prompt
        if 'Llama' in args.base_model:
            CHAT_PROMPT = "[source sentence]\n\nTranslate to {TGT}"
            SYSTEM_MESSAGE = "You are a machine translation system that translates sentences from {SRC} to {TGT}. You just respond with the translation, without any additional comments."
        elif 'bloom' in args.base_model:
            CHAT_PROMPT = "Translate to {TGT}: [source sentence] Translation:"
            SYSTEM_MESSAGE = None
        elif 'chatglm' in args.base_model:
            CHAT_PROMPT = "[source sentence]\n\nTranslate to {TGT}"
            SYSTEM_MESSAGE = "You are a machine translation system that translates sentences from {SRC} to {TGT}. You just respond with the translation, without any additional comments."
        elif 'vicuna' in args.base_model:
            CHAT_PROMPT = "[source sentence]\n\nTranslate to {TGT}"
            SYSTEM_MESSAGE = "I'm a machine translation system that translates sentences from {SRC} to {TGT}. I just respond with the translation, without any additional comments."
        else:
            raise TypeError
    mapping = {"eng": "English", "en": "English", "de": "German", "fr": "French", "zh": "Chinese", "ru": "Russian", "es": "Spanish", 
                "ces": "Czech", "ara": "Arabic", "ell": "Greek", "jpn": "Japanese", "afr": "Afrikaans", "oci": "Occitan", "swh": "Swahili"}
    chat_prompt = CHAT_PROMPT.format_map({"TGT": mapping[tgt]})
    if SYSTEM_MESSAGE:
        system_message = SYSTEM_MESSAGE.format(SRC = mapping[src], TGT = mapping[tgt])
    else:
        system_message = None
    if "70b" not in args.base_model:
        main(args.load_8bit, args.base_model, args.lora_weights, args.template_name, args.test_file, args.output_file, args.batch_size)
    else:
        main(True, args.base_model, args.lora_weights, args.template_name, args.test_file, args.output_file, args.batch_size)
        
