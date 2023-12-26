from transformers import AutoProcessor, AutoModel
import torch
import pandas as pd
from tqdm.auto import tqdm

df = pd.read_csv('/data/ephemeral/home/becky/data/train_dev.csv')

df['translation_1'] = 'None'
df['translation_2'] = 'None'

sen1_list = df['sentence_1']
sen2_list = df['sentence_2']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
model = AutoModel.from_pretrained("facebook/hf-seamless-m4t-large")
# Move the model to the designated device (GPU or CPU)
model.to(device)


def ko2en2ko_translator(kor_text : str, processor, model, device):
    # Process the input text
    text_inputs = processor(text = kor_text, src_lang = "kor", return_tensors = "pt")

    # Move input tensors to the same device as the model
    text_inputs = {key: tensor.to(device) for key, tensor in text_inputs.items()}

    # Ensure that all model parameters and buffers are on the same device
    model = model.to(device)

    # Generate the translation
    try:
        output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
    except RuntimeError as e:
        #print(f"Runtime Error: {e}")
        #print("Trying to move model and inputs to CPU for compatibility.")
        device = torch.device("cpu")
        model.to(device)
        text_inputs = {key: tensor.to(device) for key, tensor in text_inputs.items()}
        output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)

    # Flatten the output tokens and decode the first sequence to get the translated text
    flat_output_tokens = output_tokens[0].flatten().tolist()
    eng_translated_text = processor.decode(flat_output_tokens, skip_special_tokens=True)

    # Print the translated text
    #print(f"Translation from text: {eng_translated_text}")

    text_inputs = processor(text= eng_translated_text, src_lang="eng", return_tensors="pt")
    text_inputs = {key: tensor.to(device) for key, tensor in text_inputs.items()}
    model = model.to(device)
    try:
        output_tokens = model.generate(**text_inputs, tgt_lang="kor", generate_speech=False)
    except RuntimeError as e:
      #  print(f"Runtime Error: {e}")
      #  print("Trying to move model and inputs to CPU for compatibility.")
        device = torch.device("cpu")
        model.to(device)
        text_inputs = {key: tensor.to(device) for key, tensor in text_inputs.items()}
        output_tokens = model.generate(**text_inputs, tgt_lang="kor", generate_speech=False)

    # Flatten the output tokens and decode the first sequence to get the translated text
    flat_output_tokens = output_tokens[0].flatten().tolist()
    kor_translated_text = processor.decode(flat_output_tokens, skip_special_tokens=True)

    # Print the translated text
    #print(f"Translation from text: {kor_translated_text}")
    return kor_translated_text 

test_text = "나는지금코드를보고있는중이다."
print(ko2en2ko_translator(test_text, processor, model, device))
# >>> 지금 코드를 보고 있어요



for idx, sentence in enumerate(tqdm(sen1_list)):
   # print("[original] ", sentence)
    result =  ko2en2ko_translator(sentence, processor, model, device)
    df.loc[idx,'translation_1'] = result

print(">>>>>>>FINISH SENTENCE_1>>>>>>>")
# save
df.to_csv("/data/ephemeral/home/becky/data/meta/meta_translation_ver1.csv", index = False)

for idx, sentence in enumerate(tqdm(sen2_list)):
   # print("[original] ", sentence)
    result =  ko2en2ko_translator(sentence, processor, model, device)
    df.loc[idx,'translation_2'] = result

print(">>>>>>>FINISH SENTENCE_2>>>>>>>")
df.to_csv("/data/ephemeral/home/becky/data/meta/meta_translation_ver1.csv", index = False)
