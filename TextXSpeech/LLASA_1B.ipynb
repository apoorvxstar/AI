pip install xcodec2

from huggingface_hub import notebook_login
notebook_login()

import torch
import soundfile as sf
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model

# Load the Llasa-1B model and tokenizer
llasa_1b = 'HKUSTAudio/Llasa-1B'
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
model.eval().to('cuda')

# Load the XCodec2 model for audio encoding/decoding
codec_model_path = "HKUSTAudio/xcodec2"
codec_model = XCodec2Model.from_pretrained(codec_model_path)
codec_model.eval().to('cuda')



# Helper function to convert speech IDs to tokens
def ids_to_speech_tokens(speech_ids):
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

# Helper function to extract speech IDs from tokens
def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                print(f"Invalid speech token: {token_str}")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

# Function to generate speech adapted to a voice sample
def generate_speech_with_voice_sample(voice_sample_path, text, output_path="gen.wav"):
    """
    Generate speech from text using a voice sample to adapt the voice.
    
    Args:
        voice_sample_path (str): Path to the voice sample WAV file (preferably a few seconds long).
        text (str): The text to convert to speech.
        output_path (str): Path to save the generated audio (default: "gen.wav").
    """
    # Load and preprocess the voice sample
    waveform, sample_rate = torchaudio.load(voice_sample_path)
    if waveform.size(0) > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:  # Resample to 16kHz
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    prompt_wav = waveform.to('cuda')

    # Encode the voice sample into speech tokens
    with torch.no_grad():
        vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav)
        vq_code_prompt = vq_code_prompt[0, 0, :].tolist()  # Extract token IDs as a list
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    # Format the input text and create chat structure
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
    ]
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt', continue_final_message=True).to('cuda')

    # Generate speech tokens
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=10000,
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )

    # Extract the generated speech tokens (including prefix for full audio)
    generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix):-1]
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_tokens = extract_speech_ids(speech_tokens)
    speech_tokens = torch.tensor(speech_tokens).unsqueeze(0).unsqueeze(0).to('cuda')

    # Decode speech tokens to audio waveform
    with torch.no_grad():
        gen_wav = codec_model.decode_code(speech_tokens)

    # Save the generated audio
    sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
    print(f"Generated audio saved to {output_path}")

# Example usage
voice_sample_path = "Arjun.wav"  # Replace with your voice sample path
text = """In the Mahabharata War, Arjuna was a key warrior from the Pandava side in the battle of Kurukshetra. Before the beginning of the war, his mentor Krishna gave him the supreme knowledge of the Bhagavad Gita, guiding him through his moral dilemmas. Throughout the epic, Arjuna is the closest friend and companion of Krishna."""

generate_speech_with_voice_sample(voice_sample_path, text)
