import torch
import accelerate
import transformers
import json
from typing import Any
import time 

MODEL_NAME = 'tiiuae/falcon-7b-instruct' #CasualLM Arch
DEFAULT_MAX_LENGTH = 128

class Model:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.pipeline = None
        self.model = None
    
    def generate_device_map(self):
        # Get Device MAP
        config=transformers.AutoConfig.from_pretrained(MODEL_NAME)
        with accelerate.init_empty_weights():
           fake_model = transformers.AutoModelForCausalLM.from_config(config)
        device_map = accelerate.infer_auto_device_map(fake_model,max_memory={0:"3GiB","cpu":"6GiB"})
        print(json.dumps(device_map,indent=4))
        open('device_map.json','w').write(json.dumps(device_map,indent=4))

    def load(self):
        device_map = json.load(open('device_map.json'))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            offload_folder="/tmp/.offload",
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            trust_remote_code=True
        )
        self.model.eval()
        self.pipeline = transformers.pipeline(
            task="text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            trust_remote_code=True,
            max_new_tokens=100,
            repetition_penalty=1.1,
            model_kwargs={"device_map": "auto", 
                          "max_length": 1200, "temperature": 0.01, "torch_dtype":torch.bfloat16}
        )
        return self.pipeline

    def predict(self, question):
        with torch.no_grad():
            data = self.pipeline(question,max_length=DEFAULT_MAX_LENGTH)[0] #max_length=DEFAULT_MAX_LENGTH)[0]
            print('[BOT] :',data['generated_text'].split('\n')[1])
 
if __name__ == '__main__':
    llm = Model()
    # llm.generate_device_map()
    llm.load()
    question=input('[USER] : ')
    llm.predict(question)
    
 
