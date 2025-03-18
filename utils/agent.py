import os
import tiktoken
from openai import OpenAI, AzureOpenAI
from vertexai.preview.generative_models import GenerativeModel


class Agent():

    def __init__(
            self,
            initial_prompt_cls,
            round_prompt_cls,
            agent_name,
            temperature,
            model,
            model_seed=None,
            rounds_num=24,
            agents_num=6,
            azure=False,
            hf_models={},
            dry_run=False,
            dry_run_on_history=None,
        ):

        self.model = model
        self.agent_name = agent_name
        self.temperature = temperature
        self.model_seed = model_seed
        self.initial_prompt_cls = initial_prompt_cls
        self.rounds_num = rounds_num
        self.agents_num = agents_num
        self.round_prompt_cls = round_prompt_cls

        self.initial_prompt = initial_prompt_cls.return_initial_prompt()
        self.messages = [{"role": "user", "content": self.initial_prompt}]
        self.messages_history = {
            'inputs': [],
            'outputs': []
        }
        
        self.azure = azure

        self.dry_run = dry_run
        self.dry_run_on_history = dry_run_on_history
        
        if 'gemini' in self.model and not dry_run:
            assert not dry_run_on_history
            self.model_instance = GenerativeModel(model)

        if azure and not dry_run:
            assert not dry_run_on_history
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-05-15"
            )
        elif 'gpt' in model and not dry_run:
            assert not dry_run_on_history
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.hf_model = True if 'hf' in model else False
        if 'hf' in model and not dry_run:
            self.hf_model, self.hf_tokenizer, self.hf_pipeline_gen = hf_models[model]

        self.n_input_tokens = []
        self.n_output_tokens = []


    def execute_round(self, answer_history, round_idx):
        '''
        construct the prompt and call model
        '''        
        slot_prompt = self.round_prompt_cls.build_slot_prompt(answer_history,round_idx) 
        agent_response = self.prompt("user", slot_prompt)    
        return slot_prompt, agent_response

        
    def prompt(self, role, msg):
        '''
        call each model 
        '''
        if self.dry_run:
            print('DRY RUNNING')
            return '<DRYRUN></DRYRUN>'
        
        if 'gpt' in self.model:        
            messages = self.messages + [ {"role": role, "content": msg} ]
            self.messages_history['inputs'].append(messages)
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages,
                temperature=self.temperature,
                seed=self.model_seed if self.model_seed else None
            )
            content = response.choices[0].message.content
            self.messages_history['outputs'].append([ {"role": "assistant", "content": content} ])
            return content
        
        elif 'gemini' in self.model: 
            content = ''
            responses = self.model_instance.generate_content(
                self.initial_prompt + msg,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 1
                },
                stream = True
            )
            for response in responses:
                content += response.text
            return content
        
        elif self.hf_model or self.hf_model is None:
            chat = [{"role": "user", "content": self.initial_prompt + msg}]
            if not self.dry_run_on_history:
                model_input = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
                output_text = self.hf_pipeline_gen(model_input, do_sample=False, temperature = self.temperature)[0]['generated_text']
                return output_text
            model_input = self.hf_tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            self.n_input_tokens.append(model_input.size(1))
            output_text = next(self.dry_run_on_history)["full_answer"]
            self.n_output_tokens.append(len(self.hf_tokenizer.encode(output_text)))
            return output_text
        

    def num_tokens_from_messages(self, messages):
        """Returns the number of tokens used by a list of messages."""
        encoding = tiktoken.encoding_for_model(self.model)
        tokens_per_message = 3  # Base tokens per message
        tokens_per_name = 1     # Additional tokens if 'name' field is present
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Tokens for the assistant's reply
        return num_tokens
    

    def return_tokens(self):
        input_tokens, output_tokens = 0, 0
        models_cost = {
            'gpt-4o': {
                'input_cost': 2.5 / 1000000,
                'output_cost': 10 / 1000000
            },
            'gpt-4o-mini': {
                'input_cost': 0.15 / 1000000,
                'output_cost': 0.60 / 1000000
            },
            'gpt-4': {
                'input_cost': 30 / 1000000,
                'output_cost': 60 / 1000000
            },
        }
        for key, value in self.messages_history.items():
            if key == 'inputs':
                for messages in value:
                    input_tokens += self.num_tokens_from_messages(messages)
            else:
                for messages in value:
                    output_tokens += self.num_tokens_from_messages(messages)
        model_cost = models_cost[self.model]
        final_cost = (input_tokens * model_cost['input_cost']) + (output_tokens * model_cost['output_cost'])
        return input_tokens, output_tokens, final_cost
